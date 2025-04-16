import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),  # 添加层归一化
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.GELU(),  # 使用 GELU 激活函数代替 ReLU
        ) 
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']
        pooler_out = bert_out['pooler_output']
        
        return self.trans(hidden_state), self.trans(pooler_out)


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )

        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],
            nn.Flatten()
        )

        # 获取特征响应图: (batch, 2048, 7, 7) -> (batch, img_hidden_seq, middle_hidden_size)
        self.hidden_trans = nn.Sequential(
            nn.BatchNorm2d(self.full_resnet.fc.in_features),  # 添加批归一化
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),
            nn.GELU(),  # 使用 GELU 激活函数
        )

        self.trans = nn.Sequential(
            nn.LayerNorm(self.full_resnet.fc.in_features),  # 添加层归一化
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.GELU(),  # 使用 GELU 激活函数
        )
        
        # 是否进行fine-tune - 只微调后面几层，减少过拟合
        for i, param in enumerate(self.full_resnet.parameters()):
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                # 只微调后面几层
                if i > len(list(self.full_resnet.parameters())) - 30:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)
        feature = self.resnet_p(hidden_state)

        return self.hidden_trans(hidden_state), self.trans(feature)


class GradualMultiHeadCrossAttention(nn.Module):
    """改进的多头交叉注意力层"""
    def __init__(self, config):
        super(GradualMultiHeadCrossAttention, self).__init__()
        # 减少注意力头的数量以降低模型复杂度
        self.attention_nhead = max(4, config.attention_nhead // 2)
        
        # 文本到图像的注意力
        self.text_to_img_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=self.attention_nhead, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.img_to_text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=self.attention_nhead, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm_text1 = nn.LayerNorm(config.middle_hidden_size)
        self.norm_text2 = nn.LayerNorm(config.middle_hidden_size)
        self.norm_img1 = nn.LayerNorm(config.middle_hidden_size)
        self.norm_img2 = nn.LayerNorm(config.middle_hidden_size)
        
        # 前馈网络 - 使用更小的维度扩展
        self.feedforward_text = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.middle_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size)
        )
        
        self.feedforward_img = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.middle_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size)
        )
        
        # 特征融合 - 使用门控机制
        self.text_gate = nn.Sequential(
            nn.Linear(config.middle_hidden_size, 1),
            nn.Sigmoid()
        )
        self.img_gate = nn.Sequential(
            nn.Linear(config.middle_hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size),
            nn.LayerNorm(config.middle_hidden_size),
            nn.Dropout(min(0.1, config.fuse_dropout / 2)),  # 减少 dropout 比例
            nn.GELU()
        )
        
    def forward(self, text_hidden, img_hidden):
        # 文本到图像的注意力
        text_img_attn, _ = self.text_to_img_attention(
            query=text_hidden,
            key=img_hidden,
            value=img_hidden
        )
        text_residual = self.norm_text1(text_hidden + text_img_attn)
        text_output = self.norm_text2(text_residual + self.feedforward_text(text_residual))
        
        # 图像到文本的注意力
        img_text_attn, _ = self.img_to_text_attention(
            query=img_hidden,
            key=text_hidden,
            value=text_hidden
        )
        img_residual = self.norm_img1(img_hidden + img_text_attn)
        img_output = self.norm_img2(img_residual + self.feedforward_img(img_residual))
        
        # 使用加权池化获取特征表示
        text_weights = self.text_gate(text_output)
        img_weights = self.img_gate(img_output)
        
        text_feature = (text_output * text_weights).sum(dim=1) / (text_weights.sum(dim=1) + 1e-8)
        img_feature = (img_output * img_weights).sum(dim=1) / (img_weights.sum(dim=1) + 1e-8)
        
        # 特征融合
        combined = torch.cat([text_feature, img_feature], dim=1)
        fused = self.fusion(combined)
        
        return text_output, img_output, fused


class EfficientTransformerFusion(nn.Module):
    """更高效的Transformer融合层"""
    def __init__(self, config):
        super(EfficientTransformerFusion, self).__init__()
        # 减少Transformer层数
        self.num_layers = 1
        
        # 使用更高效的Transformer配置
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size,
            nhead=max(4, config.attention_nhead // 2),  # 减少注意力头数
            dropout=min(0.1, config.attention_dropout / 2),  # 降低dropout比例
            dim_feedforward=config.middle_hidden_size * 2,  # 减小前馈网络维度
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=self.num_layers
        )
        
        # 添加辅助分类器 - 帮助特征学习
        self.aux_classifier = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.num_labels),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, text_features, img_features):
        # 将文本和图像特征序列拼接
        combined = torch.cat([text_features, img_features], dim=1)
        
        # 通过Transformer处理
        output = self.transformer(combined)
        
        # 使用注意力池化
        avg_pooled = torch.mean(output, dim=1)
        max_pooled, _ = torch.max(output, dim=1)
        fused_feature = avg_pooled + max_pooled  # 结合平均池化和最大池化
        
        # 辅助分类器输出 (训练时可用于辅助损失)
        aux_logits = self.aux_classifier(fused_feature)
        
        return fused_feature, aux_logits


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 文本模型
        self.text_model = TextModel(config)
        # 图像模型
        self.img_model = ImageModel(config)
        
        # 改进的多头交叉注意力层
        self.cross_attention = GradualMultiHeadCrossAttention(config)
        
        # 高效Transformer融合层
        self.transformer_fusion = EfficientTransformerFusion(config)
        
        # 残差连接
        self.use_residual = True
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 3, config.out_hidden_size),
            nn.LayerNorm(config.out_hidden_size),
            nn.Dropout(min(0.2, config.fuse_dropout)),
            nn.GELU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        
        # 使用标签平滑的交叉熵损失，缓解过拟合
        self.label_smoothing = 0.1
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            weight=torch.tensor(config.loss_weight) if hasattr(config, 'loss_weight') and config.loss_weight else None
        )
        
        # 辅助损失权重
        self.aux_loss_weight = 0.2

    def forward(self, texts, texts_mask, imgs, labels=None):
        # 获取文本特征
        text_hidden_state, text_pooled = self.text_model(texts, texts_mask)
        
        # 获取图像特征
        img_hidden_state, img_pooled = self.img_model(imgs)
        
        # 使用改进的多头交叉注意力处理
        text_refined, img_refined, cross_features = self.cross_attention(text_hidden_state, img_hidden_state)
        
        # 使用高效Transformer进一步融合特征
        transformer_features, aux_logits = self.transformer_fusion(text_refined, img_refined)
        
        # 残差连接: 加入原始pooled特征
        if self.use_residual:
            final_features = torch.cat([cross_features, transformer_features, text_pooled + img_pooled], dim=1)
        else:
            final_features = torch.cat([cross_features, transformer_features, torch.zeros_like(text_pooled)], dim=1)
        
        # 特征融合
        fused_features = self.feature_fusion(final_features)
        
        # 分类
        logits = self.classifier(fused_features)
        prob_vec = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            # 主损失
            main_loss = self.loss_func(logits, labels)
            
            # 辅助损失
            aux_loss = nn.functional.nll_loss(aux_logits, labels)
            
            # 总损失 = 主损失 + 辅助损失*权重
            loss = main_loss + self.aux_loss_weight * aux_loss
            
            return pred_labels, loss
        else:
            return pred_labels
