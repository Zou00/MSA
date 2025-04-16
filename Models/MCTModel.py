import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),  
            nn.Dropout(config.bert_dropout * 1.2),  # 增加dropout比例
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.GELU(),
        ) 
        
        # 只微调BERT的最后几层，前面的层冻结
        for i, param in enumerate(self.bert.parameters()):
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                # 只微调最后4层
                if i >= len(list(self.bert.parameters())) - 40:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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

        # 获取特征响应图
        self.hidden_trans = nn.Sequential(
            nn.BatchNorm2d(self.full_resnet.fc.in_features),
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout * 1.2),  # 增加dropout比例
            nn.Linear(7 * 7, config.middle_hidden_size),
            nn.GELU(),
        )

        self.trans = nn.Sequential(
            nn.LayerNorm(self.full_resnet.fc.in_features),
            nn.Dropout(config.resnet_dropout * 1.2),  # 增加dropout比例
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.GELU(),
        )
        
        # 完全冻结ResNet所有参数
        for param in self.full_resnet.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)
        feature = self.resnet_p(hidden_state)

        return self.hidden_trans(hidden_state), self.trans(feature)


class AttentionPooling(nn.Module):
    """注意力加权池化层"""
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        key = self.key_proj(hidden_states)
        
        # 计算注意力分数
        scores = torch.matmul(key, self.query) / (self.query.norm() * key.norm(dim=-1) + 1e-8)
        
        # 应用softmax获取权重
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        
        # 加权sum
        pooled = torch.sum(hidden_states * weights, dim=1)
        
        return pooled


class LightWeightCrossAttention(nn.Module):
    """轻量级多头交叉注意力层"""
    def __init__(self, config):
        super(LightWeightCrossAttention, self).__init__()
        # 降低复杂度
        self.attention_nhead = 4
        
        # 降维投影 - 减少计算量
        self.dim_reduction = config.middle_hidden_size // 2
        
        # 投影层
        self.text_proj = nn.Linear(config.middle_hidden_size, self.dim_reduction)
        self.img_proj = nn.Linear(config.middle_hidden_size, self.dim_reduction)
        
        # 注意力层
        self.text_to_img_attention = nn.MultiheadAttention(
            embed_dim=self.dim_reduction,
            num_heads=self.attention_nhead, 
            dropout=0.1,
            batch_first=True
        )
        self.img_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.dim_reduction,
            num_heads=self.attention_nhead, 
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.norm_text1 = nn.LayerNorm(self.dim_reduction)
        self.norm_text2 = nn.LayerNorm(self.dim_reduction)
        self.norm_img1 = nn.LayerNorm(self.dim_reduction)
        self.norm_img2 = nn.LayerNorm(self.dim_reduction)
        
        # 简化的前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(self.dim_reduction, self.dim_reduction),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim_reduction, self.dim_reduction)
        )
        
        # 池化层
        self.text_pool = AttentionPooling(self.dim_reduction)
        self.img_pool = AttentionPooling(self.dim_reduction)
        
        # 投影回原始维度
        self.out_proj = nn.Linear(self.dim_reduction * 2, config.middle_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.norm_out = nn.LayerNorm(config.middle_hidden_size)
        
    def forward(self, text_hidden, img_hidden):
        # 降维处理
        text_reduced = self.text_proj(text_hidden)
        img_reduced = self.img_proj(img_hidden)
        
        # 文本到图像的注意力
        text_img_attn, _ = self.text_to_img_attention(
            query=text_reduced,
            key=img_reduced,
            value=img_reduced
        )
        text_residual = self.norm_text1(text_reduced + text_img_attn)
        text_output = self.norm_text2(text_residual + self.feedforward(text_residual))
        
        # 图像到文本的注意力
        img_text_attn, _ = self.img_to_text_attention(
            query=img_reduced,
            key=text_reduced,
            value=text_reduced
        )
        img_residual = self.norm_img1(img_reduced + img_text_attn)
        img_output = self.norm_img2(img_residual + self.feedforward(img_residual))
        
        # 注意力池化
        text_feature = self.text_pool(text_output)
        img_feature = self.img_pool(img_output)
        
        # 特征融合并投影回原始维度
        combined = torch.cat([text_feature, img_feature], dim=1)
        fused = self.dropout(self.out_proj(combined))
        fused = self.norm_out(fused)
        
        return text_output, img_output, fused


class SimplifiedTransformerFusion(nn.Module):
    """简化版的Transformer融合层"""
    def __init__(self, config):
        super(SimplifiedTransformerFusion, self).__init__()
        # 降低复杂度，只用一层
        self.dim_reduction = config.middle_hidden_size // 2
        
        # 投影层
        self.proj = nn.Linear(config.middle_hidden_size, self.dim_reduction)
        
        # 简化的自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.dim_reduction,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.dim_reduction)
        self.norm2 = nn.LayerNorm(self.dim_reduction)
        
        # 简化的前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(self.dim_reduction, self.dim_reduction),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim_reduction, self.dim_reduction)
        )
        
        # 池化层
        self.pool = AttentionPooling(self.dim_reduction)
        
        # 输出层
        self.output_proj = nn.Linear(self.dim_reduction, config.middle_hidden_size)
        
    def forward(self, text_features, img_features):
        # 投影降维
        text_reduced = self.proj(text_features)
        img_reduced = self.proj(img_features)
        
        # 合并特征
        combined = torch.cat([text_reduced, img_reduced], dim=1)
        
        # 自注意力处理
        attn_output, _ = self.self_attention(combined, combined, combined)
        residual = self.norm1(combined + attn_output)
        output = self.norm2(residual + self.feedforward(residual))
        
        # 池化并投影回原始维度
        pooled = self.pool(output)
        final_output = self.output_proj(pooled)
        
        return final_output


class WeightedEnsemble(nn.Module):
    """权重融合不同特征的集成模块"""
    def __init__(self, hidden_size, num_features=3):
        super(WeightedEnsemble, self).__init__()
        self.feature_weights = nn.Parameter(torch.ones(num_features) / num_features)
        self.scale_proj = nn.Linear(hidden_size * num_features, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, *features):
        # 应用特征权重
        weighted_features = [w * f for w, f in zip(F.softmax(self.feature_weights, dim=0), features)]
        
        # 合并特征
        concat_features = torch.cat(weighted_features, dim=1)
        output = self.scale_proj(concat_features)
        output = self.norm(output)
        
        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 文本模型
        self.text_model = TextModel(config)
        # 图像模型
        self.img_model = ImageModel(config)
        
        # 轻量级交叉注意力
        self.cross_attention = LightWeightCrossAttention(config)
        
        # 简化的Transformer融合
        self.transformer_fusion = SimplifiedTransformerFusion(config)
        
        # 特征集成
        self.feature_ensemble = WeightedEnsemble(config.middle_hidden_size)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.LayerNorm(config.out_hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        
        # 使用标签平滑的交叉熵损失，缓解过拟合
        self.label_smoothing = 0.2  # 增加标签平滑
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            weight=torch.tensor(config.loss_weight) if hasattr(config, 'loss_weight') and config.loss_weight else None
        )
        
    def forward(self, texts, texts_mask, imgs, labels=None):
        # 获取文本特征
        text_hidden_state, text_pooled = self.text_model(texts, texts_mask)
        
        # 获取图像特征
        img_hidden_state, img_pooled = self.img_model(imgs)
        
        # 使用轻量级交叉注意力处理
        _, _, cross_features = self.cross_attention(text_hidden_state, img_hidden_state)
        
        # 使用简化的Transformer融合特征
        transformer_features = self.transformer_fusion(text_hidden_state, img_hidden_state)
        
        # 集成原始池化特征和增强特征
        fused_features = self.feature_ensemble(cross_features, transformer_features, text_pooled * 0.7 + img_pooled * 0.3)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 应用温度缩放使预测更加平滑
        temperature = 1.2
        scaled_logits = logits / temperature
        
        prob_vec = torch.softmax(scaled_logits, dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            # 计算损失，包括权重衰减以减轻过拟合
            loss = self.loss_func(scaled_logits, labels)
            
            # 添加L2正则化
            l2_lambda = 1e-4
            l2_reg = 0.
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            loss += l2_lambda * l2_reg
            
            return pred_labels, loss
        else:
            return pred_labels
