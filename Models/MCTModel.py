import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50


class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
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
            nn.Conv2d(self.full_resnet.fc.in_features, config.img_hidden_seq, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 是否进行fine-tune
        for param in self.full_resnet.parameters():
            if config.fixed_image_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, imgs):
        hidden_state = self.resnet_h(imgs)
        feature = self.resnet_p(hidden_state)

        return self.hidden_trans(hidden_state), self.trans(feature)


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力层"""
    def __init__(self, config):
        super(MultiHeadCrossAttention, self).__init__()
        self.text_to_img_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.img_to_text_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nhead, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.middle_hidden_size)
        self.norm2 = nn.LayerNorm(config.middle_hidden_size)
        
        # 前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.middle_hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.middle_hidden_size * 4, config.middle_hidden_size)
        )
        
        # 最终特征融合
        self.fusion = nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size)
        self.dropout = nn.Dropout(config.fuse_dropout)
        
    def forward(self, text_hidden, img_hidden):
        # text_hidden: [batch_size, seq_len_text, hidden_size]
        # img_hidden: [batch_size, seq_len_img, hidden_size]
        
        # 文本到图像的注意力
        text_img_attn, _ = self.text_to_img_attention(
            query=text_hidden,
            key=img_hidden,
            value=img_hidden
        )
        text_residual = self.norm1(text_hidden + text_img_attn)
        text_output = self.norm1(text_residual + self.feedforward(text_residual))
        
        # 图像到文本的注意力
        img_text_attn, _ = self.img_to_text_attention(
            query=img_hidden,
            key=text_hidden,
            value=text_hidden
        )
        img_residual = self.norm2(img_hidden + img_text_attn)
        img_output = self.norm2(img_residual + self.feedforward(img_residual))
        
        # 获取各序列的特征表示
        text_feature = torch.mean(text_output, dim=1)
        img_feature = torch.mean(img_output, dim=1)
        
        # 特征融合
        combined = torch.cat([text_feature, img_feature], dim=1)
        fused = self.dropout(self.fusion(combined))
        
        return text_output, img_output, fused


class TransformerFusion(nn.Module):
    """基于Transformer的特征融合层"""
    def __init__(self, config):
        super(TransformerFusion, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size,
            nhead=config.attention_nhead,
            dropout=config.attention_dropout,
            dim_feedforward=config.middle_hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=2
        )
        
    def forward(self, text_features, img_features):
        # 将文本和图像特征序列拼接
        # text_features: [batch_size, seq_len_text, hidden_size]
        # img_features: [batch_size, seq_len_img, hidden_size]
        combined = torch.cat([text_features, img_features], dim=1)
        
        # 通过Transformer处理
        output = self.transformer(combined)
        
        # 平均池化获取特征表示
        return torch.mean(output, dim=1)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 文本模型
        self.text_model = TextModel(config)
        # 图像模型
        self.img_model = ImageModel(config)
        
        # 多头交叉注意力层
        self.cross_attention = MultiHeadCrossAttention(config)
        
        # Transformer融合层
        self.transformer_fusion = TransformerFusion(config)
        
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight) if hasattr(config, 'loss_weight') and config.loss_weight else None)

    def forward(self, texts, texts_mask, imgs, labels=None):
        # 获取文本特征
        text_hidden_state, text_pooled = self.text_model(texts, texts_mask)
        
        # 获取图像特征
        img_hidden_state, img_pooled = self.img_model(imgs)
        
        # 使用多头交叉注意力处理
        text_refined, img_refined, cross_features = self.cross_attention(text_hidden_state, img_hidden_state)
        
        # 使用Transformer进一步融合特征
        transformer_features = self.transformer_fusion(text_refined, img_refined)
        
        # 最终特征表示：交叉注意力特征 + Transformer融合特征
        final_features = torch.cat([cross_features, transformer_features], dim=1)
        
        # 分类
        logits = self.classifier(final_features)
        prob_vec = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return pred_labels, loss
        else:
            return pred_labels