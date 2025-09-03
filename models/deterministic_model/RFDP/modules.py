from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.nn.init as init

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
    

class CrossAttention(nn.Module):
    """
    Cross Attention模块实现
    Query来自一个序列, Key和Value来自另一个序列
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        scale: Optional[float] = None
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout率
            bias: 是否使用偏置
            scale: 缩放因子，默认为1/sqrt(d_k)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = scale or (self.head_dim ** -0.5)
        
        # 线性投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            query: [batch_size, tgt_len, d_model] 或 [tgt_len, batch_size, d_model]
            key: [batch_size, src_len, d_model] 或 [src_len, batch_size, d_model]
            value: [batch_size, src_len, d_model] 或 [src_len, batch_size, d_model]
            key_padding_mask: [batch_size, src_len] - True表示padding位置
            attn_mask: [tgt_len, src_len] 或 [batch_size * num_heads, tgt_len, src_len]
            need_weights: 是否返回注意力权重
            average_attn_weights: 是否平均所有头的注意力权重
            
        Returns:
            attn_output: [batch_size, tgt_len, d_model]
            attn_weights: [batch_size, num_heads, tgt_len, src_len] (如果need_weights=True)
        """
        # 检查输入维度
        if query.dim() == 3:
            # 假设是[batch_size, seq_len, d_model]格式
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            raise ValueError("The input's shape must be like [B, L, D].")
        
        # 线性投影
        Q = self.q_proj(query)  # [batch_size, tgt_len, d_model]
        K = self.k_proj(key)    # [batch_size, src_len, d_model]
        V = self.v_proj(value)  # [batch_size, src_len, d_model]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # 形状: [batch_size, num_heads, tgt_len, src_len]
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # 扩展为[batch_size, num_heads, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                # [batch_size, tgt_len, src_len] -> [batch_size, num_heads, tgt_len, src_len]
                attn_mask = attn_mask.unsqueeze(1)
            
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        context = torch.matmul(attn_weights, V)
        # 形状: [batch_size, num_heads, tgt_len, head_dim]
        
        # 重塑回原始形状
        context = context.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.d_model
        )
        
        # 输出投影
        attn_output = self.out_proj(context)
        
        if need_weights:
            if average_attn_weights:
                # 平均所有头的注意力权重
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class CrossAttentionLayer(nn.Module):
    """
    完整的Cross Attention层, 包含残差连接和层归一化
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ):
        super().__init__()
        
        # Cross Attention
        self.cross_attn = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feedforward网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu
        self.norm_first = norm_first
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: 目标序列 [batch_size, tgt_len, d_model]
            memory: 源序列（编码器输出）[batch_size, src_len, d_model]
            tgt_mask: 目标序列的注意力掩码
            memory_mask: 源序列的注意力掩码
            tgt_key_padding_mask: 目标序列的padding掩码
            memory_key_padding_mask: 源序列的padding掩码
        """
        
        if self.norm_first:
            # Pre-norm架构
            # Cross attention block
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.cross_attn(
                query=tgt2,
                key=memory,
                value=memory,
                attn_mask=tgt_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt = tgt + self.dropout1(tgt2)
            
            # Feedforward block
            tgt2 = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout2(tgt2)
        else:
            # Post-norm架构
            # Cross attention block
            tgt2, _ = self.cross_attn(
                query=tgt,
                key=memory,
                value=memory,
                attn_mask=tgt_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            
            # Feedforward block
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        return tgt
    

class GatedFusion(nn.Module):
    def __init__(self, skip_channels, pred_channels):
        super().__init__()
        total_channels = skip_channels + pred_channels
        
        # 学习skip和pred的权重
        self.weight_net = nn.Sequential(
            nn.Conv2d(total_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),  # 输出2个权重
            nn.Softmax(dim=1)
        )
        
        # 原始的融合conv
        self.fusion_conv = nn.Conv2d(total_channels, pred_channels, 3, padding=1)
        
    def forward(self, skip_features, pred_features):
        concat_features = torch.cat([skip_features, pred_features], dim=1)
        
        # 计算权重
        weights = self.weight_net(concat_features)  # [B, 2, H, W]
        
        # 分离权重
        skip_weight = weights[:, 0:1, :, :]
        pred_weight = weights[:, 1:2, :, :]
        
        # 加权concat
        weighted_skip = skip_features * skip_weight
        weighted_pred = pred_features * pred_weight
        weighted_concat = torch.cat([weighted_skip, weighted_pred], dim=1)
        
        # 融合
        output = self.fusion_conv(weighted_concat)
        return output
    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

# 定义通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 定义空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class FusionModule(nn.Module):
    """
    融合模块，支持多种融合策略来结合velocity predictor和texture predictor的输出
    
    Args:
        channels: 输入通道数 (C)
        fusion_type: 融合类型，支持以下选项：
            - 'add': 简单元素级相加
            - 'weighted_add': 可学习权重相加
            - 'gate_control': 门控融合
            - 'softmax_weighted': 使用softmax权重融合
            - 'channel_attention': 通道注意力融合
            - 'spatial_attention': 空间注意力融合
            - 'adaptive': 自适应特征融合
            - 'residual': 残差融合
            - 'temporal_aware': 时序感知融合
        reduction: 注意力机制中的降维比例
    """
    
    def __init__(self, channels, fusion_type='add', reduction=4):
        super().__init__()
        self.fusion_type = fusion_type
        self.channels = channels
        
        if fusion_type == 'weighted_add':
            self.alpha = nn.Parameter(torch.tensor(0.5))
            
        elif fusion_type == 'gate_control':
            self.gate_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'softmax_weighted':
            self.weight_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, kernel_size=1)  # 输出2个权重通道
            )
            
        elif fusion_type == 'channel_attention':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels * 2, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'spatial_attention':
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'adaptive':
            # 特征变换层
            self.vp_transform = nn.Conv2d(channels, channels, kernel_size=1)
            self.tp_transform = nn.Conv2d(channels, channels, kernel_size=1)
            
            # 融合权重生成网络
            self.fusion_net = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, kernel_size=1)
            )
            
        elif fusion_type == 'residual':
            self.vp_proj = nn.Conv2d(channels, channels, kernel_size=1)
            self.tp_proj = nn.Conv2d(channels, channels, kernel_size=1)
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1)
            )
            
        elif fusion_type == 'temporal_aware':
            self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1),
                nn.Sigmoid()
            )
            
    def forward(self, vp_feat, tp_feat):
        """
        Args:
            vp_feat: velocity predictor输出特征 (B, T, C, H, W)
            tp_feat: texture predictor输出特征 (B, T, C, H, W)
            
        Returns:
            fused_feat: 融合后的特征 (B, T, C, H, W)
        """
        B, T, C, H, W = vp_feat.shape
        
        if self.fusion_type == 'add':
            return vp_feat + tp_feat
            
        elif self.fusion_type == 'weighted_add':
            return self.alpha * vp_feat + (1 - self.alpha) * tp_feat
            
        elif self.fusion_type == 'gate_control':
            # 转换为(B*T, C, H, W)进行卷积操作
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 生成门控信号
            combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
            gate = self.gate_conv(combined)
            
            # 门控融合
            fused = gate * vp_reshaped + (1 - gate) * tp_reshaped
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'softmax_weighted':
            # 转换维度
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 生成权重
            combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
            weights = self.weight_conv(combined)  # (B*T, 2, H, W)
            weights = F.softmax(weights, dim=1)
            
            w_vp = weights[:, 0:1, :, :].expand_as(vp_reshaped)
            w_tp = weights[:, 1:2, :, :].expand_as(tp_reshaped)
            
            fused = w_vp * vp_reshaped + w_tp * tp_reshaped
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'channel_attention':
            # 转换维度
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 全局平均池化
            vp_pool = self.global_pool(vp_reshaped).view(B*T, C)
            tp_pool = self.global_pool(tp_reshaped).view(B*T, C)
            
            # 计算通道注意力权重
            combined = torch.cat([vp_pool, tp_pool], dim=1)
            attention = self.fc(combined).view(B*T, C, 1, 1)
            
            # 加权融合
            fused = attention * vp_reshaped + (1 - attention) * tp_reshaped
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'spatial_attention':
            # 转换维度
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 计算空间注意力
            combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
            spatial_attention = self.spatial_conv(combined)
            
            # 加权融合
            fused = spatial_attention * vp_reshaped + (1 - spatial_attention) * tp_reshaped
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'adaptive':
            # 转换维度
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 特征变换
            vp_transformed = self.vp_transform(vp_reshaped)
            tp_transformed = self.tp_transform(tp_reshaped)
            
            # 生成自适应权重
            combined = torch.cat([vp_transformed, tp_transformed], dim=1)
            fusion_weights = self.fusion_net(combined)
            fusion_weights = F.softmax(fusion_weights, dim=1)
            
            w_vp = fusion_weights[:, 0:1, :, :].expand_as(vp_transformed)
            w_tp = fusion_weights[:, 1:2, :, :].expand_as(tp_transformed)
            
            fused = w_vp * vp_transformed + w_tp * tp_transformed
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'residual':
            # 转换维度
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            # 特征投影
            vp_proj = self.vp_proj(vp_reshaped)
            tp_proj = self.tp_proj(tp_reshaped)
            
            # 基础融合
            base = vp_proj + tp_proj
            refined = self.fusion_conv(base)
            
            # 残差连接
            fused = vp_proj + refined
            return fused.view(B, T, C, H, W)
            
        elif self.fusion_type == 'temporal_aware':
            # 时序特征提取
            vp_temp = vp_feat.mean(dim=[3, 4])  # (B, T, C)
            vp_temp = vp_temp.permute(0, 2, 1)  # (B, C, T)
            vp_temp = self.temporal_conv(vp_temp)  # (B, C, T)
            vp_temp = vp_temp.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
            
            # 转换维度进行门控计算
            vp_reshaped = vp_feat.view(B*T, C, H, W)
            tp_reshaped = tp_feat.view(B*T, C, H, W)
            
            combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
            gate = self.fusion_gate(combined)
            
            fused = gate * vp_reshaped + (1 - gate) * tp_reshaped
            return fused.view(B, T, C, H, W)
            
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")


# class FusionModule(nn.Module):
#     """
#     融合模块，支持多种融合策略来结合velocity predictor和texture predictor的输出
    
#     Args:
#         channels: 输入通道数 (C)
#         fusion_type: 融合类型，支持以下选项：
#             - 'add': 简单元素级相加
#             - 'weighted_add': 可学习权重相加
#             - 'gate_control': 门控融合
#             - 'softmax_weighted': 使用softmax权重融合
#             - 'channel_attention': 通道注意力融合
#             - 'spatial_attention': 空间注意力融合
#             - 'adaptive': 自适应特征融合
#             - 'residual': 残差融合
#             - 'temporal_aware': 时序感知融合
#         reduction: 注意力机制中的降维比例
#     """
    
#     def __init__(self, channels, fusion_type='add', reduction=4):
#         super().__init__()
#         self.fusion_type = fusion_type
#         self.channels = channels
        
#         if fusion_type == 'weighted_add':
#             self.alpha = nn.Parameter(torch.tensor(0.5))
            
#         elif fusion_type == 'gate_control':
#             self.gate_conv = nn.Sequential(
#                 nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, channels, kernel_size=1),
#                 nn.Sigmoid()
#             )
            
#         elif fusion_type == 'softmax_weighted':
#             self.weight_conv = nn.Sequential(
#                 nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, 2, kernel_size=1)  # 输出2个权重通道
#             )
            
#         elif fusion_type == 'channel_attention':
#             self.global_pool = nn.AdaptiveAvgPool3d(1)
#             self.fc = nn.Sequential(
#                 nn.Linear(channels * 2, channels // reduction),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channels // reduction, channels),
#                 nn.Sigmoid()
#             )
            
#         elif fusion_type == 'spatial_attention':
#             self.spatial_conv = nn.Sequential(
#                 nn.Conv3d(channels * 2, channels // reduction, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels // reduction, 1, kernel_size=3, padding=1),
#                 nn.Sigmoid()
#             )
            
#         elif fusion_type == 'adaptive':
#             # 特征变换层
#             self.vp_transform = nn.Conv3d(channels, channels, kernel_size=1)
#             self.tp_transform = nn.Conv3d(channels, channels, kernel_size=1)
            
#             # 融合权重生成网络
#             self.fusion_net = nn.Sequential(
#                 nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, 2, kernel_size=1)
#             )
            
#         elif fusion_type == 'residual':
#             self.vp_proj = nn.Conv3d(channels, channels, kernel_size=1)
#             self.tp_proj = nn.Conv3d(channels, channels, kernel_size=1)
#             self.fusion_conv = nn.Sequential(
#                 nn.Conv3d(channels, channels, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(channels, channels, kernel_size=1)
#             )
            
#         elif fusion_type == 'temporal_aware':
#             self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#             self.fusion_gate = nn.Sequential(
#                 nn.Conv3d(channels * 2, channels, kernel_size=1),
#                 nn.Sigmoid()
#             )

#         else:
#             raise ValueError(f"Unsupported fusion type: {fusion_type}")

#     def forward(self, vp_feat, tp_feat):
#         """
#         Args:
#             vp_feat: velocity predictor输出特征 (B, T, C, H, W)
#             tp_feat: texture predictor输出特征 (B, T, C, H, W)
            
#         Returns:
#             fused_feat: 融合后的特征 (B, T, C, H, W)
#         """
#         B, T, C, H, W = vp_feat.shape
        
#         if self.fusion_type == 'add':
#             return vp_feat + tp_feat
            
#         elif self.fusion_type == 'weighted_add':
#             return self.alpha * vp_feat + (1 - self.alpha) * tp_feat
            
#         elif self.fusion_type == 'gate_control':
#             # 转换为(B*T, C, H, W)进行卷积操作
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 生成门控信号
#             combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
#             gate = self.gate_conv(combined)
            
#             # 门控融合
#             fused = gate * vp_reshaped + (1 - gate) * tp_reshaped
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'softmax_weighted':
#             # 转换维度
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 生成权重
#             combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
#             weights = self.weight_conv(combined)  # (B*T, 2, H, W)
#             weights = F.softmax(weights, dim=1)
            
#             w_vp = weights[:, 0:1, :, :].expand_as(vp_reshaped)
#             w_tp = weights[:, 1:2, :, :].expand_as(tp_reshaped)
            
#             fused = w_vp * vp_reshaped + w_tp * tp_reshaped
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'channel_attention':
#             # 转换维度
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 全局平均池化
#             vp_pool = self.global_pool(vp_reshaped).view(B*T, C)
#             tp_pool = self.global_pool(tp_reshaped).view(B*T, C)
            
#             # 计算通道注意力权重
#             combined = torch.cat([vp_pool, tp_pool], dim=1)
#             attention = self.fc(combined).view(B*T, C, 1, 1)
            
#             # 加权融合
#             fused = attention * vp_reshaped + (1 - attention) * tp_reshaped
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'spatial_attention':
#             # 转换维度
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 计算空间注意力
#             combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
#             spatial_attention = self.spatial_conv(combined)
            
#             # 加权融合
#             fused = spatial_attention * vp_reshaped + (1 - spatial_attention) * tp_reshaped
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'adaptive':
#             # 转换维度
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 特征变换
#             vp_transformed = self.vp_transform(vp_reshaped)
#             tp_transformed = self.tp_transform(tp_reshaped)
            
#             # 生成自适应权重
#             combined = torch.cat([vp_transformed, tp_transformed], dim=1)
#             fusion_weights = self.fusion_net(combined)
#             fusion_weights = F.softmax(fusion_weights, dim=1)
            
#             w_vp = fusion_weights[:, 0:1, :, :].expand_as(vp_transformed)
#             w_tp = fusion_weights[:, 1:2, :, :].expand_as(tp_transformed)
            
#             fused = w_vp * vp_transformed + w_tp * tp_transformed
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'residual':
#             # 转换维度
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             # 特征投影
#             vp_proj = self.vp_proj(vp_reshaped)
#             tp_proj = self.tp_proj(tp_reshaped)
            
#             # 基础融合
#             base = vp_proj + tp_proj
#             refined = self.fusion_conv(base)
            
#             # 残差连接
#             fused = vp_proj + refined
#             return fused.view(B, T, C, H, W)
            
#         elif self.fusion_type == 'temporal_aware':
#             # 时序特征提取
#             vp_temp = vp_feat.mean(dim=[3, 4])  # (B, T, C)
#             vp_temp = vp_temp.permute(0, 2, 1)  # (B, C, T)
#             vp_temp = self.temporal_conv(vp_temp)  # (B, C, T)
#             vp_temp = vp_temp.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
            
#             # 转换维度进行门控计算
#             vp_reshaped = vp_feat.view(B*T, C, H, W)
#             tp_reshaped = tp_feat.view(B*T, C, H, W)
            
#             combined = torch.cat([vp_reshaped, tp_reshaped], dim=1)
#             gate = self.fusion_gate(combined)
            
#             fused = gate * vp_reshaped + (1 - gate) * tp_reshaped
#             return fused.view(B, T, C, H, W)
            
#         else:
#             raise ValueError(f"Unsupported fusion type: {self.fusion_type}")