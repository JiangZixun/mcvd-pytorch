import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalRefinementNet(nn.Module):
    def __init__(self, channels=8, hidden_dim=32):
        super().__init__()
        # 空间平滑
        self.spatial_smooth = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2),  # 较大卷积核平滑
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1)
        )
        
        # 时间平滑 (1D卷积在时间维度)
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 3, padding=1)
        )
        
        # 自适应权重
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        original = x
        
        # 空间平滑
        x_spatial = x.view(B*T, C, H, W)
        x_spatial = self.spatial_smooth(x_spatial)
        x_spatial = x_spatial.view(B, T, C, H, W)
        
        # 时间平滑
        x_temporal = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, C, H, W, T]
        x_temporal = x_temporal.view(B*C*H*W, T)
        x_temporal = self.temporal_smooth(x_temporal.unsqueeze(1)).squeeze(1)
        x_temporal = x_temporal.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
        
        # 自适应融合
        weight = self.weight_net(x.view(B*T, C, H, W)).view(B, T, 1, H, W)
        refined = weight * x_spatial + (1 - weight) * x_temporal
        
        # 残差连接
        output = refined + original
        
        return output

class LightweightRefinementNet(nn.Module):
    def __init__(self, channels=8):
        super(LightweightRefinementNet, self).__init__()
        # 只用一个深度可分离卷积
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习的融合权重
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        shape = x.shape
        original = x
        
        if len(shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
        
        # 深度可分离卷积平滑
        x = self.depthwise(x)
        x = self.pointwise(x)

        if len(shape) == 5:
            x = x.view(B, T, C, H, W)
        
        # 可学习的残差连接
        return original + self.alpha * (x - original)
    