import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from einops import rearrange

from ..Earthformer.cuboid_transformer import CuboidTransformerModel
from .modules import Inception

class AttentionBlock(nn.Module):
    """轻量化注意力模块"""
    def __init__(self, in_channels, gate_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    """带残差连接的轻量化卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out

class LightweightUNet(nn.Module):
    """
    轻量化Attention-UNet
    输入: 过去6帧 [B, t_in*C, H, W]
    输出: 6个速度场 [B, t_out, H, W, 2]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32):
        super().__init__()
        
        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.frame_channels = frame_channels
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层：预测t_out个时间步的速度场
        self.velocity_head = nn.Conv2d(base_channels, out_frames*frame_channels*2, 1)  # t_out个时间步 × 通道数量 × 2个速度分量
        
    def forward(self, x):
        """
        Args:
            x: [B, t_in, C, H, W] 拼接的t_in帧
        Returns:
            velocity_fields: [B, t_out, H, W, 2] t_out个速度场
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # 底层
        b = self.bottleneck(self.pool4(e4))
        
        # 解码器 + 注意力
        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        # 输出速度场
        velocity_fields = self.velocity_head(d1)  # [B, 12, H, W]
        
        # 重塑为 [B, 6, C, H, W, 2]
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, self.frame_channels, 2, H, W).permute(0, 1, 2, 4, 5, 3)
        
        return velocity_fields

class EarthformerUNet(nn.Module):
    
    def __init__(self, model_cfg):
        super().__init__()

        # Model Configs
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.model = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )
    
    def forward(self, x):
        """
        Parameters
        ----------
        x
            Shape (B, T, C, H, W)

        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        x = self.model(x)
        x = rearrange(x, 'b t c h w -> b t h w c')
        return x
    
class InceptionUNet(nn.Module):
    """
    基于Inception模块的UNet
    输入: 过去6帧 [B, t_in*C, H, W]
    输出: 6个速度场 [B, t_out, H, W, 2]
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=32, incep_ker=[3,5,7,11], groups=8):
        super().__init__()
        
        # 输入是t_in帧拼接
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        
        # 编码器 - 使用Inception模块
        self.enc1 = Inception(in_channels, base_channels//2, base_channels, incep_ker, groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = Inception(base_channels, base_channels, base_channels * 2, incep_ker, groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = Inception(base_channels * 2, base_channels * 2, base_channels * 4, incep_ker, groups)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = Inception(base_channels * 4, base_channels * 4, base_channels * 8, incep_ker, groups)
        self.pool4 = nn.MaxPool2d(2)
        
        # 底层 - 最深层的Inception模块
        self.bottleneck = Inception(base_channels * 8, base_channels * 8, base_channels * 16, incep_ker, groups)
        
        # 解码器 - 上采样 + Inception模块
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8)
        self.dec4 = Inception(base_channels * 16, base_channels * 8, base_channels * 8, incep_ker, groups)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = Inception(base_channels * 8, base_channels * 4, base_channels * 4, incep_ker, groups)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = Inception(base_channels * 4, base_channels * 2, base_channels * 2, incep_ker, groups)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = Inception(base_channels * 2, base_channels, base_channels, incep_ker, groups)
        
        # 输出头：使用1x1卷积生成最终的速度场
        self.velocity_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, out_frames * 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, t_in, C, H, W] 输入的t_in帧
        Returns:
            velocity_fields: [B, t_out, H, W, 2] t_out个速度场
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        # 编码器路径
        e1 = self.enc1(x)           # [B, base_channels, H, W]
        e2 = self.enc2(self.pool1(e1))     # [B, base_channels*2, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))     # [B, base_channels*4, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))     # [B, base_channels*8, H/8, W/8]
        
        # 底部瓶颈层
        b = self.bottleneck(self.pool4(e4))  # [B, base_channels*16, H/16, W/16]
        
        # 解码器路径 + 跳跃连接 + 注意力机制
        d4 = self.up4(b)                    # [B, base_channels*8, H/8, W/8]
        e4_att = self.att4(e4, d4)          # 注意力增强的编码器特征
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))  # [B, base_channels*8, H/8, W/8]
        
        d3 = self.up3(d4)                   # [B, base_channels*4, H/4, W/4]
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))  # [B, base_channels*4, H/4, W/4]
        
        d2 = self.up2(d3)                   # [B, base_channels*2, H/2, W/2]
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))  # [B, base_channels*2, H/2, W/2]
        
        d1 = self.up1(d2)                   # [B, base_channels, H, W]
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))  # [B, base_channels, H, W]
        
        # 生成速度场
        velocity_fields = self.velocity_head(d1)  # [B, out_frames*2, H, W]
        
        # 重塑为 [B, out_frames, H, W, 2]
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, 2, H, W).permute(0, 1, 3, 4, 2)
        
        return velocity_fields

class LightweightInceptionUNet(nn.Module):
    """
    轻量化版本的Inception-UNet
    减少了层数和通道数，适合计算资源有限的场景
    """
    def __init__(self, frame_channels, in_frames, out_frames, base_channels=16, incep_ker=[3,5,7], groups=4):
        super().__init__()
        
        in_channels = frame_channels * in_frames
        self.in_frames = in_frames
        self.out_frames = out_frames
        
        # 编码器 - 3层
        self.enc1 = Inception(in_channels, base_channels//2, base_channels, incep_ker, groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = Inception(base_channels, base_channels, base_channels * 2, incep_ker, groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = Inception(base_channels * 2, base_channels * 2, base_channels * 4, incep_ker, groups)
        self.pool3 = nn.MaxPool2d(2)
        
        # 底层
        self.bottleneck = Inception(base_channels * 4, base_channels * 4, base_channels * 8, incep_ker, groups)
        
        # 解码器 - 3层
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4)
        self.dec3 = Inception(base_channels * 8, base_channels * 4, base_channels * 4, incep_ker, groups)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2)
        self.dec2 = Inception(base_channels * 4, base_channels * 2, base_channels * 2, incep_ker, groups)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionBlock(base_channels, base_channels)
        self.dec1 = Inception(base_channels * 2, base_channels, base_channels, incep_ker, groups)
        
        # 输出头
        self.velocity_head = nn.Conv2d(base_channels, out_frames * 2, 1)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # 底层
        b = self.bottleneck(self.pool3(e3))
        
        # 解码器
        d3 = self.up3(b)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))
        
        # 输出
        velocity_fields = self.velocity_head(d1)
        B, _, H, W = velocity_fields.shape
        velocity_fields = velocity_fields.view(B, self.out_frames, 2, H, W).permute(0, 1, 3, 4, 2)
        
        return velocity_fields

