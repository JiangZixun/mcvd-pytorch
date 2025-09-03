import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .DPIC_models import LightweightUNet, EarthformerUNet, InceptionUNet, LightweightInceptionUNet
from .RF_models import LightweightRefinementNet

class RFDPIC(nn.Module):
    """
    并行速度场预测器
    输入过去6帧 -> 一次性预测6个速度场 -> 并行生成6个未来帧
    """
    def __init__(self, 
                 dp_name: str=None,
                 rf_name: str=None,
                 rf_dp_config_file: str=None,
                 dp_mode='autoregrad',
                 rf_mode='autoregrad',
                 alpha=0.,
                 interpolation_mode="bilinear",
                 padding_mode="border"):
        super().__init__()
        cfg = self.get_base_config(rf_dp_config_file)
        dp_cfg = cfg['dp_model']
        rf_cfg = cfg['rf_model']

        # 轻量化Attention-UNet
        if dp_name == "LightweightUNet":
            model_cfg = dp_cfg[dp_name]
            self.displacement_net = LightweightUNet(
                frame_channels=model_cfg['frame_channels'],
                in_frames=model_cfg['in_frames'],
                out_frames=model_cfg['out_frames'],
                base_channels=model_cfg['base_channels']
            )
        # EarthformerUNet
        elif dp_name == "EarthformerUNet":
            # model_cfg = dp_cfg[dp_name]
            # self.displacement_net = EarthformerUNet(model_cfg)
            raise NotImplementedError
        # InceptionUNet
        elif dp_name == "InceptionUNet":
            # model_cfg = dp_cfg[dp_name]
            # self.displacement_net = InceptionUNet(
            #     frame_channels=model_cfg['frame_channels'],
            #     in_frames=model_cfg['in_frames'],
            #     out_frames=model_cfg['out_frames'],
            #     base_channels=model_cfg['base_channels'],
            #     incep_ker=model_cfg['incep_ker'],
            #     groups=model_cfg['groups']
            # )
            raise NotImplementedError
        # LightweightInceptionUNet
        elif dp_name == "LightweightInceptionUNet":
            # model_cfg = dp_cfg[dp_name]
            # self.displacement_net = LightweightInceptionUNet(
            #     frame_channels=model_cfg['frame_channels'],
            #     in_frames=model_cfg['in_frames'],
            #     out_frames=model_cfg['out_frames'],
            #     base_channels=model_cfg['base_channels'],
            #     incep_ker=model_cfg['incep_ker'],
            #     groups=model_cfg['groups']
            # )
            raise NotImplementedError
        # SimVP
        elif dp_name == "SimVP":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        # LightweightRefinementNet
        if rf_name == "LightweightRefinementNet":
            model_cfg = rf_cfg[rf_name]
            self.refine_net = LightweightRefinementNet(channels=model_cfg['channels'])
        # No Refinement Network
        else:
            self.refine_net = nn.Identity()
            # raise NotImplementedError
        
        assert dp_mode == 'parallel' or dp_mode == 'autoregrad', f"displacementPredictor's mode must be 'parallel' or 'autoregrad', but got({dp_mode})."
        assert rf_mode == 'parallel' or rf_mode == 'autoregrad', f"RefinementNet's mode must be 'parallel' or 'autoregrad', but got({rf_mode})."
        self.dp_mode = dp_mode
        self.rf_mode = rf_mode
        self.alpha = alpha
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode

        print(f"======================= RFDPIC Working Mode =======================")
        print(f"motion predictor: {dp_name}" )
        print(f"refinement net: {rf_name}")
        print(f"displacement predictor: {dp_mode}")
        print(f"refinement network: {rf_mode}")
        print(f"interpolation mode: {interpolation_mode}")
        print(f"padding mode: {padding_mode}")
        print(f"===================================================================")
    
    def get_base_config(self, config_file: str):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def forward(self, past_frames):
        """
        并行预测未来 6 帧
        Args:
            past_frames: [B, 6, C, H, W]
        Returns:
            future_frames:      [B, 6, C, H, W]
            displacement_fields:[B, 6, C, H, W, 2]
        """
        B, T, C, H, W = past_frames.shape
        last_frame = past_frames[:, -1] # [B, C, H, W]

        # <-- 位移场维度改为 BTCHW2 ----------------------------------------->
        displacement_fields = self.displacement_net(past_frames) # [B, 6, C, H, W, 2]

        future_frames = self._warp(last_frame, displacement_fields)

        if self.rf_mode == 'parallel':
            future_frames = self.refine_net(future_frames)

        return future_frames, displacement_fields


    def _warp(self, last_frame, displacement_fields):
        """
        将 6*C 个位移场并行作用于当前last_frame
        Args:
            last_frame:         [B, C, H, W]
            displacement_fields:[B, 6, C, H, W, 2]
        Returns:
            future_frames:      [B, 6, C, H, W]
        """
        B, C, H, W = last_frame.shape
        T = displacement_fields.size(1)        # =6
        device = last_frame.device

        # ----- 构造基础坐标网格并扩展到 [B, T, C, H, W, 2] -----------------
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)                  # [H, W, 2]
        base_grid = base_grid.view(1, 1, 1, H, W, 2).expand(B, T, C, -1, -1, -1)

        # ----- 新位置 = 基础坐标 + 位移 -------------------------------------
        new_pos = base_grid + displacement_fields # [B, T, C, H, W, 2]
        new_pos[..., 0] = 2.0 * new_pos[..., 0] / (W - 1) - 1.0 # 归一化到 [-1,1]
        new_pos[..., 1] = 2.0 * new_pos[..., 1] / (H - 1) - 1.0

        future_frames, cur_frame = [], last_frame # cur_frame 可能随 autoregrad 更新
        for t in range(T):
            # --- 把 B 和 C 维折叠成 batch，方便一次性 grid_sample ----------
            grid_t   = new_pos[:, t].reshape(B * C, H, W, 2)# [B*C, H, W, 2]
            input_t  = cur_frame.reshape(B * C, 1, H, W)# [B*C, 1, H, W]

            warped   = F.grid_sample(
                input_t, grid_t,
                mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=True
            ).view(B, C, H, W)

            future_frame = self.alpha * cur_frame + (1.0 - self.alpha) * warped

            # --- 自回归 & 细化（按原逻辑保持不变） -------------------------
            if self.dp_mode == 'autoregrad':
                if self.rf_mode == 'autoregrad':
                    future_frame = self.refine_net(future_frame)
                cur_frame = future_frame

            future_frames.append(future_frame)

        return torch.stack(future_frames, dim=1) # [B, 6, C, H, W]
