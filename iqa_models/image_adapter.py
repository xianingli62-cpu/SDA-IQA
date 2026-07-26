# image_adapter.py
# MFEB (Multi-level Feature Extraction Branch) for SDA-IQA: extracts multi-scale
# residual features from the input image in the pixel space and injects them
# into the U-Net downsampling path as additional residuals.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ---------- Configuration ----------
class AdapterConfig:
    def __init__(self,
                 use_se: bool = True,
                 use_gate: bool = True,
                 use_scale: bool = True,
                 use_residual_scale: bool = True,
                 gate_slope: float = 0.5,
                 gate_init: str = 'smooth',
                 se_reduction: int = 16,
                 use_frequency: bool = True,
                 use_cross_scale: bool = True,
                 use_freq_gain: bool = True,
                 use_cross_gain: bool = True):
        self.use_se = use_se
        self.use_gate = use_gate
        self.use_scale = use_scale
        self.use_residual_scale = use_residual_scale
        self.gate_slope = gate_slope
        self.gate_init = gate_init
        self.se_reduction = se_reduction
        self.use_frequency = use_frequency
        self.use_cross_scale = use_cross_scale
        self.use_freq_gain = use_freq_gain
        self.use_cross_gain = use_cross_gain


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitate = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                      nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitate(y).view(b, c, 1, 1)
        return x * y


class SpatialGate(nn.Module):
    """Soft spatial gate with residual preservation."""
    def __init__(self, slope: float = 0.5, init_type: str = 'smooth'):
        super().__init__()
        self.slope = slope
        self.conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        if init_type == 'smooth':
            nn.init.constant_(self.conv.weight, 1.0 / 9.0)
        elif init_type == 'edge':
            weight = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32) / 8.0
            self.conv.weight.data.copy_(weight.view(1, 1, 3, 3))
        elif init_type == 'zero':
            nn.init.zeros_(self.conv.weight)
        else:
            raise ValueError(f"Unknown gate_init: {init_type}")

    def forward(self, x):
        g = x.mean(1, keepdim=True)
        mask = torch.sigmoid(self.conv(g) * self.slope)
        return x * mask + x * (1 - mask) * 0.1


class ResBlock(nn.Module):
    def __init__(self, channels: int, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.block1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.block2 = nn.Conv2d(channels, channels, 1)
        if config.use_se:
            self.se = SEBlock(channels, r=config.se_reduction)
        if config.use_gate:
            self.gate = SpatialGate(slope=config.gate_slope, init_type=config.gate_init)
        if config.use_scale:
            self.scale = nn.Parameter(torch.ones(1))
        if config.use_residual_scale:
            self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.act(self.block1(x))
        h = self.block2(h)
        if self.config.use_se:
            h = self.se(h)
        if self.config.use_gate:
            h = self.gate(h)
        if self.config.use_scale:
            h = h * self.scale
        res_scale = self.residual_scale if self.config.use_residual_scale else 1.0
        return x * res_scale + h


class FEM(nn.Module):
    """Frequency-aware Enhancement Module.

    Depthwise convolution with center-initialized kernels (identity + high-pass),
    isolating high-frequency residual components to enhance the perception of
    blur and compression artifacts."""
    def __init__(self, channels: int):
        super().__init__()
        self.freq_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        with torch.no_grad():
            nn.init.constant_(self.freq_conv.weight, 0)
            center = self.freq_conv.weight.size(2) // 2
            self.freq_conv.weight[:, :, center, center] = 1.0
        self.freq_conv.bias = None

    def forward(self, x):
        high = self.freq_conv(x) - x.mean(dim=[2, 3], keepdim=True)
        return x + high


class CAFM(nn.Module):
    """Cross-scale Attention Fusion Module.

    Promotes multi-scale information interaction between adjacent adapter
    blocks through spatial and channel attention."""
    def __init__(self, curr_c: int, prev_c: int, r: int = 16):
        super().__init__()
        self.prev_down = nn.Conv2d(prev_c, curr_c, 1) if prev_c != curr_c else nn.Identity()
        self.spatial = nn.Sequential(nn.Conv2d(curr_c, curr_c // r, 3, 1, 1), nn.ReLU(True),
                                     nn.Conv2d(curr_c // r, 1, 3, 1, 1), nn.Sigmoid())
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(curr_c, curr_c // r, 1), nn.ReLU(True),
                                     nn.Conv2d(curr_c // r, curr_c, 1), nn.Sigmoid())

    def forward(self, curr, prev):
        prev = self.prev_down(prev)
        if prev.size()[2:] != curr.size()[2:]:
            prev = F.interpolate(prev, size=curr.shape[2:], mode='bilinear', align_corners=False)
        curr = curr * self.channel(curr)
        prev = prev * self.spatial(curr)
        return curr + prev


# ---------- Adapter block ----------
class AdapterBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_res_blocks: int,
                 down: bool = False, config: Optional[AdapterConfig] = None):
        super().__init__()
        self.downsample = nn.AvgPool2d(2, 2) if down else None
        self.in_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else None
        self.resnets = nn.Sequential(*[ResBlock(out_c, config) for _ in range(num_res_blocks)])
        # Frequency-aware enhancement after the residual output
        self.freq_block = FEM(out_c) if config.use_frequency else nn.Identity()
        self.freq_gain = nn.Parameter(torch.tensor(0.1)) if config.use_freq_gain else None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        x = self.resnets(x)
        if self.freq_gain is not None:
            x = x + self.freq_gain * self.freq_block(x)
        else:
            x = self.freq_block(x)
        return x


# ---------- MFEB (full branch) ----------
class MFEB(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 channels=[320, 640, 1280, 1280],
                 num_res_blocks: int = 2,
                 downscale_factor: int = 8,
                 config: Optional[AdapterConfig] = None):
        super().__init__()
        self.cfg = config or AdapterConfig()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels * downscale_factor**2, channels[0], 3, 1, 1)

        self.body = nn.ModuleList([
            AdapterBlock(channels[0], channels[0], num_res_blocks, down=False, config=self.cfg),
            AdapterBlock(channels[0], channels[1], num_res_blocks, down=True, config=self.cfg),
            AdapterBlock(channels[1], channels[2], num_res_blocks, down=True, config=self.cfg),
            AdapterBlock(channels[2], channels[3], num_res_blocks, down=True, config=self.cfg),
        ])

        # Cross-scale attention fusion between adjacent adapter blocks
        if self.cfg.use_cross_scale:
            self.cross_attn = nn.ModuleList([
                CAFM(channels[i], channels[i - 1]) if i > 0 else None
                for i in range(len(channels))
            ])
        else:
            self.cross_attn = nn.ModuleList([None] * len(channels))
        # Cross-scale learnable gain
        self.cross_gain = nn.Parameter(torch.tensor(0.1)) if self.cfg.use_cross_gain else None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'scale' in name or 'residual_scale' in name or 'freq_gain' in name or 'cross_gain' in name:
                nn.init.constant_(param, 0.1)
            elif 'gate.conv.weight' in name:
                pass
            elif 'se' in name and 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'se' in name and 'bias' in name:
                nn.init.zeros_(param)

    def load_compatible_weights(self, state_dict: Dict[str, torch.Tensor], verbose: bool = True) -> bool:
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
                if verbose:
                    print(f"  Loaded: {name}")
            else:
                skipped += 1
                if verbose:
                    print(f"  Skip: {name} (shape or key mismatch)")
        if verbose:
            print(f"[Weight Loading] Loaded: {loaded}, Skipped: {skipped}")
        return loaded > 0

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)
        features = []
        for i, block in enumerate(self.body):
            x = block(x)
            if self.cross_attn[i] is not None:
                cross_out = self.cross_attn[i](x, features[-1])
                x = x + (self.cross_gain * cross_out if self.cross_gain is not None else cross_out)
            features.append(x)
        return features


# ---------- Factory ----------
def create_adapter_variant(variant_name: str = 'all',
                           in_channels: int = 3,
                           channels=[320, 640, 1280, 1280],
                           num_res_blocks: int = 2,
                           downscale_factor: int = 8,
                           **kwargs) -> MFEB:
    # Ablation settings corresponding to the paper: FA (frequency-aware
    # enhancement) and CSA (cross-scale attention) toggles. LAAF is part of
    # the decoder and is controlled by the variant in sda_iqa.py.
    config_map = {
        'baseline': AdapterConfig(use_frequency=False, use_cross_scale=False),
        'fa': AdapterConfig(use_cross_scale=False),
        'fa+csa': AdapterConfig(),
        'laaf': AdapterConfig(use_frequency=False, use_cross_scale=False),
        'all': AdapterConfig(),
    }
    if variant_name not in config_map:
        raise ValueError(f"Unknown variant: {variant_name}")

    # Base config of the requested variant; extra keyword arguments
    # (e.g. use_freq_gain / use_cross_gain) override the defaults.
    cfg = config_map[variant_name]
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return MFEB(in_channels=in_channels,
                channels=channels,
                num_res_blocks=num_res_blocks,
                downscale_factor=downscale_factor,
                config=cfg)


if __name__ == "__main__":
    net = create_adapter_variant('all', use_freq_gain=True, use_cross_gain=True)
    x = torch.randn(1, 3, 256, 256)
    feats = net(x)
    for i, f in enumerate(feats):
        print(f"stage{i}", f.shape)