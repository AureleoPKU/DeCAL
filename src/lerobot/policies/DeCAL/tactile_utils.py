import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def make_2d_norm(num_channels: int, use_batchnorm: bool) -> nn.Module:
    if use_batchnorm:
        return nn.BatchNorm2d(num_channels)
    # Prefer 8 groups when possible; otherwise pick the largest divisor <= 8.
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class MultiFingerTactileEncoder(nn.Module):
    """
    Encode multi-finger tactile images into:
    - per-finger (local) tactile tokens
    - a global tactile summary token that can be used as a gate.

    Expected input:
        x: (batch, num_fingers, C, H, W), e.g. 10 fingertips, RGB 240x240.

    Outputs:
        local_tokens: (batch, num_fingers, hidden_dim)
        global_token: (batch, hidden_dim)
        attn_weights: (batch, num_fingers)  # attention used for global pooling
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_fingers: int = 10,
        hidden_dim: int = 256,
        cnn_channels: tuple[int, int, int] = (64, 128, 256),
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.num_fingers = num_fingers
        self.hidden_dim = hidden_dim

        c1, c2, c3 = cnn_channels

        # Shared CNN backbone over all fingertips (fine-grained local features)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c1, use_batchnorm),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c2, use_batchnorm),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c3, use_batchnorm),
            nn.SiLU(),
        )

        # Project CNN features to desired hidden_dim
        self.spatial_proj = nn.Conv2d(c3, hidden_dim, kernel_size=1)

        # Cross-finger fusion (coarser, finger-level features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.finger_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Attention pooling over fingers to get a global tactile summary
        # self.global_attn_weight = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, F, C, H, W) tactile images for F fingertips.

        Returns:
            A dict with:
                - "local_tokens": (B, F, D)
                - "global_token": (B, D)
                - "attn_weights": (B, F)
        """
        b, f, c, h, w = x.shape
        assert f == self.num_fingers, f"Expected num_fingers={self.num_fingers}, got {f}"

        # Merge batch and finger dims for CNN
        x = x.view(b * f, c, h, w)
        feat = self.backbone(x)                 # (B*F, C3, h', w')
        feat = self.spatial_proj(feat)          # (B*F, D, h', w')

        # Fine-grained to finger-level token via spatial pooling
        feat = feat.flatten(2)                  # (B*F, D, S)
        local_tokens = feat.mean(dim=-1)        # (B*F, D)
        local_tokens = local_tokens.view(b, f, self.hidden_dim)  # (B, F, D)

        # Cross-finger fusion with Transformer (sequence length = num_fingers)
        # Transformer expects (S, B, D)
        tokens_seq = local_tokens.permute(1, 0, 2)  # (F, B, D)
        encoded_seq = self.finger_encoder(tokens_seq)  # (F, B, D)
        encoded_local = encoded_seq.permute(1, 0, 2)  # (B, F, D)

        # Attention pooling to get global tactile summary
        # scores: (B, F)
        # scores = (encoded_local * self.global_attn_weight)  # (B, F, D)
        # scores = scores.sum(dim=-1)                        # (B, F)
        # attn_weights = F.softmax(scores, dim=-1)           # (B, F)

        # global_token = (attn_weights.unsqueeze(-1) * encoded_local).sum(dim=1)  # (B, D)
        global_token = encoded_local.mean(dim=1)  # (B, D)
        attn_weights = torch.full(
            (b, f), 1.0 / float(f), device=encoded_local.device, dtype=encoded_local.dtype
        )
        
        return {
            "local_tokens": encoded_local,
            "global_token": global_token,
            "attn_weights": attn_weights,
        }


class TactileFlowEncoder(nn.Module):
    """
    Encode tactile optical flow (computed from two consecutive deform frames) into per-finger tokens.
    Input: (B, F, 2, H, W) — flow_x and flow_y per pixel per finger (Farneback dense flow).
    Output: (B, F, hidden_dim) — one token per finger.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_fingers: int = 10,
        hidden_dim: int = 256,
        cnn_channels: tuple[int, int, int] = (64, 128, 256),
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_fingers = num_fingers
        self.hidden_dim = hidden_dim
        c1, c2, c3 = cnn_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c1, use_batchnorm),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c2, use_batchnorm),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            make_2d_norm(c3, use_batchnorm),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(c3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, 2, H, W) — tactile optical flow (flow_x, flow_y per pixel per finger).
        Returns:
            (B, F, hidden_dim) — per-finger flow token.
        """
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        feat = self.backbone(x)           # (B*F, c3, h', w')
        feat = self.pool(feat).flatten(1) # (B*F, c3)
        feat = self.proj(feat)            # (B*F, hidden_dim)
        return feat.view(b, f, self.hidden_dim)


class TactileImageDecoder(nn.Module):
    """
    Decode tactile latent to per-finger images. (B, D) -> (B, 10, C, 224, 224).
    14 -> 28 -> 56 -> 112 -> 224 with 4 upsampling stages.
    """

    def __init__(self, latent_dim: int, out_channels: int, num_fingers: int = 10):
        super().__init__()
        self.num_fingers = num_fingers
        self.out_channels = out_channels
        self.fc = nn.Linear(latent_dim, num_fingers * 64 * 14 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 14->28
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 28->56
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # 56->112
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 112->224
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.fc(x)
        h = h.view(b * self.num_fingers, 64, 14, 14)
        h = self.deconv(h)
        return h.view(b, self.num_fingers, self.out_channels, 224, 224)


class TactileLocalTokenImageDecoder(nn.Module):
    """
    Decode per-finger tokens from MultiFingerTactileEncoder (B, F, D) -> (B, F, C, 224, 224).
    Shared MLP + deconv per finger for auxiliary reconstruction loss on encoder local_tokens.
    """

    def __init__(self, token_dim: int, out_channels: int = 3, num_fingers: int = 10) -> None:
        super().__init__()
        self.num_fingers = num_fingers
        self.out_channels = out_channels
        self.fc = nn.Linear(token_dim, 64 * 14 * 14)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, local_tokens: torch.Tensor) -> torch.Tensor:
        b, f, d = local_tokens.shape
        x = local_tokens.reshape(b * f, d)
        h = self.fc(x)
        h = h.view(b * f, 64, 14, 14)
        h = self.deconv(h)
        return h.view(b, f, self.out_channels, 224, 224)


def _tensor_to_gray(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor in [0,1] or [-1,1] to grayscale uint8."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    if img_np.min() >= 0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    if img_np.shape[2] == 3:
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_np[:, :, 0]


def _compute_dense_flow(gray_prev: np.ndarray, gray_curr: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow. Returns (H, W, 2)."""
    return cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )


def compute_tactile_flow(tactile_images: torch.Tensor, t_prev: int = 0, t_curr: int = 1) -> torch.Tensor:
    """
    Compute optical flow between two deform frames for each finger.

    Args:
        tactile_images: (B, 10, T, C, H, W) — deform images, T temporal frames, C channels.
        t_prev: Index of previous frame (default 0 = current).
        t_curr: Index of current frame (default 1 = future).

    Returns:
        (B, 10, 2, H, W) — flow_x and flow_y per pixel per finger.
    """
    if not HAS_CV2:
        raise ImportError("cv2 is required for compute_tactile_flow. Install with: pip install opencv-python")

    b, f, t, c, h, w = tactile_images.shape
    assert t > max(t_prev, t_curr), f"Need at least {max(t_prev, t_curr) + 1} temporal frames"

    flows = []
    for bi in range(b):
        finger_flows = []
        for fi in range(f):
            img_prev = tactile_images[bi, fi, t_prev]  # (C, H, W)
            img_curr = tactile_images[bi, fi, t_curr]  # (C, H, W)
            gray_prev = _tensor_to_gray(img_prev)
            gray_curr = _tensor_to_gray(img_curr)
            flow = _compute_dense_flow(gray_prev, gray_curr)  # (H, W, 2)
            finger_flows.append(flow)
        flows.append(np.stack(finger_flows, axis=0))  # (10, H, W, 2)

    flow_np = np.stack(flows, axis=0)  # (B, 10, H, W, 2)
    flow_tensor = torch.from_numpy(flow_np).permute(0, 1, 4, 2, 3).float()  # (B, 10, 2, H, W)
    return flow_tensor.to(device=tactile_images.device)
