from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """1D Convolution followed by BatchNorm and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: str = "swish",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)  # swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for 1D signals."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        # Depthwise
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm1d(in_channels)
        # Pointwise
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm1d(out_channels)

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.act(x)
        return x


class SqueezeExcitation(nn.Module):
    """Channel-wise attention (SE block)."""
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool1d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualDSBlock(nn.Module):
    """Residual block with depthwise separable convs and SE."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.se = SqueezeExcitation(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.use_projection = stride != 1 or in_channels != out_channels
        if self.use_projection:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out = self.dropout(out)
        out = out + identity
        out = F.silu(out)
        return out


class HornQualityNet(nn.Module):
    """1D CNN for horn pass/fail classification from raw waveform."""
    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        dropout: float = 0.1,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.stem = ConvBNAct(num_input_channels, base_channels, kernel_size=11, stride=2)

        channels = [base_channels, base_channels * 2, base_channels * 3, base_channels * 4]
        strides = [1, 2, 2, 2]
        blocks = []
        in_ch = base_channels
        for out_ch, s in zip(channels, strides):
            blocks.append(
                ResidualDSBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=7,
                    stride=s,
                    dropout=dropout,
                    se_reduction=se_reduction,
                )
            )
            # Add a second block without downsampling to deepen
            blocks.append(
                ResidualDSBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=7,
                    stride=1,
                    dropout=dropout,
                    se_reduction=se_reduction,
                )
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.head_norm = nn.BatchNorm1d(in_ch)
        self.head_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.head_dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # [B]


def build_horn_quality_model(
    num_input_channels: int = 1,
    base_channels: int = 64,
    dropout: float = 0.1,
    se_reduction: int = 8,
) -> HornQualityNet:
    return HornQualityNet(
        num_input_channels=num_input_channels,
        base_channels=base_channels,
        dropout=dropout,
        se_reduction=se_reduction,
    )


class ConvNormActLite(nn.Module):
    """Lightweight Conv1d + Norm + Activation.

    Uses BatchNorm1d by default to match the existing style but with option to switch to InstanceNorm1d
    if batch sizes are very small.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | None = None,
        use_instance_norm: bool = False,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if use_instance_norm:
            self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        else:
            self.norm = nn.BatchNorm1d(out_channels)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlockLite(nn.Module):
    """Slightly advanced residual block for 1D audio.

    Two Conv-Norm-Act layers with residual connection and optional downsampling via stride on the first conv.
    Keeps parameter count modest compared to the advanced model.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
        use_instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.block1 = ConvNormActLite(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            use_instance_norm=use_instance_norm,
            dropout=dropout,
        )
        self.block2 = ConvNormActLite(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_instance_norm=use_instance_norm,
            dropout=dropout,
        )
        self.use_proj = stride != 1 or in_channels != out_channels
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels) if not use_instance_norm else nn.InstanceNorm1d(out_channels, affine=True),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = F.silu(out)
        return out


class HornQualityNetLite(nn.Module):
    """Lightweight residual CNN for pass/fail from raw 1D waveform.

    Design goals:
    - Slightly more advanced than a plain stack of Conv -> BN -> ReLU blocks
    - Residual connections, modest depth, adaptive pooling
    - Fewer parameters than the advanced architecture to reduce overfitting risk on small datasets
    """
    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        dropout: float = 0.15,
        use_instance_norm: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.stem = ConvNormActLite(num_input_channels, c, kernel_size=7, stride=2, use_instance_norm=use_instance_norm, dropout=dropout)

        stages: list[nn.Module] = []
        # Stage 1 (no downsample first, then downsample)
        stages.append(ResidualBlockLite(c, c, kernel_size=5, stride=1, dropout=dropout, use_instance_norm=use_instance_norm))
        stages.append(ResidualBlockLite(c, c, kernel_size=5, stride=2, dropout=dropout, use_instance_norm=use_instance_norm))
        # Stage 2
        stages.append(ResidualBlockLite(c, 2 * c, kernel_size=5, stride=2, dropout=dropout, use_instance_norm=use_instance_norm))
        stages.append(ResidualBlockLite(2 * c, 2 * c, kernel_size=5, stride=1, dropout=dropout, use_instance_norm=use_instance_norm))
        # Stage 3
        stages.append(ResidualBlockLite(2 * c, 3 * c, kernel_size=3, stride=2, dropout=dropout, use_instance_norm=use_instance_norm))
        stages.append(ResidualBlockLite(3 * c, 3 * c, kernel_size=3, stride=1, dropout=dropout, use_instance_norm=use_instance_norm))

        self.backbone = nn.Sequential(*stages)
        self.head_norm = nn.BatchNorm1d(3 * c) if not use_instance_norm else nn.InstanceNorm1d(3 * c, affine=True)
        self.head_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(3 * c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.head_dropout(x)
        x = self.classifier(x)
        return x.squeeze(-1)


def build_horn_quality_model_lite(
    num_input_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.15,
    use_instance_norm: bool = False,
) -> HornQualityNetLite:
    return HornQualityNetLite(
        num_input_channels=num_input_channels,
        base_channels=base_channels,
        dropout=dropout,
        use_instance_norm=use_instance_norm,
    )