import torch
import torch.nn as nn
from torchvision import models
from typing import cast


def _set_resnet_conv1_in_channels(resnet18: nn.Module, in_channels: int) -> None:
    if in_channels <= 0:
        raise ValueError(f"in_channels must be > 0, got {in_channels}")
    old_conv = cast(nn.Conv2d, resnet18.conv1)
    if int(old_conv.in_channels) == int(in_channels):
        return
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
    resnet18.conv1 = new_conv


class PDOChangeResNetClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 12, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        _set_resnet_conv1_in_channels(self.backbone, int(in_channels))
        in_features = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
