import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
    SimCLR = encoder (CNN backbone) + projection head (MLP).
    Encoder learns features; projector helps contrastive training.
    """

    def __init__(self, backbone: str = "resnet18", proj_dim: int = 128):
        super().__init__()

        if backbone == "resnet18":
            base = models.resnet18(weights=None)

            # CIFAR-10 images are 32x32, so adjust stem to avoid over-downsampling
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.encoder = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection head (2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)            # (B, feat_dim)
        z = self.projector(h)          # (B, proj_dim)
        z = F.normalize(z, dim=1)      # normalize for cosine similarity
        return z
