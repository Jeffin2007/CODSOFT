"""
encoder.py
CNN Image Encoder using pre-trained ResNet50.

Architecture decision:
- ResNet50 pre-trained on ImageNet provides powerful visual features
  without training a CNN from scratch (would require millions of images).
- We remove the final fully-connected classification layer and replace it
  with a linear projection that maps the 2048-dim ResNet features into
  the embed_dim used by the LSTM decoder.
- The CNN backbone is partially frozen: early layers capture universal
  low-level features (edges, textures) that don't need fine-tuning.
  Only the last ResNet layer block and the projection head are updated.
"""

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """ResNet50 encoder that extracts a fixed-size feature vector per image.

    Parameters
    ----------
    embed_dim : int
        Output feature dimension. Must match the decoder's embed_dim.
    fine_tune : bool
        If True, unfreeze the last ResNet block (layer4) for fine-tuning.
        Recommended to set False for the first few epochs, then True.
    """

    def __init__(self, embed_dim: int = 256, fine_tune: bool = False):
        super().__init__()

        # Load ResNet50 with ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final average-pool + fully-connected layers.
        # modules() after stripping: conv1 → bn1 → relu → maxpool →
        #   layer1 → layer2 → layer3 → layer4
        modules = list(resnet.children())[:-2]   # keeps spatial feature maps
        self.backbone = nn.Sequential(*modules)  # output: (B, 2048, 7, 7)

        # Global average pooling to collapse spatial dims → (B, 2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection: 2048 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self._set_fine_tune(fine_tune)

    # ------------------------------------------------------------------
    def _set_fine_tune(self, fine_tune: bool) -> None:
        """Freeze all backbone params, then optionally unfreeze layer4."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if fine_tune:
            # Unfreeze only the last residual block
            for param in self.backbone[-1].parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    def enable_fine_tuning(self) -> None:
        """Call this after a warm-up phase to unfreeze the last block."""
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (B, 3, 224, 224) — ImageNet-normalised

        Returns
        -------
        features : (B, embed_dim)
        """
        # Feature extraction through frozen backbone
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            feat = self.backbone(images)         # (B, 2048, 7, 7)

        feat = self.pool(feat)                   # (B, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)       # (B, 2048)
        feat = self.projection(feat)             # (B, embed_dim)
        return feat
