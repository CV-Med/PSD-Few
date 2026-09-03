"""Attention Mechanism with Bidirectional Self-Distillation (BiAT, Eq.10-15).

The module decouples global and local features (Eq.10), fuses them through an
adaptive attention gate (Eq.11-12), and distills knowledge between the two
branches in both directions (Eq.13-15). The local branch extracts M = 4 patches
from a 2x2 grid, mapped to the global space by a weight-shared projector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalFeatureExtractor(nn.Module):
    """BiAT feature extractor: global-local decoupling and adaptive fusion (Eq.10-12)."""

    def __init__(self, in_channels=512, out_channels=512, num_patches=4):
        """Create the global branch, the 2x2 patch branch, and the fusion gate."""
        super().__init__()
        self.num_patches = num_patches

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        self.local_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(start_dim=2),
        )

        self.local_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        self.fusion_attention = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2),
            nn.Softmax(dim=1),
        )

    def forward_global(self, x):
        """Return the global embedding only, bypassing the local branch."""
        return self.global_branch(x)

    def forward(self, x, return_local=False):
        """Map a batch of images to the fused BiAT embedding.

        Takes [B, 3, H, W] features from the backbone and returns the fused
        [B, D] embedding, plus the [B, M, D] local patch embeddings when
        ``return_local`` is set.
        """
        global_feat = self.global_branch(x)

        local_patches = self.local_branch(x).permute(0, 2, 1)
        local_feats = []
        for i in range(self.num_patches):
            patch_feat = self.local_proj(local_patches[:, i, :])
            local_feats.append(patch_feat)
        local_feats = torch.stack(local_feats, dim=1)
        local_feat_agg = local_feats.mean(dim=1)

        concat_feat = torch.cat([global_feat, local_feat_agg], dim=1)
        fusion_weights = self.fusion_attention(concat_feat)
        fused_features = (fusion_weights[:, 0:1] * global_feat +
                          fusion_weights[:, 1:2] * local_feat_agg)

        if return_local:
            return fused_features, local_feats
        return fused_features


class BidirectionalSelfDistillation(nn.Module):
    """Parameter-free BiAT distillation objective (Eq.13-15)."""

    def __init__(self):
        super().__init__()

    def forward(self, local_features, global_feature):
        """Compute the bidirectional self-distillation loss (Eq.13-15).

        The local-to-global term is an L2 distance between the normalized local
        aggregate and the global embedding, and the global-to-local term is a KL
        divergence that treats the global embedding as the target distribution.
        Takes [B, M, D] local features and a [B, D] global embedding and returns
        the scalar Eq.15 loss.
        """
        local_agg = local_features.mean(dim=1)

        loss_lg = F.mse_loss(F.normalize(local_agg, dim=1),
                             F.normalize(global_feature, dim=1))

        p_local = F.log_softmax(local_agg, dim=1)
        q_global = F.softmax(global_feature, dim=1)
        loss_gl = F.kl_div(p_local, q_global, reduction='batchmean')

        return loss_lg + loss_gl
