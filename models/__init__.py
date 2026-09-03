"""PSD-Few few-shot diagnosis model.

The model assembles DEIP (physics enhancement), BiAT (bidirectional
self-distillation), and SFDP (dual-layer prototype classification) into a
single trainable few-shot framework. Individual modules can be disabled at
build time for the ablation study.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ResNet18Backbone
from models.biat import LocalGlobalFeatureExtractor, BidirectionalSelfDistillation
from models.deip import DEIPModule
from models.sfdp import BasePrototypeSystem, PrototypeRectificationNetwork


class PSDFewModel(nn.Module):
    """End-to-end PSD-Few few-shot classification model."""

    def __init__(self, feat_dim=512, n_way=5, num_patches=4, num_sub_prototypes=4,
                 num_iterations=3, fusion_alpha=0.65, pretrained=True,
                 deip_config: Optional[dict] = None,
                 disabled_modules: Optional[List[str]] = None):
        """Create the backbone and the DEIP, BiAT, and SFDP components."""
        super().__init__()
        if disabled_modules is None:
            disabled_modules = []
        self.disabled_modules = disabled_modules
        self.deip_enabled = "DEIP" not in disabled_modules
        self.biat_enabled = "BiAT" not in disabled_modules
        self.sfdp_enabled = "SFDP" not in disabled_modules

        self.feat_dim = feat_dim
        self.n_way = n_way
        self.fusion_alpha = fusion_alpha

        self.backbone = ResNet18Backbone(pretrained=pretrained)

        self.local_global_extractor = LocalGlobalFeatureExtractor(
            in_channels=feat_dim, out_channels=feat_dim, num_patches=num_patches)
        self.prototype_system = BasePrototypeSystem(
            feat_dim=feat_dim, num_sub_prototypes=num_sub_prototypes)
        self.bi_distill = BidirectionalSelfDistillation()
        self.proto_rectify = PrototypeRectificationNetwork(
            feat_dim=feat_dim, num_iterations=num_iterations)

        if self.deip_enabled and deip_config is not None:
            self.deip = DEIPModule(**deip_config)
        else:
            self.deip = None

    def extract_features(self, x, return_local=False):
        """Encode an image batch into BiAT (global-local) embeddings.

        Takes [B, 3, H, W] images and returns the fused [B, D] embedding, or the
        fused embedding together with the [B, M, D] local features when
        ``return_local`` is set. Falls back to the global branch when BiAT is
        disabled.
        """
        x = self.backbone(x)
        if not self.biat_enabled:
            features = self.local_global_extractor.forward_global(x)
            return (features, None) if return_local else features
        features, local_features = self.local_global_extractor(x, return_local=True)
        if return_local:
            return features, local_features
        return features

    def forward(self, support_images, support_labels, query_images, n_way,
                query_labels=None, use_rectify=True):
        """Classify a query set from an N-way support set.

        Augments the support images with DEIP during training, encodes the
        support and query sets with BiAT, builds the dual-layer prototypes
        (Eq.16-17), rectifies the class prototypes over T iterations (Eq.18-20),
        and fuses the coarse and fine cosine logits (Eq.21-23). Takes support
        and query image tensors with their labels and the way count, and returns
        a dict with the fused/coarse/fine logits, the query features, and the
        BiAT distillation loss (Eq.15).
        """
        if self.training and self.deip is not None:
            support_images = self.deip(support_images)

        if self.training and self.biat_enabled:
            support_features, _ = self.extract_features(support_images, return_local=True)
            query_features, query_local = self.extract_features(query_images, return_local=True)
        else:
            support_features = self.extract_features(support_images, return_local=False)
            query_features = self.extract_features(query_images, return_local=False)
            query_local = None

        use_rectify = use_rectify and self.sfdp_enabled

        class_prototypes, sub_prototypes = self.prototype_system.compute_prototypes(
            support_features, support_labels, n_way)

        query_norm = F.normalize(query_features, dim=1, eps=1e-8)
        proto_norm = F.normalize(class_prototypes, dim=1, eps=1e-8)
        init_logits = torch.mm(query_norm, proto_norm.t())

        if use_rectify:
            rectified_protos = self.proto_rectify(
                class_prototypes, query_features, init_logits, query_labels)
        else:
            rectified_protos = class_prototypes

        rectified_norm = F.normalize(rectified_protos, dim=1, eps=1e-8)
        coarse_logits = torch.mm(query_norm, rectified_norm.t())

        if self.sfdp_enabled:
            fine_sims = []
            for c in range(n_way):
                sub_proto_c = F.normalize(sub_prototypes[c], dim=1, eps=1e-8)
                sim_c = torch.mm(query_norm, sub_proto_c.t())
                attn_weights = self.prototype_system.sub_proto_attention(sub_proto_c)
                attn_weights = F.softmax(attn_weights, dim=0)
                weighted_sim = (sim_c * attn_weights.t()).sum(dim=1)
                fine_sims.append(weighted_sim)
            fine_logits = torch.stack(fine_sims, dim=1)
        else:
            fine_logits = coarse_logits

        fused_logits = self.fusion_alpha * coarse_logits + (1 - self.fusion_alpha) * fine_logits

        distill_loss = torch.tensor(0.0, device=support_images.device)
        if query_local is not None and self.training:
            distill_loss = self.bi_distill(query_local, query_features)

        return {
            "logits": fused_logits,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "query_features": query_features,
            "distill_loss": distill_loss,
        }


def build_model(config: dict, disabled_modules: Optional[List[str]] = None) -> PSDFewModel:
    """Build a PSD-Few model from a configuration dictionary.

    Reads the model and DEIP sections of ``config`` and returns a
    ``PSDFewModel`` with the requested modules disabled.
    """
    model_cfg = config["model"]
    deip_cfg = config.get("deip") or {}
    return PSDFewModel(
        feat_dim=model_cfg.get("feat_dim", 512),
        n_way=config["data"]["n_way"],
        num_patches=model_cfg.get("num_patches", 4),
        num_sub_prototypes=model_cfg.get("num_sub_prototypes", 4),
        num_iterations=model_cfg.get("num_iterations", 3),
        fusion_alpha=model_cfg.get("fusion_alpha", 0.65),
        pretrained=model_cfg.get("pretrained", True),
        deip_config=deip_cfg,
        disabled_modules=disabled_modules,
    )
