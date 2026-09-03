"""Self-calibrating Few-shot Dual-layer Prototype Classification (SFDP, Eq.16-23).

The module builds class prototypes and M = 4 Farthest Point Sampling (FPS)
sub-prototypes per class (Eq.16-17), rectifies the class prototypes with a
differentiable two-layer MLP over T iterations (Eq.18-20), and fuses coarse and
fine cosine similarities for classification (Eq.21-23).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePrototypeSystem(nn.Module):
    """SFDP prototype builder: class prototypes (Eq.16) and FPS sub-prototypes (Eq.17)."""

    def __init__(self, feat_dim, num_sub_prototypes=4):
        """Create the sub-prototype attention network and store the count."""
        super().__init__()
        self.feat_dim = feat_dim
        self.num_sub_prototypes = num_sub_prototypes

        self.sub_proto_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 4, 1),
        )

    def compute_prototypes(self, support_features, support_labels, n_way):
        """Compute the class and sub-prototypes of each class.

        Takes normalized [n_support, D] support features with their labels and
        returns the [n_way, D] class prototypes and the
        [n_way, num_sub_prototypes, D] sub-prototypes.
        """
        support_features_norm = F.normalize(support_features, dim=1, eps=1e-8)

        class_prototypes = []
        sub_prototypes = []

        for c in range(n_way):
            mask = (support_labels == c)
            feats_c = support_features_norm[mask]

            if feats_c.size(0) == 0:
                class_prototypes.append(torch.zeros(self.feat_dim, device=support_features.device))
                sub_prototypes.append(torch.zeros(
                    self.num_sub_prototypes, self.feat_dim, device=support_features.device))
                continue

            class_prototypes.append(feats_c.mean(dim=0))

            if feats_c.size(0) >= self.num_sub_prototypes:
                indices = self._fps_sampling(feats_c, self.num_sub_prototypes)
                sub_proto = feats_c[indices]
            else:
                indices = torch.randint(0, feats_c.size(0), (self.num_sub_prototypes,))
                sub_proto = feats_c[indices]
            sub_prototypes.append(sub_proto)

        return torch.stack(class_prototypes), torch.stack(sub_prototypes)

    @staticmethod
    def _fps_sampling(features, k):
        """Select k maximally separated points via Farthest Point Sampling."""
        n = features.size(0)
        selected = []

        first_idx = torch.randint(0, n, (1,)).item()
        selected.append(first_idx)

        for _ in range(k - 1):
            max_min_dist = -1
            best_idx = 0
            for i in range(n):
                if i in selected:
                    continue
                min_dist = min(torch.norm(features[i] - features[j], p=2).item()
                               for j in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            selected.append(best_idx)

        return torch.tensor(selected, device=features.device)


class PrototypeRectificationNetwork(nn.Module):
    """SFDP prototype calibrator (Eq.18-20).

    A two-layer ReLU MLP (Phi_rect) refines the class prototypes over T
    iterations from the error vector e_c = [p_c, q_c] - p_c.
    """

    def __init__(self, feat_dim=512, num_iterations=3):
        """Create the two-layer ReLU rectification MLP."""
        super().__init__()
        self.feat_dim = feat_dim
        self.num_iterations = num_iterations

        self.rectify_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, prototypes, query_features, query_preds, query_labels=None):
        """Rectify the class prototypes iteratively (Eq.20).

        For each class, the query cluster center q_c (Eq.18) and the error vector
        (Eq.19) are computed, and the prototype is updated as
        p = norm(p + Phi_rect(e)). Takes [n_way, D] prototypes, [n_query, D]
        query features, and the initial [n_query, n_way] logits (or the
        ground-truth query labels during training), and returns the rectified
        [n_way, D] prototypes.
        """
        n_way = prototypes.size(0)
        current_protos = prototypes

        for _ in range(self.num_iterations):
            corrections = []

            for c in range(n_way):
                if query_labels is not None and self.training:
                    mask = (query_labels == c)
                else:
                    pred_class = query_preds.argmax(dim=1)
                    mask = (pred_class == c)

                if mask.sum() == 0:
                    corrections.append(torch.zeros_like(current_protos[c]))
                    continue

                avg_query = query_features[mask].mean(dim=0)

                proto_c = current_protos[c]
                error = torch.cat([proto_c, avg_query - proto_c], dim=0)
                correction = self.rectify_net(error)

                corrections.append(correction)

            current_protos = current_protos + torch.stack(corrections)
            current_protos = F.normalize(current_protos, dim=1, eps=1e-8)

        return current_protos
