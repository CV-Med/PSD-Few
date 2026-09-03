"""Joint training objective of PSD-Few (Eq.24).

L_total = L_cls + lambda_d * L_distill + lambda_f * L_fine

The classification term is cross-entropy on the fused logits, the distillation
term is the BiAT bidirectional self-distillation (Eq.13-15), and the fine term
is cross-entropy on the SFDP fine-grained logits (Eq.25). The optimal weights
are lambda_d = 0.8 and lambda_f = 0.15.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSDFewLoss(nn.Module):
    """PSD-Few joint loss with classification, distillation, and fine terms."""

    def __init__(self, lambda_d=0.8, lambda_f=0.15):
        """Store the distillation and fine-grained loss weights."""
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_f = lambda_f

    def forward(self, outputs, query_labels):
        """Compute the joint loss (Eq.24).

        Takes the dict returned by ``PSDFewModel.forward`` and the [n_query]
        ground-truth query labels, and returns the scalar loss together with a
        dict of its components.
        """
        logits = outputs["logits"]
        fine_logits = outputs["fine_logits"]
        distill_loss = outputs["distill_loss"]

        cls_loss = F.cross_entropy(logits, query_labels)
        fine_loss = F.cross_entropy(fine_logits, query_labels)

        total = cls_loss + self.lambda_d * distill_loss + self.lambda_f * fine_loss

        loss_dict = {
            "cls": cls_loss.item(),
            "distill": (self.lambda_d * distill_loss).item(),
            "fine": (self.lambda_f * fine_loss).item(),
            "total": total.item(),
        }
        return total, loss_dict
