"""Grad-CAM attention visualization for PSD-Few (manuscript Fig.11).

Produces saliency heatmaps over the ResNet-18 backbone (targeting ``layer4``)
to highlight the regions the model focuses on for a given query image.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _script_dir)

from data.dataset import FewShotDataset
from data.transforms import build_val_transform
from models import build_model


def _episode_forward(model, query_imgs, support_imgs, support_lbls, n_way):
    """Run the PSD-Few forward with gradients only on the query branch.

    The support prototypes are computed under no_grad and the query image is
    passed through the model, returning the coarse logits and query features.
    """
    with torch.no_grad():
        support_features = model.extract_features(support_imgs)
        class_prototypes, sub_prototypes = model.prototype_system.compute_prototypes(
            support_features, support_lbls, n_way)

    query_features = model.extract_features(query_imgs)

    query_norm = F.normalize(query_features, dim=1, eps=1e-8)
    proto_norm = F.normalize(class_prototypes, dim=1, eps=1e-8)
    init_logits = torch.mm(query_norm, proto_norm.t())
    rectified_protos = model.proto_rectify(class_prototypes, query_features, init_logits, None)

    rectified_norm = F.normalize(rectified_protos, dim=1, eps=1e-8)
    coarse_logits = torch.mm(query_norm, rectified_norm.t())
    return coarse_logits, query_features


class GradCAM:
    """Gradient-weighted Class Activation Mapping on the backbone ``layer4``."""

    def __init__(self, model, target_layer=None):
        """Register forward and backward hooks on the target convolution."""
        self.model = model
        self.target_layer = target_layer or model.backbone.layer4[-1].conv2
        self.activations = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(self._save_activation)
        self._bh = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, query_imgs, support_imgs, support_lbls, n_way, class_idx=None):
        """Compute the Grad-CAM heatmap for the given query images."""
        self.model.zero_grad()
        self.model.eval()
        logits, _ = _episode_forward(self.model, query_imgs, support_imgs, support_lbls, n_way)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits[torch.arange(logits.size(0)), class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=query_imgs.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        """Remove the registered hooks."""
        self._fh.remove()
        self._bh.remove()


def visualize_gradcam(model, query_img, support_imgs, support_lbls, n_way,
                      save_path="outputs/gradcam.png", device=None):
    """Overlay the Grad-CAM heatmap on the query image and save it."""
    import matplotlib.pyplot as plt

    device = device or next(model.parameters()).device
    cam_tool = GradCAM(model)
    cam = cam_tool(query_img.unsqueeze(0).to(device), support_imgs.to(device),
                   support_lbls.to(device), n_way)
    cam_tool.close()

    img = query_img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(img * std + mean, 0, 1)

    heatmap = cam.cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(heatmap, alpha=0.5)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved to {save_path}")


def main():
    """Parse arguments and render the Grad-CAM figure for a query image."""
    import yaml
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="archive_44",
                        choices=["miniimagenet", "archive_44", "archive_15"])
    parser.add_argument("--ckpt", type=str, default="weights/best_checkpoint.pth")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Class to visualize (random image if not given)")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="outputs/gradcam.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset_cfg = config["datasets"][args.dataset]
    n_way = config["data"]["n_way"]
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    model = build_model(config, disabled_modules=["DEIP"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()

    if args.image_path is not None:
        from PIL import Image
        transform = build_val_transform(dataset_cfg["image_size"])
        query_img = transform(Image.open(args.image_path).convert("RGB"))
    else:
        dataset = FewShotDataset(dataset_cfg["data_root"], split=args.split,
                                 image_size=dataset_cfg["image_size"])
        cls_name = args.class_name or dataset.classes[0]
        samples = dataset.get_class_samples(cls_name, n_way + 15)
        support_imgs = samples[:n_way]
        query_img = samples[n_way]
        support_lbls = torch.arange(n_way, dtype=torch.long)

    visualize_gradcam(model, query_img, support_imgs, support_lbls, n_way,
                      save_path=args.save_path, device=device)


if __name__ == "__main__":
    main()
