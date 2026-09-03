"""Episodic few-shot evaluation with global confusion-matrix visualization.

Corresponds to Figure 9 of the manuscript (confusion matrices of PSD-Few on
Brain44, Brain15, and miniImageNet). Builds a global (N_classes x N_classes)
confusion matrix by mapping the local episode labels back to the global class
indices.
"""

import argparse
import os
import random
import sys
import textwrap

import matplotlib
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _script_dir)

from data.dataset import FewShotDataset
from models import build_model


def _make_white_to_green_cmap():
    """Build the white-to-green colormap used for the heatmaps."""
    colors = [
        (1.0, 1.0, 1.0), (0.93, 0.98, 0.93), (0.78, 0.91, 0.78),
        (0.56, 0.79, 0.56), (0.24, 0.64, 0.24), (0.00, 0.39, 0.00),
    ]
    return LinearSegmentedColormap.from_list("white_to_green", colors, N=256)


def _wrap_labels(labels, width=14, max_lines=2):
    """Wrap class-name labels so they fit on the heatmap ticks."""
    wrapped = []
    for s in labels:
        s = str(s)
        lines = textwrap.wrap(s.replace("_", " "), width=width) or [s]
        lines = lines[:max_lines]
        wrapped.append("\n".join(lines))
    return wrapped


def _downsample_ticks(n_class, max_labels):
    """Return a subset of tick indices when there are many classes."""
    if n_class <= max_labels:
        return np.arange(n_class)
    step = int(np.ceil(n_class / max_labels))
    return np.arange(0, n_class, step)


def plot_confusion_matrices(cm_counts, class_names, save_prefix="outputs/confusion_matrix",
                            title_prefix="Confusion Matrix", annotate_threshold_pct=1.0,
                            max_xtick_labels=44):
    """Render the count and row-normalized confusion matrices as PNG and SVG.

    Writes ``{save_prefix}_counts.*`` and ``{save_prefix}_rownorm.*`` files and
    returns the count matrix and its row-normalized counterpart.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    cm_counts = np.asarray(cm_counts, dtype=np.int64)
    num_classes = cm_counts.shape[0]
    assert cm_counts.shape[0] == cm_counts.shape[1], "confusion matrix must be square"

    sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
    sums[sums == 0] = 1.0
    cm_norm = cm_counts / sums * 100.0
    cmap = _make_white_to_green_cmap()

    base = max(7.0, min(0.45 * num_classes, 16.0))
    figsize = (base, base * 0.9)
    wrapped_names = _wrap_labels(class_names)
    xticks_show = _downsample_ticks(num_classes, max_xtick_labels)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_counts, interpolation="nearest", cmap=cmap)
    ax.set_title(f"{title_prefix} (Counts)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(xticks_show)
    ax.set_xticklabels([wrapped_names[i] for i in xticks_show], rotation=45, ha="right")
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels(wrapped_names)
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect("equal")
    vmax = cm_counts.max() if cm_counts.size else 1
    thresh = vmax / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm_counts[i, j]
            pct = cm_norm[i, j]
            if i == j and v > 0:
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color=("white" if v > thresh else "black"),
                        fontsize=9, fontweight="bold")
            elif pct >= annotate_threshold_pct:
                ax.text(j, i, f"{pct:.1f}", ha="center", va="center",
                        color=("white" if v > thresh else "black"), fontsize=8)
    ax.plot(np.arange(num_classes), np.arange(num_classes), linewidth=1.2)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_counts.png", dpi=300)
    fig.savefig(f"{save_prefix}_counts.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=100.0)
    ax.set_title(f"{title_prefix} (Row-normalized %)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(xticks_show)
    ax.set_xticklabels([wrapped_names[i] for i in xticks_show], rotation=45, ha="right")
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels(wrapped_names)
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(num_classes):
        for j in range(num_classes):
            pct = cm_norm[i, j]
            if pct >= annotate_threshold_pct:
                ax.text(j, i, f"{pct:.1f}", ha="center", va="center",
                        color=("black" if pct < 50 else "white"), fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_rownorm.png", dpi=300)
    fig.savefig(f"{save_prefix}_rownorm.svg")
    plt.close(fig)

    return cm_counts, cm_norm


@torch.no_grad()
def episodic_eval_with_confusion(config, dataset_cfg, ckpt_path, split="test",
                                 n_way=5, k_shot=5, n_query=15, n_episodes=200,
                                 save_prefix="outputs/episodic_confusion",
                                 device=None, num_seeds=1):
    """Run episodic evaluation and build a global confusion matrix.

    Accumulates the episode predictions into a dataset-wide confusion matrix by
    mapping the local episode labels to the global class indices, and returns
    the overall query accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, disabled_modules=["DEIP"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = FewShotDataset(
        root_dir=dataset_cfg["data_root"], split=split,
        image_size=dataset_cfg["image_size"])
    class_names = dataset.classes
    num_classes = len(class_names)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_count = 0
    episode_accs = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pbar = tqdm(range(n_episodes), desc=f"{n_way}-way {k_shot}-shot episodic eval")
        for _ in pbar:
            episode_classes = random.sample(class_names, n_way)
            support_data, query_data = [], []
            support_labels_local, query_labels_local = [], []
            episode_class_indices = []

            for local_id, cls_name in enumerate(episode_classes):
                global_id = class_to_idx[cls_name]
                episode_class_indices.append(global_id)
                imgs = dataset.get_class_samples(cls_name, k_shot + n_query)
                support_data.append(imgs[:k_shot])
                query_data.append(imgs[k_shot:])
                support_labels_local.extend([local_id] * k_shot)
                query_labels_local.extend([local_id] * n_query)

            support_imgs = torch.cat(support_data, dim=0).to(device)
            query_imgs = torch.cat(query_data, dim=0).to(device)
            support_lbls = torch.tensor(support_labels_local, dtype=torch.long, device=device)
            query_lbls = torch.tensor(query_labels_local, dtype=torch.long, device=device)

            outputs = model(support_imgs, support_lbls, query_imgs, n_way,
                            use_rectify=True)
            logits = outputs["logits"]
            preds_local = torch.argmax(logits, dim=1)

            correct = (preds_local == query_lbls).sum().item()
            total_correct += correct
            total_count += query_lbls.size(0)
            episode_accs.append(correct / max(1, query_lbls.size(0)))
            pbar.set_postfix({"acc": f"{total_correct / max(1, total_count):.4f}"})

            for t_local, p_local in zip(query_lbls.cpu().numpy(), preds_local.cpu().numpy()):
                global_cm[episode_class_indices[t_local], episode_class_indices[p_local]] += 1

    overall_acc = total_correct / max(1, total_count)
    episode_accs = np.array(episode_accs)
    mean_acc = episode_accs.mean()
    ci95 = 1.96 * episode_accs.std() / np.sqrt(len(episode_accs))

    print("\n" + "=" * 60)
    print(f"{n_way}-way {k_shot}-shot episodic few-shot evaluation")
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"Per-episode mean: {mean_acc:.4f} ± {ci95:.4f} (95% CI)")
    print("=" * 60 + "\n")

    np.save(f"{save_prefix}_counts.npy", global_cm)
    np.savetxt(f"{save_prefix}_counts.csv", global_cm, fmt="%d", delimiter=",")
    plot_confusion_matrices(
        global_cm, class_names, save_prefix=save_prefix,
        title_prefix=f"{num_classes}-way Episodic Confusion Matrix (built from {n_way}-way episodes)",
        annotate_threshold_pct=1.0, max_xtick_labels=44)
    return overall_acc


def main():
    """Parse the command line and run the episodic confusion-matrix evaluation."""
    import yaml
    parser = argparse.ArgumentParser(description="Episodic eval + global confusion matrix")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="archive_44",
                        choices=["miniimagenet", "archive_44", "archive_15"])
    parser.add_argument("--ckpt", type=str, default="weights/best_checkpoint.pth")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--save_prefix", type=str, default="outputs/episodic_confusion")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset_cfg = config["datasets"][args.dataset]

    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    episodic_eval_with_confusion(
        config, dataset_cfg, ckpt_path=args.ckpt, split=args.split,
        n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
        n_episodes=args.n_episodes,
        save_prefix=args.save_prefix, device=device, num_seeds=args.seeds)


if __name__ == "__main__":
    main()
