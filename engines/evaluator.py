"""Episodic evaluation engine.

Evaluates the model with N-way K-shot episodes. The manuscript protocol samples
E test episodes per fixed seed and reports the average Top-1 accuracy with a
95% confidence interval computed over the pooled episodes.
"""

import numpy as np
import torch

from data.dataset import EpisodeSampler
from utils.metrics import top1_accuracy, mean_confidence_interval
from utils.reproducibility import set_seed


class Evaluator:
    """N-way K-shot episodic evaluator with multi-seed pooling."""

    def __init__(self, model, config, n_way=5, k_shot=5, n_query=15, device=None):
        """Store the model, the episode settings, and the device."""
        self.model = model
        self.config = config
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.device = device or torch.device(config.get("device", "cuda:0"))
        self.test_episodes = config["training"].get("test_episodes", 100)

    @torch.no_grad()
    def evaluate(self, dataset, n_way=None, k_shot=None, n_episodes=None,
                 num_seeds=1, verbose=False):
        """Evaluate the model on a few-shot dataset.

        Runs ``n_episodes`` episodes per seed (100 by default), pools the query
        accuracy across all episodes, and returns the accuracy together with the
        per-episode mean, standard deviation, and 95% confidence interval.
        """
        n_way = n_way or self.n_way
        k_shot = k_shot or self.k_shot
        n_episodes = n_episodes or self.test_episodes

        self.model.eval()
        total_correct = 0
        total_count = 0
        episode_accs = []

        for seed in range(num_seeds):
            set_seed(seed)
            sampler = EpisodeSampler(dataset, n_way, k_shot, self.n_query, n_episodes)
            for support_imgs, support_lbls, query_imgs, query_lbls in sampler:
                support_imgs = support_imgs.to(self.device)
                support_lbls = support_lbls.to(self.device)
                query_imgs = query_imgs.to(self.device)
                query_lbls = query_lbls.to(self.device)

                outputs = self.model(support_imgs, support_lbls, query_imgs,
                                     n_way, use_rectify=True)
                logits = outputs["logits"]

                _, preds = torch.max(logits, 1)
                correct = (preds == query_lbls).sum().item()
                count = query_lbls.size(0)
                total_correct += correct
                total_count += count
                episode_accs.append(correct / count)

        accuracy = top1_accuracy(total_correct, total_count)
        mean, std, ci95 = mean_confidence_interval(episode_accs)

        if verbose:
            print("=" * 60)
            print(f"{n_way}-way {k_shot}-shot episodic evaluation ({dataset.split})")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  Mean: {mean:.4f} ± {ci95:.4f} (95% CI)")
            print(f"  Std: {std:.4f} | Episodes: {len(episode_accs)} | Seeds: {num_seeds}")
            print("=" * 60)

        return {"accuracy": accuracy, "mean": mean, "std": std,
                "ci95": ci95, "n": len(episode_accs), "n_seeds": num_seeds}
