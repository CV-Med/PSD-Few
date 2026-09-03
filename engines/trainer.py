"""Meta-training engine for PSD-Few.

Trains the model with episode-level (N-way K-shot) sampling under the joint
loss of Eq.24, validates periodically, and saves the best checkpoint to
``weights/best_checkpoint.pth``.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import EpisodeSampler
from engines.evaluator import Evaluator
from losses.loss import PSDFewLoss
from utils.experiment_recorder import ExperimentRecorder
from utils.logger import Logger


class Trainer:
    """Runs the episode-level meta-training loop for a PSD-Few model."""

    def __init__(self, model: nn.Module, train_dataset, val_dataset, config: dict,
                 n_way: int = 5, k_shot: int = 5, n_query: int = 15,
                 device=None, recorder: Optional[ExperimentRecorder] = None,
                 logger: Optional[Logger] = None, checkpoint_dir: str = "weights"):
        """Set up the optimizer, scheduler, loss, and evaluator from the config."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.recorder = recorder
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device(config.get("device", "cuda:0"))
        self.logger = logger or Logger(log_dir="./logs", name="train")

        tcfg = config["training"]
        self.epochs = tcfg.get("epochs", 200)
        self.train_episodes = tcfg.get("train_episodes", 200)
        self.val_episodes = tcfg.get("val_episodes", 100)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.eval_every = tcfg.get("eval_every", 10)
        self.early_stop_patience = tcfg.get("early_stop_patience", 30)
        self.lr_min = float(tcfg.get("lr_min", 1e-7))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(tcfg.get("learning_rate", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr_min)

        loss_cfg = config["loss"]
        self.criterion = PSDFewLoss(
            lambda_d=loss_cfg.get("lambda_d", 0.8),
            lambda_f=loss_cfg.get("lambda_f", 0.15),
        )

        self.evaluator = Evaluator(model, config, n_way=n_way, k_shot=k_shot,
                                   n_query=n_query, device=self.device)

        self.best_metric = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one meta-training epoch over sampled episodes.

        Returns the mean loss and training accuracy of the epoch.
        """
        self.model.train()

        sampler = EpisodeSampler(self.train_dataset, self.n_way, self.k_shot,
                                 self.n_query, self.train_episodes)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(sampler, desc=f"Epoch {epoch}/{self.epochs}")
        for support_imgs, support_lbls, query_imgs, query_lbls in pbar:
            support_imgs = support_imgs.to(self.device)
            support_lbls = support_lbls.to(self.device)
            query_imgs = query_imgs.to(self.device)
            query_lbls = query_lbls.to(self.device)

            outputs = self.model(
                support_imgs, support_lbls, query_imgs, self.n_way,
                query_labels=query_lbls, use_rectify=True)

            loss, _ = self.criterion(outputs, query_lbls)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs["logits"], 1)
            total_correct += (preds == query_lbls).sum().item()
            total_count += query_lbls.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{total_correct / max(1, total_count):.3f}",
            })

        self.scheduler.step()
        return {"loss": total_loss / max(1, len(sampler)),
                "acc": total_correct / max(1, total_count)}

    def validate(self, dataset=None, k_shot=None, n_episodes=None, num_seeds=1):
        """Evaluate the model on the validation split and return the metrics."""
        return self.evaluator.evaluate(
            dataset or self.val_dataset, n_way=self.n_way,
            k_shot=k_shot or self.k_shot,
            n_episodes=n_episodes or self.val_episodes,
            num_seeds=num_seeds)

    def fit(self, epochs: Optional[int] = None, load_checkpoint: bool = True):
        """Train the model for ``epochs`` epochs with early stopping.

        Optionally resumes from ``weights/best_checkpoint.pth``, validates every
        ``eval_every`` epochs, saves the best state, and records the result.
        """
        if epochs is not None:
            self.epochs = epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.lr_min)

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            if load_checkpoint:
                ckpt = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
                if os.path.exists(ckpt):
                    try:
                        state = torch.load(ckpt, map_location=self.device)
                        self.model.load_state_dict(state)
                        self.logger.info(f"Auto-loaded checkpoint from {ckpt}")
                    except RuntimeError as e:
                        self.logger.warning(
                            f"Checkpoint {ckpt} incompatible with the current model "
                            f"({e}); training from scratch.")

        early_stop_counter = 0
        for epoch in range(1, self.epochs + 1):
            metrics = self.train_epoch(epoch)
            lr_current = self.scheduler.get_last_lr()[0]
            msg = (f"Epoch {epoch:3d}/{self.epochs} | "
                   f"Loss {metrics['loss']:.4f} | Train Acc {metrics['acc']:.4f} | "
                   f"LR {lr_current:.2e}")

            if epoch % self.eval_every == 0:
                val = self.validate()
                acc = val["accuracy"]
                msg += f" | Val {self.n_way}-way {self.k_shot}-shot: {acc:.4f} ± {val['ci95']:.4f}"
                if acc > self.best_metric:
                    self.best_metric = acc
                    self.best_epoch = epoch
                    early_stop_counter = 0
                    if self.checkpoint_dir:
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))
                else:
                    early_stop_counter += 1
            else:
                msg += " | Val: skipped"

            self.logger.info(msg)

            if early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        final_val = self.validate()
        final_acc = final_val["accuracy"]
        if final_acc > self.best_metric:
            self.best_metric = final_acc
            self.best_epoch = self.epochs
            if self.checkpoint_dir:
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))
        self.logger.info(f"Final val accuracy: {final_acc:.4f}")

        self.logger.info(f"Best val accuracy: {self.best_metric:.4f} @ epoch {self.best_epoch}")

        if self.recorder is not None:
            self.recorder.record_metric("best_accuracy", self.best_metric)
            self.recorder.save()
