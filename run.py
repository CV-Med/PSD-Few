"""PSD-Few entry point: train / test / eval / ablation.

Usage:
    python run.py --mode train   --dataset miniimagenet
    python run.py --mode test    --dataset archive_44 --n_way 5
    python run.py --mode test    --dataset archive_15 --n_way 3
    python run.py --mode ablation --dataset archive_44

The script changes to the project root so all relative paths (config, data,
weights) resolve from the project directory.
"""

import argparse
import json
import os

import torch
import yaml

from data.dataset import FewShotDataset
from engines.evaluator import Evaluator
from engines.trainer import Trainer
from models import build_model
from utils.experiment_recorder import ExperimentRecorder
from utils.logger import Logger
from utils.reproducibility import set_seed


_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

BEST_CKPT = "weights/best_checkpoint.pth"


def build_datasets(config, dataset_cfg):
    """Build train/val/test datasets for the active dataset."""
    data_cfg = config["data"]
    split_ratio = tuple(data_cfg["split_ratio"])
    ds_kwargs = dict(
        image_size=dataset_cfg["image_size"],
        split_ratio=split_ratio,
        seed=config["seed"],
    )
    train_ds = FewShotDataset(dataset_cfg["data_root"], "train", **ds_kwargs)
    val_ds = FewShotDataset(dataset_cfg["data_root"], "val", **ds_kwargs)
    test_ds = FewShotDataset(dataset_cfg["data_root"], "test", **ds_kwargs)
    return train_ds, val_ds, test_ds


def run_test(model, test_ds, config, n_way, k_shot, n_query, num_seeds):
    """Evaluate the model on the test set for 1-shot and K-shot settings."""
    evaluator = Evaluator(model, config, n_way=n_way, k_shot=k_shot, n_query=n_query)
    results = {}
    for shot in (1, k_shot):
        r = evaluator.evaluate(test_ds, n_way=n_way, k_shot=shot,
                               num_seeds=num_seeds, verbose=True)
        results[f"{n_way}way_{shot}shot"] = r
    return results


def run_ablation(config, train_ds, val_ds, device, logger,
                 n_way, k_shot, n_query, ckpt, base_disabled):
    """Systematic module ablation (DEIP / BiAT / SFDP) via short fine-tuning."""
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train first.")
    full_state = torch.load(ckpt, map_location=device)
    finetune_epochs = config["ablation"]["finetune_epochs"]
    innovation_names = config["ablation"]["innovation_names"]

    model_full = build_model(config, disabled_modules=base_disabled).to(device)
    model_full.load_state_dict(full_state)
    trainer_full = Trainer(model_full, train_ds, val_ds, config, n_way=n_way, k_shot=k_shot,
                           n_query=n_query, device=device, checkpoint_dir=None, logger=logger)
    full_acc = trainer_full.validate()["accuracy"]
    logger.info(f"Full model | Val {n_way}-way {k_shot}-shot: {full_acc:.4f}")

    recorder = ExperimentRecorder(innovation_names,
                                  config["sota"].get(config["data"]["dataset_name"], {}),
                                  {"baseline_accuracy": full_acc}, 1,
                                  dataset_name=config["data"]["dataset_name"])
    recorder.record_metric("best_accuracy", full_acc)

    for name in innovation_names:
        logger.info(f"\n{'=' * 60}\nAblating: {name}\n{'=' * 60}")
        disabled = list(base_disabled or []) + [name]
        model_abl = build_model(config, disabled_modules=disabled).to(device)
        model_abl.load_state_dict(full_state, strict=False)

        trainer_abl = Trainer(model_abl, train_ds, val_ds, config, n_way=n_way,
                              k_shot=k_shot, n_query=n_query, device=device,
                              checkpoint_dir=None, logger=logger)
        trainer_abl.fit(epochs=finetune_epochs, load_checkpoint=False)
        acc = trainer_abl.best_metric
        logger.info(f"{name} ablated | Best val: {acc:.4f} (gap {full_acc - acc:+.4f})")
        recorder.record_ablation_batch(name, [full_acc], [acc], metric_name="accuracy")

    recorder.save()
    logger.info("Ablation results saved to experiment_summary.json")


def main():
    """Parse arguments and dispatch to the requested mode."""
    parser = argparse.ArgumentParser(
        description="PSD-Few: Physical-Constraints and Self-Distillation for "
                    "Few-Shot Brain MRI Diagnosis")
    parser.add_argument("--mode", type=str, choices=["train", "test", "eval", "ablation"],
                        default="train")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, choices=["miniimagenet", "archive_44", "archive_15"],
                        default="miniimagenet")
    parser.add_argument("--n_way", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=BEST_CKPT)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["datasets"][args.dataset]
    data_cfg = config["data"]
    data_cfg["dataset_name"] = args.dataset

    n_way = args.n_way if args.n_way is not None else data_cfg["n_way"]
    k_shot = data_cfg["k_shot"]
    n_query = data_cfg["n_query"]
    num_seeds = args.seeds if args.seeds is not None else config["training"]["num_seeds"]

    set_seed(config.get("seed", 42), config.get("deterministic", True))
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    logger = Logger(log_dir="./logs", name=f"{args.dataset}_{args.mode}")

    train_ds, val_ds, test_ds = build_datasets(config, dataset_cfg)
    logger.info(f"Dataset {args.dataset}: train {len(train_ds.classes)} classes, "
                f"val {len(val_ds.classes)}, test {len(test_ds.classes)}")

    base_disabled = ["DEIP"] if not dataset_cfg["use_deip"] else None

    if args.mode == "ablation":
        run_ablation(config, train_ds, val_ds, device, logger,
                     n_way, k_shot, n_query, args.ckpt, base_disabled)
        return

    model = build_model(config, disabled_modules=base_disabled).to(device)
    logger.info(f"PSD-Few model parameters: "
                f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if args.mode == "train":
        recorder = ExperimentRecorder(config["ablation"]["innovation_names"],
                                      config["sota"].get(args.dataset, {}), {},
                                      1, dataset_name=args.dataset)
        trainer = Trainer(model, train_ds, val_ds, config, n_way=n_way, k_shot=k_shot,
                          n_query=n_query, device=device, recorder=recorder, logger=logger)
        trainer.fit(epochs=args.epochs)

    elif args.mode in ("test", "eval"):
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}. Train first.")
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        logger.info(f"Loaded checkpoint from {args.ckpt}")

        results = run_test(model, test_ds, config, n_way, k_shot, n_query, num_seeds)

        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/test_results_{args.dataset}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to {out_path}")

        sota = config["sota"].get(args.dataset, {})
        for key, r in results.items():
            sota_val = sota.get(key)
            if sota_val is not None:
                logger.info(f"{key}: {r['accuracy']:.4f} vs manuscript {sota_val:.4f} "
                            f"(diff {100 * (r['accuracy'] - sota_val):+.2f}%)")


if __name__ == "__main__":
    main()
