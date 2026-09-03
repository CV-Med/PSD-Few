"""Experiment tracking for the PSD-Few ablation study.

Records per-innovation ablation accuracy, effect size (Cohen's d), and the
comparison against SOTA baselines into ``experiment_summary.json``.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


class ExperimentRecorder:
    def __init__(self, innovation_names: List[str], sota_baselines: Dict[str, float],
                 baseline_metrics: Dict[str, float], iteration_round: int,
                 dataset_name: str = ""):
        """Store the experiment context and initialize the result containers."""
        self.innovation_names = innovation_names
        self.sota_baselines = sota_baselines
        self.baseline_metrics = baseline_metrics
        self.iteration_round = iteration_round
        self.dataset_name = dataset_name
        self.metrics: Dict[str, float] = {}
        self.ablation_results: List[Dict] = []
        self.failures: List[Dict] = []

    def record_metric(self, name: str, value: float):
        """Store a single metric value."""
        self.metrics[name] = value

    def record_ablation_batch(self, innovation_name: str,
                              with_scores: List[float], without_scores: List[float],
                              metric_name: str = "accuracy",
                              claimed_improvement: str = ""):
        """Record the ablation outcome for one module.

        Summarizes the with/without scores, the difference, and Cohen's d when
        at least three runs are available, and appends the entry.
        """
        n = len(with_scores)
        mean_with = float(np.mean(with_scores))
        mean_without = float(np.mean(without_scores))
        std_with = float(np.std(with_scores)) if n > 1 else 0.0
        std_without = float(np.std(without_scores)) if n > 1 else 0.0

        cohens_d = None
        interpretation = "insufficient_data"
        if n >= 3:
            pooled_std = np.sqrt((std_with ** 2 + std_without ** 2) / 2.0)
            if pooled_std > 1e-8:
                cohens_d = float((mean_with - mean_without) / pooled_std)
            else:
                cohens_d = 0.0
            interpretation = self._interpret_effect_size(cohens_d)

        self.ablation_results.append({
            "innovation": innovation_name,
            "metric_name": metric_name,
            "mean_with": mean_with,
            "mean_without": mean_without,
            "std_with": std_with,
            "std_without": std_without,
            "n": n,
            "delta": mean_with - mean_without,
            "effect_size": {"cohens_d": cohens_d, "interpretation": interpretation},
            "claimed_improvement": claimed_improvement,
        })

    @staticmethod
    def _interpret_effect_size(cohens_d: float) -> str:
        """Map a Cohen's d value to a qualitative label."""
        if cohens_d >= 0.8:
            return "large"
        elif cohens_d >= 0.5:
            return "medium"
        elif cohens_d >= 0.2:
            return "small"
        elif cohens_d >= -0.2:
            return "negligible"
        return "negative"

    def record_failure(self, scenario: str, effect: str, analysis: str = ""):
        """Record a reported training failure."""
        self.failures.append({"scenario": scenario, "effect": effect, "analysis": analysis})

    def save(self, output_path: str = "experiment_summary.json"):
        """Write the full experiment summary to a JSON file."""
        summary = {
            "iteration_round": self.iteration_round,
            "dataset_name": self.dataset_name,
            "overall_metrics": dict(self.baseline_metrics, **self.metrics),
            "sota_baselines": dict(self.sota_baselines),
            "innovation_results": self.ablation_results,
            "failure_cases": self.failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
