"""
Model monitoring with Evidently.

Detects two failure modes that kill recommendation quality in production:

1. Data drift — the distribution of incoming user queries or podcast
   metadata shifts away from what the model was trained on. Example:
   a new podcast category explodes in popularity; the model has never
   seen it and will under-recommend it.

2. Model performance decay — NDCG / CTR drops over time as user tastes
   evolve or the podcast catalogue changes. Triggers a retraining alert
   when the drop exceeds a configurable threshold.

Reports are saved as HTML (human-readable) and JSON (for CI/CD gates).

Usage:
    monitor = RecommendationMonitor()
    monitor.run(reference_df, current_df, output_dir="reports/")
    if monitor.has_drift:
        trigger_retraining()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        ColumnDriftMetric,
        DatasetDriftMetric,
        DatasetMissingValuesSummaryMetric,
    )
    from evidently.report import Report
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger.warning("Evidently not installed. Monitoring will use simple statistical fallback.")
    EVIDENTLY_AVAILABLE = False


class RecommendationMonitor:
    """
    Monitors data drift and model performance for the recommendation pipeline.

    Two workflows:

    Offline (batch): Compare a reference window (e.g. last 30 days of training
    data) against a current window (e.g. last 7 days of production traffic).
    Run daily via GitHub Actions or a cron job.

    Online (real-time): Log each request's features to a buffer; flush and
    compare against the reference distribution every N requests.

    Example:
        monitor = RecommendationMonitor(drift_threshold=0.15)
        report = monitor.run_drift_report(reference_df, current_df)
        print(f"Drift detected: {monitor.has_drift}")
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        performance_threshold: float = 0.05,
        reference_window_days: int = 30,
        monitoring_window_days: int = 7,
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.reference_window_days = reference_window_days
        self.monitoring_window_days = monitoring_window_days

        self.has_drift: bool = False
        self.has_performance_decay: bool = False
        self._last_report: Optional[dict] = None

    # ── Data drift ────────────────────────────────────────────────────────────

    def run_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_dir: str = "reports/",
        numerical_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a full data drift report comparing reference vs current data.

        Args:
            reference_df:         Reference dataset (training distribution).
            current_df:           Current production data window.
            output_dir:           Where to save HTML + JSON reports.
            numerical_features:   Numeric columns to monitor (auto-detected if None).
            categorical_features: Categorical columns to monitor.

        Returns:
            Summary dict with drift flags and per-feature drift scores.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_features = numerical_features or self._detect_numerical(reference_df)
        cat_features = categorical_features or self._detect_categorical(reference_df)

        logger.info(
            f"Running drift report | ref={len(reference_df)} rows "
            f"cur={len(current_df)} rows | "
            f"num_features={num_features} cat_features={cat_features}"
        )

        if EVIDENTLY_AVAILABLE:
            return self._run_evidently_report(
                reference_df, current_df, output_path, num_features, cat_features
            )
        else:
            return self._run_statistical_fallback(
                reference_df, current_df, output_path, num_features
            )

    def _run_evidently_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_path: Path,
        num_features: list[str],
        cat_features: list[str],
    ) -> dict:
        """Full Evidently drift report."""
        column_mapping = ColumnMapping(
            numerical_features=num_features,
            categorical_features=cat_features,
        )

        report = Report(metrics=[
            DatasetDriftMetric(),
            DataQualityPreset(),
            DataDriftPreset(),
        ])
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save HTML report
        html_path = output_path / f"drift_report_{timestamp}.html"
        report.save_html(str(html_path))
        logger.info(f"Drift report saved: {html_path}")

        # Extract summary from JSON
        report_dict = report.as_dict()
        drift_metric = report_dict["metrics"][0]["result"]

        dataset_drift = drift_metric.get("dataset_drift", False)
        drift_share = drift_metric.get("share_of_drifted_columns", 0.0)

        self.has_drift = dataset_drift or (drift_share > self.drift_threshold)

        summary = {
            "timestamp": timestamp,
            "has_drift": self.has_drift,
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drift_threshold": self.drift_threshold,
            "n_reference": len(reference_df),
            "n_current": len(current_df),
            "html_report": str(html_path),
        }

        # Save JSON summary for CI/CD gates
        json_path = output_path / f"drift_summary_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._last_report = summary
        if self.has_drift:
            logger.warning(
                f"DRIFT DETECTED — drift_share={drift_share:.2%} "
                f"(threshold={self.drift_threshold:.2%}). Retraining recommended."
            )
        else:
            logger.info(f"No significant drift detected (drift_share={drift_share:.2%}).")

        return summary

    def _run_statistical_fallback(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_path: Path,
        num_features: list[str],
    ) -> dict:
        """
        Simple PSI (Population Stability Index) fallback when Evidently is not installed.

        PSI < 0.10: no drift
        PSI 0.10–0.25: moderate drift, investigate
        PSI > 0.25: significant drift, retrain
        """
        psi_scores = {}
        for col in num_features:
            if col in reference_df.columns and col in current_df.columns:
                psi_scores[col] = self._compute_psi(
                    reference_df[col].dropna(), current_df[col].dropna()
                )

        avg_psi = sum(psi_scores.values()) / max(len(psi_scores), 1)
        self.has_drift = avg_psi > self.drift_threshold

        summary = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "method": "psi_fallback",
            "has_drift": self.has_drift,
            "avg_psi": round(avg_psi, 4),
            "per_feature_psi": {k: round(v, 4) for k, v in psi_scores.items()},
            "drift_threshold": self.drift_threshold,
        }

        json_path = output_path / "drift_summary_fallback.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"PSI drift check | avg_psi={avg_psi:.4f} | drift={self.has_drift}")
        return summary

    # ── Performance monitoring ────────────────────────────────────────────────

    def check_performance_decay(
        self,
        baseline_ndcg: float,
        current_ndcg: float,
        output_dir: str = "reports/",
    ) -> dict:
        """
        Detect NDCG performance decay between the baseline and current model.

        Args:
            baseline_ndcg: NDCG@10 at training time (saved in MLflow).
            current_ndcg:  NDCG@10 computed on recent production data.
            output_dir:    Where to save the performance report JSON.

        Returns:
            Summary dict with decay flag and delta.
        """
        delta = baseline_ndcg - current_ndcg
        relative_drop = delta / max(baseline_ndcg, 1e-6)
        self.has_performance_decay = relative_drop > self.performance_threshold

        summary = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "baseline_ndcg": round(baseline_ndcg, 4),
            "current_ndcg": round(current_ndcg, 4),
            "delta": round(delta, 4),
            "relative_drop_pct": round(relative_drop * 100, 2),
            "performance_threshold_pct": round(self.performance_threshold * 100, 2),
            "has_performance_decay": self.has_performance_decay,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "performance_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.has_performance_decay:
            logger.warning(
                f"PERFORMANCE DECAY — NDCG dropped {relative_drop:.1%} "
                f"({baseline_ndcg:.4f} → {current_ndcg:.4f}). Retraining recommended."
            )
        else:
            logger.info(
                f"Performance stable | NDCG {baseline_ndcg:.4f} → {current_ndcg:.4f} "
                f"(drop={relative_drop:.1%})"
            )

        return summary

    @property
    def should_retrain(self) -> bool:
        """True if either drift or performance decay was detected."""
        return self.has_drift or self.has_performance_decay

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Population Stability Index between two continuous distributions."""
        breakpoints = pd.cut(reference, bins=bins, retbins=True)[1]
        breakpoints[0] = -float("inf")
        breakpoints[-1] = float("inf")

        ref_counts = pd.cut(reference, bins=breakpoints).value_counts(sort=False)
        cur_counts = pd.cut(current, bins=breakpoints).value_counts(sort=False)

        ref_pct = (ref_counts / len(reference)).clip(lower=1e-6)
        cur_pct = (cur_counts / len(current)).clip(lower=1e-6)

        psi = ((cur_pct - ref_pct) * (cur_pct / ref_pct).apply(lambda x: x if x > 0 else 1e-6).apply(
            lambda x: __import__("math").log(x)
        )).sum()
        return abs(float(psi))

    @staticmethod
    def _detect_numerical(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()

    @staticmethod
    def _detect_categorical(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)

    # Simulate reference vs current with slight drift
    reference = pd.DataFrame({
        "embedding_score": rng.normal(0.5, 0.15, 1000),
        "collab_score": rng.normal(0.6, 0.12, 1000),
        "play_count_norm": rng.beta(2, 5, 1000),
        "category": rng.choice(["Tech", "True Crime", "Comedy", "Science"], 1000),
    })
    current = pd.DataFrame({
        # Shift embedding_score distribution slightly
        "embedding_score": rng.normal(0.55, 0.18, 300),
        "collab_score": rng.normal(0.58, 0.14, 300),
        "play_count_norm": rng.beta(2, 4, 300),
        "category": rng.choice(["Tech", "True Crime", "Comedy", "Science", "Sports"], 300),
    })

    monitor = RecommendationMonitor(drift_threshold=0.15)
    drift_summary = monitor.run_drift_report(reference, current, output_dir="reports/test/")
    print("\nDrift summary:", json.dumps(drift_summary, indent=2))

    perf_summary = monitor.check_performance_decay(
        baseline_ndcg=0.72,
        current_ndcg=0.65,
        output_dir="reports/test/",
    )
    print("\nPerformance summary:", json.dumps(perf_summary, indent=2))
    print(f"\nShould retrain: {monitor.should_retrain}")
