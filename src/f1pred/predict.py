"""Inference script for the F1 prediction project.

Loads saved artifacts (model + preprocessing state), rebuilds features from
current CSVs according to the provided config, and writes per-race predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import add_common_args, get_logger, load_config, load_artifacts, output_dir_from_cfg
from .data_ingest import load_tables
from .build_features import build_feature_matrix


def _rank_within_groups(scores: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Return position ranks (1 best) within each group using stable sort."""
    ranks = np.empty_like(scores, dtype=int)
    df = pd.DataFrame({"score": scores, "group": groups})
    for g, idx in df.groupby("group").groups.items():
        i = np.fromiter(idx, dtype=int)
        order = i[np.argsort(-scores[i], kind="mergesort")]
        ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def predict_main(argv: List[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Predict with saved F1 model")
    add_common_args(parser)
    parser.add_argument("--artifacts", default=None, help="Directory containing model.joblib + preprocess_state.joblib")
    parser.add_argument("--out-csv", default=None, help="Where to save predictions CSV; defaults under output dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.exp)
    log = get_logger(level=cfg.get("logging", {}).get("level", "INFO"))

    out_dir = output_dir_from_cfg(cfg)

    if not args.artifacts:
        args.artifacts = str(out_dir)
    model, state = load_artifacts(args.artifacts)

    # Load current data and rebuild features using *fit=False*
    tables = load_tables(cfg)
    X, y, groups, feature_names, _ = build_feature_matrix(tables, cfg, fit=False, state=state)

    scores = model.predict(X)
    positions = _rank_within_groups(scores, groups)

    # Compose output frame with IDs if available
    cols = {}
    for c in ("race_id", "driver_id", "constructor_id"):
        if c in tables.get("results", pd.DataFrame()).columns:
            cols[c] = tables["results"][c].values
    pred_df = pd.DataFrame({**cols, "score": scores, "predicted_position": positions})

    out_csv = Path(args.out_csv) if args.out_csv else (Path(out_dir) / "predictions.csv")
    pred_df.to_csv(out_csv, index=False)
    log.info(f"Wrote predictions -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    predict_main()
