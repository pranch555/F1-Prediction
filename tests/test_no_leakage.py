"""
Leakage-focused tests.

These tests make sure that our cross-validation never mixes rows
from the same race into both train and validation, and that the
target label (or obvious proxies) is not present among features.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure `src/` is importable when running `pytest` from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1pred.utils import load_config, make_group_kfold  # noqa: E402
from f1pred.data_ingest import load_tables  # noqa: E402
from f1pred.build_features import build_feature_matrix  # noqa: E402


def _get_cfg():
    """
    Load the base config and optional experiment overrides.
    You can set env vars:
      F1_CFG   -> path to base YAML (default: configs/base.yml)
      F1_EXPS  -> comma-separated list of experiment YAMLs
    """
    base = os.environ.get("F1_CFG", str(ROOT / "configs" / "base.yml"))
    exps_env = os.environ.get("F1_EXPS")
    exps = [p.strip() for p in exps_env.split(",")] if exps_env else None
    return load_config(base, exps)


@pytest.mark.leakage
def test_groupkfold_has_no_group_overlap():
    """
    For each fold, the set of group labels (i.e., races) in train and valid
    must be disjoint. This catches accidental leakage where rows from the
    same race land in both splits.
    """
    cfg = _get_cfg()
    tables = load_tables(cfg)
    X, y, groups, feat_names, state = build_feature_matrix(tables, cfg, fit=True)

    n_splits = int(cfg.get("split", {}).get("n_splits", 5))
    cv = make_group_kfold(n_splits=n_splits)

    for tr_idx, va_idx in cv.split(X, y, groups):
        tr_groups = set(groups.iloc[tr_idx])
        va_groups = set(groups.iloc[va_idx])
        overlap = tr_groups & va_groups
        assert not overlap, f"Group leakage: races present in both train and valid: {sorted(overlap)[:5]}"


@pytest.mark.leakage
def test_target_not_in_features():
    """
    Ensure the label or obvious proxies are NOT included as features.
    """
    cfg = _get_cfg()
    tables = load_tables(cfg)
    X, y, groups, feat_names, state = build_feature_matrix(tables, cfg, fit=True)

    # Target name comes from config; default to 'winner' if missing.
    target = cfg.get("target", {}).get("label", "winner")
    forbidden = {
        target,            # exact label
        "finish_pos",      # common naming
        "position",
        "podium",
        "points",          # season points can be a strong proxy if post-race
    }

    present = sorted(forbidden.intersection(set(feat_names)))
    assert not present, f"Feature leakage: target-like columns present in features: {present}"