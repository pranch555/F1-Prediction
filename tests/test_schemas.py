"""
Schema/contract tests for the feature pipeline.

We validate that the feature builder returns well-formed outputs:
- X columns align with feat_names
- X is numeric (or bool) and finite
- y/groups lengths match X
- groups has no missing values and is a string-like/identifier column
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

from f1pred.utils import load_config  # noqa: E402
from f1pred.data_ingest import load_tables  # noqa: E402
from f1pred.build_features import build_feature_matrix  # noqa: E402


def _get_cfg():
    base = os.environ.get("F1_CFG", str(ROOT / "configs" / "base.yml"))
    exps_env = os.environ.get("F1_EXPS")
    exps = [p.strip() for p in exps_env.split(",")] if exps_env else None
    return load_config(base, exps)


@pytest.mark.schema
def test_feature_matrix_contract():
    """
    Basic contract: shapes, names, and types.
    """
    cfg = _get_cfg()
    tables = load_tables(cfg)
    X, y, groups, feat_names, state = build_feature_matrix(tables, cfg, fit=True)

    # 1) Shapes align
    assert len(X) == len(y) == len(groups), "X, y, and groups must have equal length"

    # 2) Feature names exactly match columns
    assert list(X.columns) == list(feat_names), "X.columns must match feat_names in order"

    # 3) All features numeric or boolean
    bad = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    assert not bad, f"Non-numeric feature columns found: {bad}"

    # 4) No NaN/Inf in features
    x_values = X.to_numpy()
    assert np.isfinite(x_values).all(), "Features must be finite (no NaN/Inf)"

    # 5) Groups have no missing and look identifier-like (object/string)
    assert groups.isna().sum() == 0, "groups contains missing values"
    assert (groups.map(lambda v: isinstance(v, (str, int))).all()), "groups should be string/int identifiers"

    # 6) y should be numeric (ranking/regression) or boolean/int (classification)
    assert (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)), "y must be numeric or boolean"


@pytest.mark.schema
def test_minimum_join_keys_present():
    """
    Sanity check that core join keys exist in at least one table; this helps
    catch accidental column renames during ingestion.
    We don't enforce every table to have all keys, but at least one table
    should carry: year, Grand Prix, Code (driver code).
    """
    cfg = _get_cfg()
    tables = load_tables(cfg)

    keys = {"year", "Grand Prix", "Code"}
    has_keys = False
    for name, df in tables.items():
        if keys.issubset(set(df.columns)):
            has_keys = True
            break

    assert has_keys, "Expected at least one ingested table to include join keys {'year','Grand Prix','Code'}"