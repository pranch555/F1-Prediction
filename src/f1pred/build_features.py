# src/f1pred/build_features.py
from __future__ import annotations

import re
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd


# -----------------------------
# Column resolution + targets
# -----------------------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """
    Return the actual column name in df that matches any name in candidates,
    ignoring case and non-alphanumerics (so raceId == race_id, positionOrder == position_order).
    """
    if df is None or df.empty:
        return None
    norm_map = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in norm_map:
            return norm_map[key]
    return None


def _normalize_finish_positions(results: pd.DataFrame) -> pd.Series:
    """
    Return integer finishing positions per row, tolerant to varied column names.

    Accepts numeric/text columns such as:
      - positionOrder / position / pos
      - finish_position / finishPosition / finishing_position
      - place / rank
      - positionText (e.g., "R", "DNF", "DSQ") -> coerced to NaN

    Any non-numeric (DNF/DSQ/etc.) gets pushed to the back of its race group (max+1).
    """
    if results is None or results.empty:
        raise ValueError("results table is empty; cannot compute finish positions.")

    candidates = [
        "finish_position", "finishPosition",
        "positionOrder", "position", "pos",
        "final_position", "finish_pos", "finishing_position",
        "place", "rank", "positionText",
    ]
    col = _find_col(results, candidates)
    if col is None:
        raise ValueError(
            "Could not locate finish position column in results. "
            f"Looked for {candidates}"
        )

    # Coerce to numeric. Text like "R", "DNF" -> NaN
    s = pd.to_numeric(results[col], errors="coerce")

    if s.isna().any():
        race_key = _find_col(results, ["race_id", "raceId"])
        if race_key is not None:
            def _fill_group(x: pd.Series) -> pd.Series:
                m = np.nanmax(x.values)
                if not np.isfinite(m):  # whole group NaN: shove far back
                    return x.fillna(99999)
                return x.fillna(int(m) + 1)

            s = results.assign(_s=s).groupby(race_key, dropna=False)["_s"].transform(_fill_group)
        else:
            m = np.nanmax(s.values)
            fillv = int(m) + 1 if np.isfinite(m) else 99999
            s = s.fillna(fillv)

    return s.astype(int)


# -----------------------------
# Feature engineering helpers
# -----------------------------

def _safe_num(series: pd.Series, default: float = np.nan, dtype: str | None = None) -> pd.Series:
    """Convert to numeric with safe fill."""
    s = pd.to_numeric(series, errors="coerce").fillna(default)
    if dtype == "int":
        s = s.astype(int)
    return s


def _join_race_meta(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Join year/round onto results if available."""
    if results is None or races is None or results.empty or races.empty:
        return results.assign(year=np.nan, round=np.nan)

    race_key_res = _find_col(results, ["race_id", "raceId"])
    race_key_rac = _find_col(races, ["race_id", "raceId"])
    if race_key_res is None or race_key_rac is None:
        return results.assign(year=np.nan, round=np.nan)

    year_col = _find_col(races, ["year"])
    round_col = _find_col(races, ["round"])

    keep_cols = [c for c in [race_key_rac, year_col, round_col] if c is not None]
    meta = races[keep_cols].copy()
    rename_map = {}
    if race_key_rac is not None:
        rename_map[race_key_rac] = "_race_key"
    if year_col:
        rename_map[year_col] = "year"
    if round_col:
        rename_map[round_col] = "round"
    meta = meta.rename(columns=rename_map)

    res = results.copy()
    res["_race_key"] = results[race_key_res]
    res = res.merge(meta, on="_race_key", how="left")
    return res


def _rolling_form(values: pd.Series, window: int = 3) -> pd.Series:
    """Previous-N rolling mean (shifted by 1 so current race is not included)."""
    return values.shift().rolling(window, min_periods=1).mean()


# -----------------------------
# Main entry point
# -----------------------------

def build_feature_matrix(
    tables: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Build X, y, groups for training/inference.

    Returns:
        X:           np.ndarray [n_samples, n_features]
        y:           np.ndarray [n_samples] (1 = winner, 2 = P2, ...)
        groups:      np.ndarray [n_samples] (race grouping for CV)
        feat_names:  List[str] feature names in column order
        state:       Dict[str, Any] (fitted state if needed)
    """
    # Required tables
    results = tables.get("results")
    if results is None or results.empty:
        raise ValueError("Missing or empty 'results' table.")

    races = tables.get("races")
    qualifying = tables.get("qualifying")

    # Resolve common keys
    race_key = _find_col(results, ["race_id", "raceId"])
    driver_key = _find_col(results, ["driver_id", "driverId"])
    team_key = _find_col(results, ["constructor_id", "constructorId"])
    grid_key = _find_col(results, ["grid"])

    # Targets & groups
    y = _normalize_finish_positions(results).clip(lower=1)  # ensure 1 = winner
    groups = (
        results[race_key].to_numpy()
        if race_key is not None else
        np.full(len(results), -1, dtype=int)
    )

    # --------------------
    # Base numeric features
    # --------------------
    feats: Dict[str, pd.Series] = {}

    # Grid position (0 means pit lane start in some datasets)
    if grid_key:
        feats["grid"] = _safe_num(results[grid_key], default=-1, dtype="int")
        feats["grid_is_pit"] = (feats["grid"] == 0).astype(int)
    else:
        feats["grid"] = pd.Series(np.full(len(results), -1), index=results.index, dtype=int)
        feats["grid_is_pit"] = (feats["grid"] == 0).astype(int)

    # IDs as numeric (useful as weak signals; many models will handle them)
    if driver_key:
        feats["driverId_num"] = _safe_num(results[driver_key], default=-1, dtype="int")
    if team_key:
        feats["constructorId_num"] = _safe_num(results[team_key], default=-1, dtype="int")

    # --------------------
    # Race meta & simple rolling form
    # --------------------
    res_meta = _join_race_meta(results, races)
    finish_numeric = _normalize_finish_positions(results)

    # Driver recent form (prev 3 results within season if year available)
    if driver_key:
        if "year" in res_meta.columns:
            feats["driver_form3"] = (
                res_meta
                .assign(_drv=results[driver_key].values, _fin=finish_numeric.values)
                .groupby(["_drv", "year"], dropna=False)["_fin"]
                .transform(_rolling_form)
            )
        else:
            feats["driver_form3"] = (
                res_meta
                .assign(_drv=results[driver_key].values, _fin=finish_numeric.values)
                .groupby(["_drv"], dropna=False)["_fin"]
                .transform(_rolling_form)
            )

    # Team recent form
    if team_key:
        if "year" in res_meta.columns:
            feats["team_form3"] = (
                res_meta
                .assign(_tm=results[team_key].values, _fin=finish_numeric.values)
                .groupby(["_tm", "year"], dropna=False)["_fin"]
                .transform(_rolling_form)
            )
        else:
            feats["team_form3"] = (
                res_meta
                .assign(_tm=results[team_key].values, _fin=finish_numeric.values)
                .groupby(["_tm"], dropna=False)["_fin"]
                .transform(_rolling_form)
            )

    # --------------------
    # Qualifying signal: best recorded qual position per driver/race
    # --------------------
    if (
        qualifying is not None and not qualifying.empty
        and driver_key and race_key
    ):
        q_race = _find_col(qualifying, ["race_id", "raceId"])
        q_driver = _find_col(qualifying, ["driver_id", "driverId"])
        q_pos = _find_col(qualifying, ["position", "positionOrder", "pos"])
        if q_race and q_driver and q_pos:
            q = qualifying[[q_race, q_driver, q_pos]].copy()
            q.columns = ["_race", "_driver", "_qpos"]
            q["_qpos"] = pd.to_numeric(q["_qpos"], errors="coerce")
            qbest = q.groupby(["_race", "_driver"], dropna=False)["_qpos"].min().reset_index()

            key_df = pd.DataFrame({
                "_race": results[race_key] if race_key else -1,
                "_driver": results[driver_key] if driver_key else -1,
            })
            qjoin = key_df.merge(qbest, on=["_race", "_driver"], how="left")
            feats["q_pos_best"] = qjoin["_qpos"]

    # --------------------
    # Finalize feature frame
    # --------------------
    feat_df = pd.DataFrame(feats, index=results.index)

    # Basic numeric imputation
    for c in feat_df.columns:
        if pd.api.types.is_numeric_dtype(feat_df[c]):
            feat_df[c] = feat_df[c].fillna(feat_df[c].mean())
        else:
            feat_df[c] = feat_df[c].fillna(0)

    feat_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=float)

    state: Dict[str, Any] = {
        "feat_names": feat_names,
        "n_samples": len(results),
        "config_used": dict(cfg or {}),
    }

    return X, y.to_numpy(dtype=int), groups.astype(int), feat_names, state