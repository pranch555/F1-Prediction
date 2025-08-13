# src/f1pred/data_ingest.py
"""
Data ingestion utilities for the F1 Prediction project.

Reads raw CSVs into pandas DataFrames based on file names declared
in the YAML config and normalizes a few schema details so downstream
code can assume consistent join keys.

Contract:
    load_tables(cfg) -> dict[str, pd.DataFrame]

Required:
    - A table named 'results' must be present.
      It must include ids: raceId, driverId, constructorId.
      If a finishing position exists (position/positionOrder/positionText/etc.),
      we expose a numeric 'finish_position' helper column without deleting originals.
      If 'year' / 'Grand Prix' / 'Code' are missing, we try to derive them by
      joining with 'races' and 'drivers' tables when present.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


_FINISH_CANDIDATES: List[str] = [
    "finish_position", "finishPosition", "finish_pos",
    "position", "positionOrder", "pos", "Position",
    "positionText",
]


def _maybe_parse_dates(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Parse any date-like columns listed in `cols` if they exist."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _coerce_finish_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric 'finish_position' column when possible,
    leaving original columns intact.
    """
    if "finish_position" in df.columns and pd.api.types.is_numeric_dtype(df["finish_position"]):
        return df  # already good

    for cand in _FINISH_CANDIDATES:
        if cand in df.columns:
            s = df[cand].copy()

            # Special handling for Ergast 'positionText' which may contain 'R', 'DQ', 'W', etc.
            if cand == "positionText":
                # Try numeric; non-numeric (e.g., 'R') -> NaN
                s = pd.to_numeric(s, errors="coerce")

            # General numeric coercion for any other stringy candidate
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors="coerce")

            if "finish_position" not in df.columns:
                df["finish_position"] = s
            else:
                # prefer the first usable series; if finish_position exists but is all NaN,
                # and new candidate has values, fillna from it.
                df["finish_position"] = df["finish_position"].fillna(s)

            # We do NOT drop/rename the original column; keep it available.
            break

    # Small cleanup: if someone stored zero as winner, normalize to 1-based outside (feature builder clips).
    return df


def _normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Light normalization of potential header variants and convenience columns."""
    # Driver code can arrive under different headers
    if "Name Code" in df.columns and "Code" not in df.columns:
        df = df.rename(columns={"Name Code": "Code"})
    for alt in ("Driver Code", "driver_code"):
        if alt in df.columns and "Code" not in df.columns:
            df = df.rename(columns={alt: "Code"})

    # Race name normalization to 'Grand Prix' if it appears directly in results (rare)
    for alt in ("race_name", "Race", "grand_prix", "GrandPrix", "name"):
        if alt in df.columns and "Grand Prix" not in df.columns:
            df = df.rename(columns={alt: "Grand Prix"})

    # Derive 'finish_position' helper
    df = _coerce_finish_position(df)

    # Derive year from a date column if present
    if "year" not in df.columns:
        for dcol in ("Date", "date"):
            if dcol in df.columns:
                df["year"] = pd.to_datetime(df[dcol], errors="coerce").dt.year
                break

    return df


def load_tables(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load tables declared in cfg['data']['archive_files'] from cfg['data']['raw_dir'].

    - 'results' is required and must include raceId/driverId/constructorId.
    - Finishing position columns are optional, but strongly recommended.
    - If 'races'/'drivers' are loaded too, we augment results with:
        * 'year' and 'Grand Prix' from races (name -> Grand Prix)
        * 'Code' from drivers (driver code)
    """
    data_cfg = cfg.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "Data/raw"))
    files = dict(data_cfg.get("archive_files", {}))

    # Back-compat with older minimal configs (optional)
    if not files:
        if "winners_csv" in data_cfg:
            win = data_cfg["winners_csv"]
            files["results"] = win if os.path.isabs(win) else Path(win).name
        if "fastest_laps_csv" in data_cfg:
            fl = data_cfg["fastest_laps_csv"]
            files["fastest_laps"] = fl if os.path.isabs(fl) else Path(fl).name

    parse_dates = list(data_cfg.get("parse_dates", []))

    tables: Dict[str, pd.DataFrame] = {}
    for name, rel in files.items():
        path = Path(rel)
        if not path.is_absolute():
            path = raw_dir / rel

        if not path.exists():
            if name == "results":
                raise FileNotFoundError(f"Missing CSV for required table 'results': {path}")
            print(f"[ingest] WARNING: missing CSV for optional table '{name}': {path} — skipping.")
            continue

        df = pd.read_csv(path)
        if parse_dates:
            df = _maybe_parse_dates(df, parse_dates)
        if name == "results":
            df = _normalize_results(df)

        tables[name] = df

    if "results" not in tables:
        raise ValueError("'results' table is required to anchor rows for X/y.")

    # Minimal schema check on results: ids must be present
    res = tables["results"]
    base_required = {"raceId", "driverId", "constructorId"}
    missing_base = [c for c in base_required if c not in res.columns]
    if missing_base:
        raise ValueError(
            f"'results' table is missing required id columns: {missing_base}. Present: {list(res.columns)}"
        )

    # Optional augmentation: add 'year' + 'Grand Prix' from races
    races = tables.get("races")
    if races is not None:
        # Ergast races.csv: raceId,year,round,circuitId,name,date,time,url
        needed = {"raceId", "year", "name"}
        if needed.issubset(races.columns):
            res = res.merge(races[["raceId", "year", "name"]], on="raceId", how="left")
            res = res.rename(columns={"name": "Grand Prix"})

    # Optional augmentation: add 'Code' from drivers
    drivers = tables.get("drivers")
    if drivers is not None:
        # Typical drivers.csv: driverId,driverRef,number,code,forename,surname,...
        if {"driverId", "code"}.issubset(drivers.columns):
            res = res.merge(drivers[["driverId", "code"]], on="driverId", how="left")
            res = res.rename(columns={"code": "Code"})

    # Ensure we expose a 'finish_position' if any candidate exists
    res = _coerce_finish_position(res)

    # As a guardrail, require that at least one finish-related column exists.
    if not any(c in res.columns for c in ["finish_position", "position", "positionOrder", "positionText"]):
        raise ValueError(
            "'results' has no finishing position field (finish_position/position/positionOrder/positionText). "
            "Point your config to the full results.csv, not winners.csv."
        )

    tables["results"] = res
    return tables