from __future__ import annotations
from pathlib import Path

__version__ = "0.1.0"

def package_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)

# Optional: only list the names you intend to proxy
_LAZY_EXPORTS = {
    "set_seed", "get_logger", "timer", "timeit",
    "add_common_args", "load_config", "output_dir_from_cfg", "save_artifacts",
    "save_joblib", "load_joblib", "save_json", "load_json", "save_csv", "read_csv",
    "make_group_kfold", "scoring_classification", "group_sizes_from_labels",
    "summarize_binary_classification", "validate_columns", "reduce_mem_usage",
    "make_race_key", "add_prev_wins", "chronological_sort",
}

def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from . import utils as _u  # imported only on first access
        try:
            return getattr(_u, name)
        except AttributeError as e:
            raise AttributeError(f"'f1pred' has no attribute '{name}'. "
                                 f"Define it in f1pred.utils.") from e
    raise AttributeError(f"module 'f1pred' has no attribute '{name}'")

__all__ = ["__version__", "package_path", *_LAZY_EXPORTS]