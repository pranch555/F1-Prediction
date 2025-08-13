

"""
Models package bootstrap + registry.

Usage:

    # 1) Register models in their own files:
    # src/f1pred/models/logistic_regression.py
    #
    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.linear_model import LogisticRegression
    # from . import register
    #
    # @register("logreg")
    # def build(C=1.0, max_iter=2000, class_weight="balanced"):
    #     return Pipeline([
    #         ("scale", StandardScaler()),
    #         ("clf", LogisticRegression(C=C, max_iter=max_iter,
    #                                    class_weight=class_weight, solver="lbfgs")),
    #     ])
    #
    # 2) In your code:
    # from f1pred.models import discover, build_model, list_models
    # discover()
    # model = build_model("logreg", C=0.8)

This file stays lightweight and only exposes a small registry + discovery.
"""

from __future__ import annotations

import importlib
import pkgutil
import pathlib
from typing import Callable, Dict, List

# -----------------------------------------------------------------------------
# Global registry
# -----------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[..., object]] = {}


def register(name: str):
    """
    Decorator to register a model builder under a short name.

    Example:
        @register("rf")
        def build_random_forest(**kwargs) -> sklearn.base.BaseEstimator:
            return RandomForestClassifier(**kwargs)
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register(name): name must be a non-empty string")

    def deco(builder: Callable[..., object]):
        if not callable(builder):
            raise TypeError("register() expects a callable builder")
        if name in REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        REGISTRY[name] = builder
        return builder

    return deco


def build_model(name: str, **params):
    """
    Instantiate an unfitted model by registry name.
    Call `discover()` once at startup so all submodules can register themselves.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Did you call discover()? "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name](**params)


def list_models() -> List[str]:
    """Return a sorted list of available registry names."""
    return sorted(REGISTRY.keys())


def discover() -> Dict[str, Callable[..., object]]:
    """
    Auto-import all submodules in this package so their @register decorators run.

    This will import every *.py file in `src/f1pred/models/` except:
    - __init__.py
    - files starting with underscore '_'

    Returns the REGISTRY dict for convenience.
    """
    pkg_path = pathlib.Path(__file__).parent
    for mod in pkgutil.iter_modules([str(pkg_path)]):
        if mod.ispkg:
            # allow nested packages if you add them later
            importlib.import_module(f"{__name__}.{mod.name}")
            continue
        if mod.name in {"__init__"} or mod.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
    return REGISTRY


__all__ = [
    "register",
    "build_model",
    "list_models",
    "discover",
    "REGISTRY",
]