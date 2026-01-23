from __future__ import annotations

import os
from typing import Any

import torch.nn as nn

from .backend import QuantizeBackend
from .model.quantize import quantize_model


def _parse_env_quantize_backend() -> QuantizeBackend | None:
    val = os.getenv("FOUROVERSIX_QUANTIZE_BACKEND")
    if not val:
        return None

    val = val.strip().lower()
    if val in {"auto", "none"}:
        return None

    if val in {"real", "kernel", "kernels"}:
        # Prefer fast kernels when possible.
        # Note: QuantizeBackend.cuda is currently disabled in backend.py, so triton
        # is effectively the only "real" kernel option today.
        return QuantizeBackend.triton

    if val in {"pseudo", "ref", "reference"}:
        return QuantizeBackend.pytorch

    try:
        return QuantizeBackend(val)
    except Exception as e:  # noqa: BLE001
        msg = (
            "Invalid FOUROVERSIX_QUANTIZE_BACKEND value: "
            f"{val!r}. Expected one of: auto/none, real, pseudo, "
            f"or {', '.join([b.value for b in QuantizeBackend])}."
        )
        raise ValueError(msg) from e


def apply_ptq(
    model: nn.Module,
    *,
    exclude_layers: list[str] | None = None,
    quantize_backend: QuantizeBackend | None = None,
    a_quantize_kwargs: dict[str, Any] | None = None,
    w_quantize_kwargs: dict[str, Any] | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """Apply post-training quantization (PTQ) to a model.

    This is a small convenience wrapper around `quantize_model`.

    Backend selection priority:
    1) explicit `quantize_backend`
    2) env var `FOUROVERSIX_QUANTIZE_BACKEND`
    3) default auto-selection (None)
    """

    if a_quantize_kwargs is None:
        a_quantize_kwargs = {}

    if w_quantize_kwargs is None:
        w_quantize_kwargs = {}

    if quantize_backend is None:
        quantize_backend = _parse_env_quantize_backend()

    # Plumb backend into FP4Linear via quantize_model kwargs.
    # Also allow per-(a,w) kwargs for symmetry with README; `quantize_model` currently
    # only has a single `quantize_backend` knob, so we prioritize explicit args.
    if "quantize_backend" not in kwargs:
        kwargs["quantize_backend"] = quantize_backend

    quantize_model(
        model,
        exclude_layers=exclude_layers,
        **kwargs,
    )
