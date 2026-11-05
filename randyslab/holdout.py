"""Sealed out-of-sample holdout shared by RandysLab and QuantumRandy.

Everything before ``HOLDOUT_START`` has been used for selection, tuning or observation by an earlier research pass
(QuantumRandy docs/PHASE1_EVALUATION_SPINE.md). Bars at or after it are dropped by the data loaders unless a
pre-registration opens the holdout. A pre-registration lists every candidate before anyone looks, and the holdout can be
opened with only one of them: a second, different pre-registration is refused. Each opening is written to the trial
ledger.

The holdout is the window [``HOLDOUT_START``, ``HOLDOUT_END``): the 56 daily bars of the 2026-11-05 archive refresh.
Bars from ``HOLDOUT_END`` on are forward data, evaluated only by observations registered before them, so opening the
holdout never returns them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import trial_ledger

HOLDOUT_START = pd.Timestamp("2026-09-06T00:00:00Z")
HOLDOUT_END = pd.Timestamp("2026-11-01T00:00:00Z")


class HoldoutSealedError(RuntimeError):
    pass


def seal(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Drop holdout bars (index at or after ``HOLDOUT_START``)."""
    return frame[frame.index < HOLDOUT_START]


def holdout_rows(frame: pd.DataFrame | pd.Series) -> int:
    return int((frame.index >= HOLDOUT_START).sum())


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Read and validate a pre-registration: a JSON object with a non-empty ``candidates`` list and a ``hypothesis``."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or not payload.get("candidates") or not str(payload.get("hypothesis", "")).strip():
        raise ValueError("pre-registration needs a non-empty 'candidates' list and a 'hypothesis'")
    return {**payload, "sha256": hashlib.sha256(raw).hexdigest(), "path": Path(path).as_posix()}


def open_holdout(
    frame: pd.DataFrame,
    preregistration_path: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return ``frame`` through the holdout window (bars before ``HOLDOUT_END``), once per pre-registration, and record
    the access.

    ``ledger_path`` defaults to the active trial ledger; opening is refused when no ledger is active, because an
    unrecorded opening could not be checked later.
    """
    prereg = load_preregistration(preregistration_path)
    ledger = Path(ledger_path) if ledger_path else trial_ledger.active_path()
    if ledger is None:
        raise HoldoutSealedError("configure a trial ledger before opening the holdout")
    previous = {
        row.get("extra_preregistration_sha256")
        for row in (trial_ledger.load(ledger) if ledger.exists() else [])
        if row.get("source") == "holdout_open"
    }
    previous.discard(None)
    if previous and prereg["sha256"] not in previous:
        raise HoldoutSealedError(
            "the holdout was already opened with a different pre-registration; it is no longer a clean holdout"
        )
    if trial_ledger.active_path() != ledger:
        trial_ledger.configure(ledger)
    trial_ledger.record(
        source="holdout_open",
        candidate={"preregistration": prereg["path"], "candidates": prereg["candidates"]},
        window={"start": HOLDOUT_START.isoformat(), "end": HOLDOUT_END.isoformat()},
        holdout_access=True,
        extra={"extra_preregistration_sha256": prereg["sha256"], "repeat_access": bool(previous)},
    )
    return frame[frame.index < HOLDOUT_END]
