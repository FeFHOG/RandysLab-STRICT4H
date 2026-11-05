"""Append-only ledger of candidate evaluations, so selection statistics can count every trial.

Deflated Sharpe and PBO need the true number of candidates tried, including rejects, rewrites, hand-built cohorts and
parameter-search trials. Purged zoos and unsaved search trees lost that count before 2026-11.

Recording is a no-op until ``configure`` is called (entry-point scripts do this; tests do not) or the
``RANDY_TRIAL_LEDGER`` environment variable names a ledger file. Each record is one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENV_VAR = "RANDY_TRIAL_LEDGER"

_lock = threading.Lock()
_state: dict[str, Any] = {"path": None, "run_id": None, "context": {}}


def configure(path: str | Path, *, run_id: str | None = None, context: dict[str, Any] | None = None) -> Path:
    """Start recording to ``path`` for this process. ``context`` (e.g. repo, code_version, symbol) goes on every row."""
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _state["path"] = ledger_path
        _state["run_id"] = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        _state["context"] = dict(context or {})
    return ledger_path


def configure_default(
    repo_dir: str | Path,
    *,
    run_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> Path:
    """Record to ``<repo_dir>/reports/trial_ledger/trials.jsonl`` unless ``RANDY_TRIAL_LEDGER`` points elsewhere."""
    repo = Path(repo_dir).resolve()
    path = Path(os.environ.get(ENV_VAR) or repo / "reports" / "trial_ledger" / "trials.jsonl")
    base = {"repo": repo.name, "code_version": code_version(repo)}
    return configure(path, run_id=run_id, context={**base, **(context or {})})


def disable() -> None:
    with _lock:
        _state.update({"path": None, "run_id": None, "context": {}})


def active_path() -> Path | None:
    if _state["path"] is not None:
        return _state["path"]
    env = os.environ.get(ENV_VAR)
    return Path(env) if env else None


def record(
    *,
    source: str,
    candidate: str | dict[str, Any],
    window: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool | None = None,
    holdout_access: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append one evaluation. Returns the row, or None when no ledger is active."""
    path = active_path()
    if path is None:
        return None
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": _state["run_id"],
        "source": source,
        "candidate_id": candidate_id(candidate),
        "candidate": candidate,
        "window": window or {},
        "passed": passed,
        "holdout_access": bool(holdout_access),
        "metrics": _clean(metrics or {}),
        **_state["context"],
        **(extra or {}),
    }
    line = json.dumps(row, ensure_ascii=True, default=str)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return row


def load(path: str | Path) -> list[dict[str, Any]]:
    """Read a ledger, skipping a truncated last line if a writer was interrupted."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for selection statistics: evaluations, distinct candidates, and both per source."""
    by_source = Counter(row.get("source", "") for row in rows)
    distinct_by_source: dict[str, set[str]] = {}
    for row in rows:
        distinct_by_source.setdefault(row.get("source", ""), set()).add(row.get("candidate_id", ""))
    return {
        "evaluations": len(rows),
        "distinct_candidates": len({row.get("candidate_id", "") for row in rows}),
        "holdout_accesses": sum(1 for row in rows if row.get("holdout_access")),
        "evaluations_by_source": dict(by_source),
        "distinct_candidates_by_source": {source: len(ids) for source, ids in distinct_by_source.items()},
    }


def candidate_id(candidate: str | dict[str, Any]) -> str:
    text = candidate if isinstance(candidate, str) else json.dumps(candidate, sort_keys=True, default=str)
    return "c_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def code_version(repo_dir: str | Path) -> str:
    """Short git commit of ``repo_dir`` with ``-dirty`` when the worktree has changes; ``unknown`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": file_path.as_posix(),
        "bytes": file_path.stat().st_size,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }


def _clean(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            out[key] = value if isinstance(value, str | bool) or value is None else str(value)
            continue
        out[key] = number if math.isfinite(number) else None
    return out
