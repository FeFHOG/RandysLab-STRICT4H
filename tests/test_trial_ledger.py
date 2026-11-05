from __future__ import annotations

import threading
from pathlib import Path

import pytest

from randyslab import trial_ledger


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv(trial_ledger.ENV_VAR, raising=False)
    trial_ledger.disable()
    yield
    trial_ledger.disable()


def test_record_is_a_no_op_until_configured(tmp_path: Path):
    assert trial_ledger.record(source="test", candidate="zscore(close,10)") is None
    assert not any(tmp_path.iterdir())


def test_records_are_appended_with_context_and_clean_metrics(tmp_path: Path):
    path = trial_ledger.configure(tmp_path / "ledger" / "trials.jsonl", run_id="r1", context={"repo": "QR"})
    trial_ledger.record(
        source="qr_mcts",
        candidate="zscore(close,10)",
        metrics={"sharpe": 0.5, "ic": float("nan"), "label": "x"},
        passed=False,
    )
    trial_ledger.record(source="rl_stage1_tpe", candidate={"ema_fast": 12, "ema_slow": 26}, holdout_access=True)
    rows = trial_ledger.load(path)
    assert [row["source"] for row in rows] == ["qr_mcts", "rl_stage1_tpe"]
    assert rows[0]["run_id"] == "r1" and rows[0]["repo"] == "QR"
    assert rows[0]["metrics"] == {"sharpe": 0.5, "ic": None, "label": "x"}
    assert rows[1]["candidate_id"] == trial_ledger.candidate_id({"ema_slow": 26, "ema_fast": 12})


def test_environment_variable_enables_recording(tmp_path: Path, monkeypatch):
    target = tmp_path / "env.jsonl"
    monkeypatch.setenv(trial_ledger.ENV_VAR, str(target))
    trial_ledger.record(source="test", candidate="abs(close)")
    assert len(trial_ledger.load(target)) == 1


def test_summary_counts_distinct_candidates_and_holdout_accesses(tmp_path: Path):
    path = trial_ledger.configure(tmp_path / "t.jsonl")
    for formula in ["a", "a", "b"]:
        trial_ledger.record(source="qr_mcts", candidate=formula)
    trial_ledger.record(source="qr_blind_2026", candidate="a", holdout_access=True)
    summary = trial_ledger.summarize(trial_ledger.load(path))
    assert summary["evaluations"] == 4
    assert summary["distinct_candidates"] == 2
    assert summary["holdout_accesses"] == 1
    assert summary["distinct_candidates_by_source"] == {"qr_mcts": 2, "qr_blind_2026": 1}


def test_concurrent_writers_do_not_interleave_lines(tmp_path: Path):
    path = trial_ledger.configure(tmp_path / "t.jsonl")

    def worker(offset: int) -> None:
        for i in range(200):
            trial_ledger.record(source="thread", candidate=f"f{offset}_{i}", metrics={"x": i})

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(trial_ledger.load(path)) == 800


def test_load_skips_a_truncated_last_line(tmp_path: Path):
    path = trial_ledger.configure(tmp_path / "t.jsonl")
    trial_ledger.record(source="s", candidate="a")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"source": "s", "candid')
    assert len(trial_ledger.load(path)) == 1
