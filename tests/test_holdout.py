from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from randyslab import trial_ledger
from randyslab.data import load_market
from randyslab.holdout import HOLDOUT_START, HoldoutSealedError, open_holdout, seal
from randyslab.regimes import outside_regimes, regime_labels


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv(trial_ledger.ENV_VAR, raising=False)
    trial_ledger.disable()
    yield
    trial_ledger.disable()


def _write_market(tmp_path: Path) -> tuple[Path, Path]:
    index = pd.date_range(HOLDOUT_START - pd.Timedelta(hours=16), periods=8, freq="4h", tz="UTC")
    ohlcv = pd.DataFrame({"timestamp": index, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    ohlcv.to_csv(tmp_path / "ohlcv.csv", index=False)
    pd.DataFrame({"timestamp": index[::2], "funding_rate": 1e-4}).to_csv(tmp_path / "funding.csv", index=False)
    return tmp_path / "ohlcv.csv", tmp_path / "funding.csv"


def _prereg(tmp_path: Path, name: str, candidates: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"hypothesis": "funding carry", "candidates": candidates}), encoding="utf-8")
    return path


def test_seal_and_loader_drop_holdout_bars_by_default(tmp_path: Path):
    ohlcv_path, funding_path = _write_market(tmp_path)
    ohlcv, funding = load_market(ohlcv_path, funding_path)
    assert len(ohlcv) == 4 and ohlcv.index.max() < HOLDOUT_START
    assert funding.index.equals(ohlcv.index)
    frame = pd.DataFrame({"x": range(3)}, index=pd.DatetimeIndex([HOLDOUT_START - pd.Timedelta(hours=4), HOLDOUT_START,
                                                                   HOLDOUT_START + pd.Timedelta(hours=4)]))
    assert len(seal(frame)) == 1


def test_opening_requires_a_ledger_and_a_valid_preregistration(tmp_path: Path):
    ohlcv_path, funding_path = _write_market(tmp_path)
    good = _prereg(tmp_path, "prereg.json", ["neg(zscore(funding_rate,42))"])
    with pytest.raises(HoldoutSealedError, match="trial ledger"):
        load_market(ohlcv_path, funding_path, preregistration=good)
    trial_ledger.configure(tmp_path / "trials.jsonl")
    (tmp_path / "empty.json").write_text(json.dumps({"hypothesis": "x", "candidates": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="candidates"):
        load_market(ohlcv_path, funding_path, preregistration=tmp_path / "empty.json")


def test_holdout_opens_once_per_preregistration_and_is_recorded(tmp_path: Path):
    ohlcv_path, funding_path = _write_market(tmp_path)
    ledger = trial_ledger.configure(tmp_path / "trials.jsonl")
    first = _prereg(tmp_path, "first.json", ["a"])
    ohlcv, _ = load_market(ohlcv_path, funding_path, preregistration=first)
    assert len(ohlcv) == 8
    load_market(ohlcv_path, funding_path, preregistration=first)  # re-reading the same pre-registration is allowed
    with pytest.raises(HoldoutSealedError, match="different pre-registration"):
        open_holdout(ohlcv, _prereg(tmp_path, "second.json", ["b"]))
    opens = [row for row in trial_ledger.load(ledger) if row["source"] == "holdout_open"]
    assert [row["repeat_access"] for row in opens] == [False, True]
    assert all(row["holdout_access"] for row in opens)


def test_regime_labels():
    index = pd.date_range("2025-10-09", "2025-10-13", freq="4h", tz="UTC")
    labels = regime_labels(index)
    inside = labels.eq("liquidation_cascade_2025_10_10")
    assert inside.sum() == 12
    assert outside_regimes(index).equals(~inside)
