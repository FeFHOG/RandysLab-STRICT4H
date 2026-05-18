from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CostProfile:
    taker_bps: float = 4.0
    slippage_bps: float = 2.0
    funding_multiplier: float = 1.0
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0


@dataclass(frozen=True)
class Windows:
    training_start: str = "2019-09-08"
    training_end: str = "2021-01-01"
    validation_start: str = "2021-01-01"
    validation_end: str = "2023-01-01"
    long_start: str = "2019-09-08"
    long_end: str = "2025-10-14"
    blind_start: str = "2024-01-01"
    blind_end: str = "2025-11-24"


@dataclass(frozen=True)
class Paths:
    ohlcv_csv: str = "data/BTCUSDT_4h.csv"
    funding_csv: str = "data/BTCUSDT_funding.csv"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class Screening:
    top_k_stage1: int = 40
    stable_k: int = 5
    min_mean_monthly_true: float = 0.005
    min_monthly_true: float = 0.0
    max_dd_mean: float = 0.30
    max_switch_density_mean: float = 0.12
    scenario_taker_bps: tuple[float, ...] = (3.0, 4.0, 6.0)
    scenario_funding_mult: tuple[float, ...] = (0.5, 1.0, 1.5)


@dataclass(frozen=True)
class Settings:
    symbol: str = "BTC/USDT:USDT"
    exchange: str = "binanceusdm"
    initial_equity: float = 10_000.0
    bar_hours: float = 4.0
    risk_free_rate: float = 0.03
    n_trials: int = 120
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    costs: CostProfile = field(default_factory=CostProfile)
    windows: Windows = field(default_factory=Windows)
    screening: Screening = field(default_factory=Screening)


def _merge_dataclass(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    return cls(**raw)


def load_settings(path: str | Path) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(
        symbol=raw.get("symbol", Settings.symbol),
        exchange=raw.get("exchange", Settings.exchange),
        initial_equity=raw.get("initial_equity", Settings.initial_equity),
        bar_hours=raw.get("bar_hours", Settings.bar_hours),
        risk_free_rate=raw.get("risk_free_rate", Settings.risk_free_rate),
        n_trials=raw.get("n_trials", Settings.n_trials),
        seed=raw.get("seed", Settings.seed),
        paths=_merge_dataclass(Paths, raw.get("paths")),
        costs=_merge_dataclass(CostProfile, raw.get("costs")),
        windows=_merge_dataclass(Windows, raw.get("windows")),
        screening=_merge_dataclass(Screening, raw.get("screening")),
    )
