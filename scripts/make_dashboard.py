from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoquant.config import load_settings
from autoquant.data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from autoquant.engine import run_strict4h
from autoquant.metrics import equity_curve, summarize


def load_theta(path: Path, candidate_index: int) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw[candidate_index] if raw else {}
    return raw


def ledger_records(ledger: pd.DataFrame) -> list[dict]:
    equity = equity_curve(ledger["r_net"])
    drawdown = equity / equity.cummax() - 1.0
    rows = ledger.reset_index().copy()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    rows["equity"] = equity.to_numpy()
    rows["drawdown"] = drawdown.to_numpy()
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "funding_rate",
        "signal",
        "exposure",
        "delta_exposure",
        "r_net",
        "equity",
        "drawdown",
    ]
    return rows[cols].round(8).to_dict("records")


def read_table(path: Path, limit: int = 12) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).head(limit).to_dict("records")


def dashboard_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AutoQuant STRICT4H Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #181a20;
      --panel: #1e2329;
      --panel-2: #0b0e11;
      --ink: #eaecef;
      --muted: #848e9c;
      --line: #2b3139;
      --green: #0ecb81;
      --red: #f6465d;
      --yellow: #f0b90b;
      --blue: #2f8cff;
      --shade: rgba(240, 185, 11, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    .sub { margin-top: 6px; color: var(--muted); font-size: 13px; }
    main { padding: 18px 28px 32px; max-width: 1500px; margin: 0 auto; }
    .grid { display: grid; gap: 14px; }
    .metrics { grid-template-columns: repeat(6, minmax(130px, 1fr)); }
    .metric, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric { padding: 12px 13px; min-height: 74px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 22px; margin-top: 8px; font-weight: 700; }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr 220px 220px;
      gap: 12px;
      align-items: end;
      margin: 16px 0;
    }
    .field label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
    input[type="range"] { width: 100%; accent-color: var(--yellow); }
    select, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--panel-2);
      color: var(--ink);
    }
    section { padding: 14px; margin-top: 14px; }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    canvas { width: 100%; display: block; background: var(--panel-2); border-radius: 6px; }
    #priceCanvas { height: 520px; }
    #equityCanvas { height: 260px; }
    #ddCanvas { height: 210px; }
    .two { grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr); align-items: start; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    code { white-space: pre-wrap; word-break: break-word; color: #d8dee9; }
    .note { color: var(--muted); line-height: 1.55; font-size: 13px; }
    @media (max-width: 980px) {
      main { padding: 14px; }
      .metrics, .two, .toolbar { grid-template-columns: 1fr; }
      #priceCanvas { height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>AutoQuant STRICT4H 策略实验室</h1>
    <div class="sub" id="subtitle"></div>
  </header>
  <main>
    <div class="grid metrics" id="metrics"></div>
    <div class="toolbar">
      <div class="field">
        <label>时间窗口</label>
        <input id="range" type="range" min="120" max="1200" value="420" step="20" />
      </div>
      <div class="field">
        <label>显示K线根数</label>
        <input id="bars" type="number" min="120" max="2000" value="420" step="20" />
      </div>
      <div class="field">
        <label>窗口位置</label>
        <select id="position">
          <option value="end">最新</option>
          <option value="train">训练期末</option>
          <option value="validation">验证期末</option>
          <option value="full">全样本抽样</option>
        </select>
      </div>
    </div>
    <section>
      <div class="section-head"><b>K线、持仓与交易点</b><span>币圈配色：绿涨红跌；绿色上箭头=增加多头，红色下箭头=增加空头</span></div>
      <canvas id="priceCanvas"></canvas>
    </section>
    <div class="grid two">
      <section>
        <div class="section-head"><b>权益曲线</b><span>从 1.0 开始复利</span></div>
        <canvas id="equityCanvas"></canvas>
      </section>
      <section>
        <div class="section-head"><b>当前参数 theta</b><span>Stage I 候选，仅供研究</span></div>
        <code id="theta"></code>
      </section>
    </div>
    <section>
      <div class="section-head"><b>回撤曲线</b><span>黄色阴影和竖线标出当前回测展示段的开始与结束</span></div>
      <canvas id="ddCanvas"></canvas>
    </section>
    <div class="grid two">
      <section>
        <div class="section-head"><b>Stage I 前几名</b><span>训练窗口里看起来最好的参数</span></div>
        <div id="trials"></div>
      </section>
      <section>
        <div class="section-head"><b>Stage II 稳健性筛选</b><span>越能扛成本和验证期越好</span></div>
        <div id="robust"></div>
      </section>
    </div>
    <section class="note">
      这不是实盘交易面板。论文的思路是“实验室/质检台”：先用严格执行、成本、资金费率、验证窗口和压力测试筛策略，筛不过就继续调教。
    </section>
  </main>
  <script>
    const payload = __PAYLOAD__;
    const rows = payload.rows;
    const windows = payload.windows;

    function fmtPct(x) { return (x * 100).toFixed(2) + "%"; }
    function fmtNum(x) { return Number(x).toFixed(3); }
    function metric(label, value) { return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`; }
    function setupCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: rect.width, h: rect.height };
    }
    function grid(ctx, w, h) {
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#2b3139";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const y = h * i / 5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    }
    function domain(data, key) {
      let min = Infinity, max = -Infinity;
      for (const d of data) {
        min = Math.min(min, d[key]);
        max = Math.max(max, d[key]);
      }
      return [min, max];
    }
    function yScale(min, max, h, pad=20) {
      if (max === min) max = min + 1;
      return v => pad + (max - v) / (max - min) * (h - pad * 2);
    }
    function windowRows() {
      const n = Math.min(+document.getElementById("bars").value || 420, rows.length);
      const pos = document.getElementById("position").value;
      if (pos === "full") {
        const step = Math.max(1, Math.floor(rows.length / n));
        return rows.filter((_, i) => i % step === 0).slice(0, n);
      }
      let end = rows.length;
      if (pos === "train") end = rows.findIndex(r => r.timestamp >= windows.training_end.replace("T", " ").slice(0, 16));
      if (pos === "validation") end = rows.findIndex(r => r.timestamp >= windows.validation_end.replace("T", " ").slice(0, 16));
      if (end <= 0) end = rows.length;
      return rows.slice(Math.max(0, end - n), end);
    }
    function drawPrice() {
      const data = windowRows();
      const {ctx, w, h} = setupCanvas(document.getElementById("priceCanvas"));
      grid(ctx, w, h);
      const lo = Math.min(...data.map(d => d.low));
      const hi = Math.max(...data.map(d => d.high));
      const y = yScale(lo, hi, h, 26);
      const step = w / data.length;
      const bodyW = Math.max(2, Math.min(8, step * 0.62));
      data.forEach((d, i) => {
        const x = i * step + step / 2;
        const up = d.close >= d.open;
        ctx.strokeStyle = up ? "#0ecb81" : "#f6465d";
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(x, y(d.high));
        ctx.lineTo(x, y(d.low));
        ctx.stroke();
        const top = y(Math.max(d.open, d.close));
        const bot = y(Math.min(d.open, d.close));
        ctx.fillRect(x - bodyW / 2, top, bodyW, Math.max(1, bot - top));
        if (d.delta_exposure > 0) drawTriangle(ctx, x, y(d.low) + 10, "#0ecb81", "up");
        if (d.delta_exposure < 0) drawTriangle(ctx, x, y(d.high) - 10, "#f6465d", "down");
      });
      ctx.strokeStyle = "#f0b90b";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      data.forEach((d, i) => {
        const x = i * step + step / 2;
        const yy = h - 52 - d.exposure * 24;
        if (i === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
      });
      ctx.stroke();
      label(ctx, data[0].timestamp + "  →  " + data[data.length - 1].timestamp, 10, 18);
      label(ctx, "price " + lo.toFixed(1) + " - " + hi.toFixed(1), 10, h - 10);
    }
    function drawTriangle(ctx, x, y, color, dir) {
      ctx.fillStyle = color;
      ctx.beginPath();
      if (dir === "up") {
        ctx.moveTo(x, y);
        ctx.lineTo(x - 5, y + 11);
        ctx.lineTo(x + 5, y + 11);
      } else {
        ctx.moveTo(x, y);
        ctx.lineTo(x - 5, y - 11);
        ctx.lineTo(x + 5, y - 11);
      }
      ctx.fill();
    }
    function label(ctx, text, x, y) {
      ctx.fillStyle = "#848e9c";
      ctx.font = "12px Segoe UI";
      ctx.fillText(text, x, y);
    }
    function drawLine(canvasId, key, color, name, highlight=false) {
      const data = windowRows();
      const {ctx, w, h} = setupCanvas(document.getElementById(canvasId));
      grid(ctx, w, h);
      if (highlight) {
        ctx.fillStyle = "rgba(240, 185, 11, 0.10)";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "#f0b90b";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0.5, 0); ctx.lineTo(0.5, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(w - 0.5, 0); ctx.lineTo(w - 0.5, h); ctx.stroke();
      }
      const [min, max] = domain(data, key);
      const y = yScale(min, max, h, 22);
      const step = w / Math.max(data.length - 1, 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((d, i) => {
        const x = i * step;
        const yy = y(d[key]);
        if (i === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
      });
      ctx.stroke();
      label(ctx, `${name} ${min.toFixed(3)} - ${max.toFixed(3)}`, 10, 18);
    }
    function table(data, cols) {
      if (!data.length) return "<div class='note'>暂无数据</div>";
      return `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        data.map(r => `<tr>${cols.map(c => `<td>${String(r[c] ?? "").slice(0, 80)}</td>`).join("")}</tr>`).join("") +
        "</tbody></table>";
    }
    function render() {
      document.getElementById("subtitle").textContent = `${payload.symbol} · ${payload.start} 至 ${payload.end} · ${payload.note}`;
      document.getElementById("metrics").innerHTML = [
        metric("CAGR", fmtPct(payload.summary.cagr)),
        metric("月几何收益", fmtPct(payload.summary.monthly_geom)),
        metric("Sharpe", fmtNum(payload.summary.sharpe)),
        metric("最大回撤", fmtPct(payload.summary.max_dd)),
        metric("交易次数", payload.summary.trades),
        metric("换手密度", fmtPct(payload.summary.switch_density)),
      ].join("");
      document.getElementById("theta").textContent = JSON.stringify(payload.theta, null, 2);
      document.getElementById("trials").innerHTML = table(payload.trials, ["trial", "score", "monthly_geom", "sharpe", "max_dd", "trades"]);
      document.getElementById("robust").innerHTML = table(payload.robust, ["candidate_id", "mean_monthly_true", "min_monthly_true", "maxDD_mean", "switch_density_mean"]);
      drawPrice();
      drawLine("equityCanvas", "equity", "#2f8cff", "equity");
      drawLine("ddCanvas", "drawdown", "#f6465d", "drawdown", true);
    }
    document.getElementById("range").addEventListener("input", e => {
      document.getElementById("bars").value = e.target.value;
      render();
    });
    document.getElementById("bars").addEventListener("change", render);
    document.getElementById("position").addEventListener("change", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strict4h.yaml")
    ap.add_argument("--theta", default="reports/stage1/top_candidates.json")
    ap.add_argument("--candidate-index", type=int, default=0)
    ap.add_argument("--window", choices=["all", "training", "validation", "long", "blind"], default="all")
    ap.add_argument("--out", default="reports/dashboard.html")
    args = ap.parse_args()

    settings = load_settings(args.config)
    ohlcv = read_ohlcv(settings.paths.ohlcv_csv)
    funding = align_funding_to_ohlcv(ohlcv, read_funding(settings.paths.funding_csv))
    if args.window != "all":
        start = getattr(settings.windows, f"{args.window}_start")
        end = getattr(settings.windows, f"{args.window}_end")
        ohlcv = slice_window(ohlcv, start, end)
        funding = funding.reindex(ohlcv.index)

    theta = load_theta(Path(args.theta), args.candidate_index)
    ledger = run_strict4h(ohlcv, funding, theta, settings.costs)
    payload = {
        "symbol": settings.symbol,
        "start": ledger.index.min().strftime("%Y-%m-%d"),
        "end": ledger.index.max().strftime("%Y-%m-%d"),
        "note": "Stage I 候选参数全样本观察，不代表可实盘",
        "windows": settings.windows.__dict__,
        "theta": theta,
        "summary": summarize(ledger, settings.risk_free_rate),
        "rows": ledger_records(ledger),
        "trials": read_table(Path("reports/stage1/trials.csv")),
        "robust": read_table(Path("reports/stage2/robust_summary.csv")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dashboard_html(payload), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
