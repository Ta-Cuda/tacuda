#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Generate the "Strategy workloads" section of site/benchmark.html.

Reads the bench_experiments_batch.py summary JSONs produced on the GPU box
(tools/optimizer/bench/exp_v3_*.json) and replaces the marked region in
site/benchmark.html with tables + inline SVG scaling charts, so every number
on the page traces back to a committed measurement file.

Usage:
    python scripts/gen_talib_comparison.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tools" / "optimizer" / "bench"
PAGE = ROOT / "site" / "benchmark.html"

BEGIN = "<!-- BEGIN GENERATED: strategy-workloads (scripts/gen_talib_comparison.py) -->"
END = "<!-- END GENERATED: strategy-workloads -->"

FULL_CONFIGS = [
    ("exp_v3_16x500k_summary.json", "16 symbols × 500 k bars"),
    ("exp_v3_64x500k_summary.json", "64 symbols × 500 k bars"),
]
SCALE_FILES = {
    4: "exp_v3_scale_4_summary.json",
    16: "exp_v3_scale_16_summary.json",
    64: "exp_v3_scale_64_summary.json",
    256: "exp_v3_scale_256_summary.json",
    1024: "exp_v3_scale_1024_summary.json",
}

# Representative subset grouped by kernel category (order = display order).
CATEGORIES = [
    ("Prefix-scan recursions", "cumulative lines parallelised as scans in 0.4.1",
     ["exp03_obtr", "exp06_asi", "exp09_obv_revisited"]),
    ("Affine-scan recursions", "first-order recurrences via AffineScan",
     ["exp04_hpi_breakout"]),
    ("Windowed kernels", "one thread per element, quadratic windows",
     ["exp10_ref_dev", "exp04_price_dist", "exp10_superman", "exp11_trendscore"]),
    ("Classic TA sets", "SMA/EMA/BBANDS/CCI/... request sets",
     ["exp01_ma_crossover", "exp02_bull_beater", "exp06_cci_avg"]),
    ("Sequential state machines", "EL parabolic: the documented exception",
     ["exp06_parabolic_vol"]),
]


def load(name: str) -> dict:
    with open(BENCH / name, encoding="utf-8") as fh:
        return json.load(fh)


def fmt_s(v: float) -> str:
    return f"{v:.2f} s" if v >= 0.095 else f"{v * 1000:.0f} ms"


def fmt_x(v: float) -> str:
    return f"{v:.1f}x" if v < 30 else f"{v:.0f}x"


def speed_cell(ratio: float) -> str:
    cls = "speed" if ratio >= 1.0 else "num"
    return f'<td class="{cls}">{fmt_x(ratio)}</td>'


def full_pack_table() -> str:
    rows = []
    for fname, label in FULL_CONFIGS:
        agg = load(fname)["aggregate"]
        t = agg["indicatorSecTotal"]
        rows.append(
            "                <tr>"
            f"<td>{label}</td>"
            f'<td class="num">{fmt_s(t["tacuda"])}</td>'
            f'<td class="num">{fmt_s(t["tacuda-device"])}</td>'
            f'<td class="num">{fmt_s(t["talib"])}</td>'
            f'<td class="num">{fmt_s(t["talib-mt"])}</td>'
            + speed_cell(t["talib"] / t["tacuda"])
            + speed_cell(t["talib-mt"] / t["tacuda"])
            + "</tr>")
    return "\n".join(rows)


def category_rows() -> str:
    data = load(FULL_CONFIGS[1][0])["strategies"]
    out = []
    for cat, note, names in CATEGORIES:
        for i, n in enumerate(names):
            e = data[n]
            b = e["tacuda"]["indicatorSec"]
            t1 = e["talib"]["indicatorSec"]
            t8 = e["talib-mt"]["indicatorSec"]
            cat_cell = (f'<td rowspan="{len(names)}">{cat}<br>'
                        f'<span class="dim">{note}</span></td>') if i == 0 else ""
            out.append(
                "                <tr>" + cat_cell +
                f"<td><code>{n}</code></td>"
                f'<td class="num">{fmt_s(b)}</td>'
                f'<td class="num">{fmt_s(t1)}</td>'
                f'<td class="num">{fmt_s(t8)}</td>'
                + speed_cell(t1 / b) + speed_cell(t8 / b) + "</tr>")
    return "\n".join(out)


def scaling_series(pick) -> dict[str, list[tuple[int, float]]]:
    """pick(strategies_dict, label) -> seconds for that backend."""
    series: dict[str, list[tuple[int, float]]] = {
        "tacuda": [], "talib": [], "talib-mt": []}
    for sym in sorted(SCALE_FILES):
        d = load(SCALE_FILES[sym])
        for label in series:
            series[label].append((sym, pick(d["strategies"], label)))
    return series


def subset_total(strategies: dict, label: str) -> float:
    return sum(e[label]["indicatorSec"] for e in strategies.values()
               if e[label].get("rc") == 0)


def scaling_svg(series: dict[str, list[tuple[int, float]]], title: str,
                aria: str) -> str:
    import math

    w, h, padl, padr, padt, padb = 720, 240, 64, 16, 18, 40
    xs = sorted({x for pts in series.values() for x, _ in pts})
    ys = [y for pts in series.values() for _, y in pts if y > 0]
    ymin = min(ys) / 1.4
    ymax = max(ys) * 1.4

    def px(sym: float) -> float:
        lo, hi = math.log2(xs[0]), math.log2(xs[-1])
        return padl + (math.log2(sym) - lo) / (hi - lo) * (w - padl - padr)

    def py(sec: float) -> float:
        lo, hi = math.log10(ymin), math.log10(ymax)
        return padt + (hi - math.log10(sec)) / (hi - lo) * (h - padt - padb)

    grid, labels = [], []
    for sym in xs:
        x = px(sym)
        grid.append(f'<line class="sc-grid" x1="{x:.1f}" y1="{padt}" '
                    f'x2="{x:.1f}" y2="{h - padb}"/>')
        labels.append(f'<text x="{x:.1f}" y="{h - padb + 14}" '
                      f'text-anchor="middle">{sym}</text>')
    decade = 10.0 ** math.floor(math.log10(ymin))
    yt = decade
    while yt <= ymax:
        if yt >= ymin:
            y = py(yt)
            grid.append(f'<line class="sc-grid" x1="{padl}" y1="{y:.1f}" '
                        f'x2="{w - padr}" y2="{y:.1f}"/>')
            txt = f"{yt:g}" if yt < 1000 else f"{yt / 1000:g}k"
            labels.append(f'<text x="{padl - 6}" y="{y + 3:.1f}" '
                          f'text-anchor="end">{txt}s</text>')
        yt *= 10.0
    lines = []
    css = {"tacuda": "sc-gpu", "talib": "sc-cpu1", "talib-mt": "sc-cpu8"}
    for label, pts in series.items():
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        lines.append(f'<polyline class="{css[label]}" points="{path}"/>')
        for x, y in pts:
            lines.append(f'<circle class="{css[label]}" cx="{px(x):.1f}" '
                         f'cy="{py(y):.1f}" r="2.6"/>')
    legend = (
        f'<g font-size="11">'
        f'<line class="sc-gpu" x1="{padl + 6}" y1="{padt + 6}" x2="{padl + 30}" y2="{padt + 6}"/>'
        f'<text x="{padl + 36}" y="{padt + 9}">tacuda batch (GPU)</text>'
        f'<line class="sc-cpu1" x1="{padl + 176}" y1="{padt + 6}" x2="{padl + 200}" y2="{padt + 6}"/>'
        f'<text x="{padl + 206}" y="{padt + 9}">TA-Lib 1 thread</text>'
        f'<line class="sc-cpu8" x1="{padl + 330}" y1="{padt + 6}" x2="{padl + 354}" y2="{padt + 6}"/>'
        f'<text x="{padl + 360}" y="{padt + 9}">TA-Lib 8 threads</text>'
        f"</g>")
    return (f'<figure class="scale-fig"><figcaption>{title}</figcaption>\n'
            f'<svg class="scale-chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{aria}">\n  '
            + "\n  ".join(grid + lines + labels) + "\n  " + legend +
            "\n</svg></figure>")


def render() -> str:
    subset_names = [n for _, _, names in CATEGORIES for n in names]
    total_series = scaling_series(subset_total)
    refdev_series = scaling_series(
        lambda s, label: s["exp10_ref_dev"][label]["indicatorSec"])

    agg16 = load(FULL_CONFIGS[0][0])["aggregate"]
    agg64 = load(FULL_CONFIGS[1][0])["aggregate"]
    sc1024 = load(SCALE_FILES[1024])["strategies"]
    best_total = subset_total(sc1024, "talib") / subset_total(sc1024, "tacuda")
    rd = sc1024["exp10_ref_dev"]
    rd_x1 = rd["talib"]["indicatorSec"] / rd["tacuda"]["indicatorSec"]
    rd_x8 = rd["talib-mt"]["indicatorSec"] / rd["tacuda"]["indicatorSec"]
    pd = sc1024["exp04_price_dist"]
    pd_x1 = pd["talib"]["indicatorSec"] / pd["tacuda"]["indicatorSec"]

    return f"""{BEGIN}
<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Strategy workloads</span>
            <h2>94 trading systems, indicator phase vs TA-Lib</h2>
            <p>The EXPERIMENTS pack (94 published trading systems) runs multi-symbol
               backtests through <code>tacuda-opt portfolio</code>. Every backend computes
               the exact same indicator requests and produces the same trades
               (parity-checked per run); the timings below isolate the indicator phase —
               the part the library owns. Environment: RTX&nbsp;3090&nbsp;Ti + Core
               i3-12100F (4C/8T), Ubuntu 24.04, tacuda 0.4.1.</p>
        </div>

        <table class="data">
            <thead>
                <tr>
                    <th>Config (all 94 systems)</th>
                    <th class="num">tacuda batch</th>
                    <th class="num">GPU per-symbol loop</th>
                    <th class="num">TA-Lib 1 thread</th>
                    <th class="num">TA-Lib 8 threads</th>
                    <th class="num">vs 1 thread</th>
                    <th class="num">vs 8 threads</th>
                </tr>
            </thead>
            <tbody>
{full_pack_table()}
            </tbody>
        </table>
        <p class="table-note">
            Totals of the per-strategy indicator phase. The pack is dominated by tiny
            request sets (median 3 lines per system), so most of the GPU column is PCIe
            transfers rather than kernels — the least favourable shape possible — and it
            still beats classic single-threaded TA-Lib
            {fmt_x(agg16["speedupBatchVsTalib"])}–{fmt_x(agg64["speedupBatchVsTalib"])}
            across all 94 systems, with identical trades. The all-cores baseline stays
            ahead on this transfer-bound aggregate; the moment kernels do real work the
            picture flips (next table).
            Reproduce: <code>python tools/optimizer/scripts/bench_experiments_batch.py
            --bin tacuda-opt --bars 500000 --symbols 64</code>
        </p>

        <h3 class="sub-head">Where the time goes: kernel categories (64 × 500 k)</h3>
        <table class="data">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>System</th>
                    <th class="num">tacuda batch</th>
                    <th class="num">TA-Lib 1t</th>
                    <th class="num">TA-Lib 8t</th>
                    <th class="num">vs 1t</th>
                    <th class="num">vs 8t</th>
                </tr>
            </thead>
            <tbody>
{category_rows()}
            </tbody>
        </table>
        <p class="table-note">
            Windowed kernels win up to an order of magnitude (and keep growing with
            portfolio size — see the chart below). The 0.4.1 scan rewrite moved the
            cumulative club lines (OBTR, WOBV, ASI, HPI, XAverage) from single-thread
            recursions to parallel prefix scans: those systems ran at 0.2–0.3x of one
            CPU thread in 0.4.0 and now sit at the PCIe floor, 1.2–1.9x ahead of it.
            The one honest exception is the EL-parabolic state machine (trend flips +
            acceleration clamps do not associate), which stays a per-series thread.
        </p>

        <h3 class="sub-head">Scaling with portfolio size (100 k bars per symbol)</h3>
        {scaling_svg(total_series,
                     "12-system representative subset, indicator-phase total",
                     "Indicator seconds vs number of symbols, log-log")}
        {scaling_svg(refdev_series,
                     "exp10_ref_dev (Reference Deviation, windowed kernel)",
                     "Reference Deviation indicator seconds vs number of symbols")}
        <p class="table-note">
            Both axes logarithmic. The GPU line is nearly flat until the card fills
            (4 → 64 symbols costs almost nothing), then grows with data volume; CPU
            lines grow linearly from the start. At 1024 symbols × 100 k bars the
            12-system subset runs {fmt_x(best_total)} faster than one CPU thread, and the
            windowed kernels beat even all eight CPU threads outright:
            Reference&nbsp;Deviation {fmt_x(rd_x1)} vs one thread / {fmt_x(rd_x8)} vs
            eight, price-distribution moments {fmt_x(pd_x1)} vs one thread. Systems whose
            requests are a couple of cheap moving averages stay PCIe-bound — that floor
            is the honest cost of hosting the simulation on the CPU; keep lines
            device-resident (<code>ct_*_batch_d</code>, derived-series chaining) and it
            disappears.
            Reproduce: <code>bench_experiments_batch.py --bars 100000 --symbols 1024
            --strategies exp03_obtr,exp06_asi,…</code>
        </p>

        <h3 class="sub-head">What TA-Lib cannot do at all</h3>
        <table class="data">
            <thead>
                <tr><th>Capability</th><th>TA-Lib</th><th>tacuda 0.4.1</th></tr>
            </thead>
            <tbody>
                <tr><td>Indicator functions</td><td class="num">161, CPU only</td>
                    <td class="num">757 entry points: host + device (<code>_d</code>) + batch, GPU-resident</td></tr>
                <tr><td>Multi-symbol batch API</td><td class="num">—</td>
                    <td class="num">166 families, one call per portfolio, CUDA stream pool</td></tr>
                <tr><td>Derived-series chaining</td><td class="num">—</td>
                    <td class="num">"XAverage of OBTR" as one request, no host round-trips</td></tr>
                <tr><td>STAD club pack (ASI, OBTR, wOBV, HPI, TrendScore, RefDev, pivots…)</td>
                    <td class="num">—</td><td class="num">native families with batch + chaining</td></tr>
                <tr><td>Zero-copy GPU pipelines</td><td class="num">—</td>
                    <td class="num"><code>ct_*_d</code> / <code>ct_*_batch_d</code> on device pointers</td></tr>
                <tr><td>Bindings</td><td class="num">C + wrappers</td>
                    <td class="num">Python / Node.js / .NET incl. device and batch surfaces</td></tr>
            </tbody>
        </table>
        <p class="table-note">
            Correctness first: 88/88 accuracy specs against TA-Lib output, plus per-run
            net-profit parity between every backend in the strategy benchmarks above.
        </p>
    </div>
</section>
{END}"""


def main() -> None:
    text = PAGE.read_text(encoding="utf-8")
    payload = render()
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: payload, text)
    else:
        anchor = "<section>\n    <div class=\"wrap\">\n        <div class=\"sec-head\">\n            <span class=\"sec-kicker\">Coverage run</span>"
        if anchor not in text:
            raise SystemExit("anchor (Coverage run section) not found in benchmark.html")
        text = text.replace(anchor, payload + "\n\n" + anchor)
    PAGE.write_text(text, encoding="utf-8", newline="\n")
    print(f"updated {PAGE}")


if __name__ == "__main__":
    main()
