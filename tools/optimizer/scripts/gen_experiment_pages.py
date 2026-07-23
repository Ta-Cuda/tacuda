#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Static-site generator for the EXPERIMENTS strategy pack.

Reads:
  * experiments_meta.py           - authored titles, categories, descriptions
  * <src>/ExpVol*.cpp             - rule summaries from implementation comments
  * <results>/summary.json        - parameter tables + run status
  * <results>/<name>.bt.json      - default-parameter backtest (with trades)
  * <results>/<name>.ga.json      - genetic search summary
  * <results>/<name>.wfo.json     - walk-forward summary

Writes:
  * <site>/experiments.html                 - hub page
  * <site>/experiments/<name>.html          - one page per system
  * <site>/sitemap.xml                      - refreshed with all pages

Usage:
    python gen_experiment_pages.py --repo /path/to/tacuda
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments_meta import CATEGORIES, META, VOLUMES  # noqa: E402

VER = "0.3.0"
ASSET_V = "0.3.0-5"
BASE_URL = "https://tacuda.com"
GH = "https://github.com/Ta-Cuda/tacuda"
DATE = "2026-07-06"

BENCH_BARS = "5 000"
BENCH_NOTE = ("synthetic daily series (5 000 bars, seed 42), commission 2 bps + "
              "1 tick slippage per side, capital 10 000, GPU backend "
              "(RTX 3090 Ti)")
FITNESS = "netprofit − 2·maxdd"


# --------------------------------------------------------------- utilities

def esc(s: str) -> str:
    return html.escape(s, quote=True)


def fmt(v, nd: int = 2, dash: str = "—") -> str:
    """Format a possibly-missing number with thousands separators."""
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if f != f:  # NaN
        return dash
    s = f"{f:,.{nd}f}"
    return s


def fmt_pct(v, nd: int = 1) -> str:
    return dash_or(v, lambda f: f"{f * 100.0:,.{nd}f}%")


def dash_or(v, fn) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return fn(f)


def params_str(params: dict) -> str:
    def one(v):
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:g}"
    return " ".join(f"{k}={one(v)}" for k, v in params.items())


def sentence_clip(text: str, limit: int = 158) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "." in cut:
        return cut[:cut.rfind(".") + 1]
    return cut[:cut.rfind(" ")] + "…"


# --------------------------------------------------------------- data load

def load_rule_summaries(src_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(src_dir, "ExpVol*.cpp"))):
        lines = open(path, encoding="utf-8").read().splitlines()
        i = 0
        while i < len(lines):
            m = re.match(r"^// --- (exp\S+) -+", lines[i])
            if m:
                block = []
                i += 1
                while i < len(lines) and lines[i].startswith("//"):
                    block.append(lines[i][2:].strip())
                    i += 1
                out[m.group(1)] = " ".join(block)
            else:
                i += 1
    return out


def load_results(results_dir: str, name: str) -> dict:
    res = {}
    for kind in ("bt", "ga", "wfo"):
        path = os.path.join(results_dir, f"{name}.{kind}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                res[kind] = json.load(fh)
    return res


# --------------------------------------------------------------- svg chart

def equity_svg(trades: list[dict], initial: float) -> str:
    """Trade-level equity curve as an inline SVG (no external deps)."""
    if not trades:
        return ""
    pts = [(0, 0.0)]
    eq = 0.0
    for t in sorted(trades, key=lambda t: (t["exitBar"], t["entryBar"])):
        pnl = t.get("pnl")
        if pnl is None:
            continue
        eq += float(pnl)
        pts.append((int(t["exitBar"]), eq))
    if len(pts) < 3:
        return ""
    # Downsample, always keeping the last point.
    if len(pts) > 220:
        step = (len(pts) - 1) / 219.0
        pts = [pts[round(i * step)] for i in range(219)] + [pts[-1]]

    w, h, padl, padr, padt, padb = 720, 200, 56, 12, 14, 22
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    span_x = max(1, xmax - xmin)
    span_y = ymax - ymin

    def px(x):
        return padl + (x - xmin) / span_x * (w - padl - padr)

    def py(y):
        return padt + (ymax - y) / span_y * (h - padt - padb)

    line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
    fill = (f"{px(xs[0]):.1f},{py(0):.1f} " + line +
            f" {px(xs[-1]):.1f},{py(0):.1f}")
    zero_y = py(0.0)
    zero = ""
    if ymin <= 0.0 <= ymax:
        zero = (f'<line class="eq-zero" x1="{padl}" y1="{zero_y:.1f}" '
                f'x2="{w - padr}" y2="{zero_y:.1f}"/>')
    return f"""<svg class="eq-chart" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" aria-label="Trade-level equity curve">
  {zero}
  <polygon class="eq-fill" points="{fill}"/>
  <polyline class="eq-line" points="{line}"/>
  <text x="{padl}" y="{padt + 4}" dominant-baseline="hanging">{fmt(ymax, 0)}</text>
  <text x="{padl}" y="{h - padb - 4}">{fmt(ymin, 0)}</text>
  <text x="{padl}" y="{h - 6}">bar {xmin}</text>
  <text x="{w - padr}" y="{h - 6}" text-anchor="end">bar {xmax}</text>
</svg>"""


# --------------------------------------------------------------- templates

def page_shell(*, title: str, description: str, canonical: str, body: str,
               depth: int, og_type: str = "article",
               jsonld_headline: str = "") -> str:
    p = "../" * depth
    headline = jsonld_headline or title
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)}</title>
    <meta name="description" content="{esc(description)}">
    <meta name="robots" content="index,follow,max-image-preview:large">
    <meta name="author" content="Pavel Deecalov">
    <meta name="application-name" content="TaCuda">
    <link rel="canonical" href="{canonical}">

    <meta property="og:type" content="{og_type}">
    <meta property="og:site_name" content="TaCuda">
    <meta property="og:locale" content="en_US">
    <meta property="og:title" content="{esc(title)}">
    <meta property="og:description" content="{esc(description)}">
    <meta property="og:url" content="{canonical}">
    <meta property="og:image" content="{BASE_URL}/og-image.png">
    <meta property="og:image:width" content="1400">
    <meta property="og:image:height" content="900">
    <meta property="og:image:alt" content="TaCuda GPU technical analysis icon pack preview">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{esc(title)}">
    <meta name="twitter:description" content="{esc(description)}">
    <meta name="twitter:image" content="{BASE_URL}/og-image.png">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{p}styles.css?v={ASSET_V}">

    <link rel="icon" href="{p}favicon.ico" sizes="any">
    <link rel="icon" href="{p}favicon.svg" type="image/svg+xml">
    <link rel="icon" type="image/png" sizes="48x48" href="{p}favicon-48x48.png">
    <link rel="icon" type="image/png" sizes="32x32" href="{p}favicon-32x32.png">
    <link rel="icon" type="image/png" sizes="16x16" href="{p}favicon-16x16.png">
    <link rel="apple-touch-icon" sizes="180x180" href="{p}apple-touch-icon.png">
    <link rel="mask-icon" href="{p}icons/safari-pinned-tab.svg" color="#00e5ff">
    <link rel="manifest" href="{p}site.webmanifest">
    <meta name="theme-color" content="#07111f">
    <meta name="msapplication-TileColor" content="#07111f">
    <meta name="msapplication-config" content="{p}browserconfig.xml">
</head>
<body>
<a class="skip-link" href="#content">Skip to content</a>

<nav class="nav" aria-label="Primary">
    <div class="wrap nav-inner">
        <a class="nav-logo" href="{p}index.html">
            <img class="logo-mark" src="{p}favicon.svg" width="22" height="22" alt="" aria-hidden="true">
            TACUDA <span class="ver">v{VER}</span>
        </a>
        <div class="nav-links" id="navLinks">
            <a href="{p}index.html#api">API</a>
            <a href="{p}index.html#functions">Functions</a>
            <a href="{p}benchmark.html">Benchmark report</a>
            <a href="{p}optimizer.html">Optimizer</a>
            <a href="{p}experiments.html">Experiments</a>
            <a class="ext" href="{GH}" target="_blank" rel="noopener">GitHub ↗</a>
        </div>
        <button class="nav-burger" id="navBurger" aria-label="Menu">≡</button>
        <button class="theme-btn" id="themeBtn" aria-label="Toggle theme">
            <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4l1.4-1.4"/></svg>
            <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>
    </div>
</nav>

<main id="content">
{body}
</main>

<footer>
    <div class="wrap foot-inner">
        <span>© 2024–2026 TaCuda · Apache-2.0</span>
        <div class="foot-links">
            <a href="{p}index.html">Home</a>
            <a href="{p}optimizer.html">Optimizer</a>
            <a href="{p}experiments.html">Experiments</a>
            <a href="{GH}" target="_blank" rel="noopener">GitHub</a>
        </div>
    </div>
</footer>

<script src="{p}script.js?v={ASSET_V}"></script>
<script type="application/ld+json">
{{
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "mainEntityOfPage": "{canonical}",
    "headline": "{esc(headline)}",
    "description": "{esc(description)}",
    "image": "{BASE_URL}/og-image.png",
    "author": {{ "@type": "Person", "name": "Pavel Deecalov" }},
    "dateModified": "{DATE}",
    "about": {{
        "@type": "SoftwareSourceCode",
        "name": "TaCuda",
        "codeRepository": "{GH}",
        "programmingLanguage": ["CUDA", "C++"]
    }}
}}
</script>
</body>
</html>
"""


def metrics_cells(m: dict) -> str:
    return (f'<td class="num">{fmt(m.get("netProfit"))}</td>'
            f'<td class="num">{fmt_pct(m.get("maxDrawdownPct"))}</td>'
            f'<td class="num">{fmt(m.get("profitFactor"))}</td>'
            f'<td class="num">{fmt(m.get("sharpe"))}</td>'
            f'<td class="num">{m.get("numTrades", "—")}</td>'
            f'<td class="num">{fmt_pct(m.get("winRate"))}</td>')


METRICS_HEAD = ('<th class="num">Net profit</th><th class="num">MaxDD%</th>'
                '<th class="num">PF</th><th class="num">Sharpe</th>'
                '<th class="num">Trades</th><th class="num">Win%</th>')


def strategy_page(name: str, meta: dict, params: list[dict], rules: str,
                  res: dict, prev_next: tuple) -> str:
    title = meta["title"]
    vol = meta["vol"]
    cat = CATEGORIES[meta["cat"]]
    bt = res.get("bt", {})
    ga = res.get("ga", {})
    wfo = res.get("wfo", {})
    btm = bt.get("metrics", {})
    src_file = f"ExpVol{vol:02d}.cpp"

    desc_meta = sentence_clip(
        f"{title} — {meta['desc'][0]}".replace("\n", " "))

    # --- header ------------------------------------------------------------
    wfe = wfo.get("wfe")
    head = f"""<header class="page-head">
    <div class="wrap">
        <span class="sec-kicker">EXPERIMENTS · Volume {vol} · {esc(cat)}</span>
        <h1>{esc(title)}</h1>
        <p>{esc(meta["desc"][0])}</p>

        <div class="env-grid">
            <div class="env-cell">
                <span class="env-k">Source</span>
                <span class="env-v">STAD Club, Vol. {vol}</span>
                <span class="env-d">original: {esc(meta["original"])}</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Category</span>
                <span class="env-v">{esc(cat)}</span>
                <span class="env-d">{len(params)} tunable parameter{"s" if len(params) != 1 else ""}</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Backtest (defaults)</span>
                <span class="env-v">{btm.get("numTrades", "—")} trades</span>
                <span class="env-d">PF {fmt(btm.get("profitFactor"))}, win {fmt_pct(btm.get("winRate"))}</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Walk-forward</span>
                <span class="env-v">WFE {fmt(wfe)}</span>
                <span class="env-d">6 windows, genetic per window</span>
            </div>
        </div>
    </div>
</header>"""

    # --- mechanics ----------------------------------------------------------
    mech = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Mechanics</span>
            <h2>How the system trades</h2>
            <p>{esc(meta["desc"][1])}</p>
        </div>
        <div class="rules-note">
            <strong>Implemented rules.</strong> {esc(rules)}
        </div>
        <p class="api-note">
            Registered as <code>{name}</code> in <a href="{GH}/blob/main/tools/optimizer/src/strategy/{src_file}" target="_blank" rel="noopener"><code>tools/optimizer/src/strategy/{src_file}</code></a>.
            Like every system in the pack it emits target positions at bar closes;
            the simulator fills them at the next bar open, and intrabar stop/limit
            orders are modeled with close-confirmation (see the
            <a href="../optimizer.html">optimizer page</a> for the execution model).
        </p>
    </div>
</section>"""

    # --- parameters ----------------------------------------------------------
    rows = "\n".join(
        f'            <tr><td><code>{esc(p["name"])}</code></td>'
        f'<td class="num">{p["default"]}</td>'
        f'<td class="num">{p["lo"]} .. {p["hi"]}</td>'
        f'<td class="num">{p["step"]}</td></tr>'
        for p in params)
    grid_flags = " ".join(
        f"--grid {p['name']}={p['lo']}..{p['hi']}:{p['step']}" for p in params)
    cmd = (f"# backtest with defaults\n"
           f"tacuda-opt backtest --synthetic 5000 --seed 42 --strategy {name} \\\n"
           f"    --commission-bps 2 --slippage-ticks 1\n\n"
           f"# genetic search over the full declared space\n"
           f"tacuda-opt genetic --synthetic 5000 --seed 42 --strategy {name} \\\n"
           f"    --commission-bps 2 --slippage-ticks 1 \\\n"
           f"    {grid_flags} \\\n"
           f"    --population 48 --generations 30 --fitness \"netprofit - 2*maxdd\"")
    par = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Parameters</span>
            <h2>Declared optimization space</h2>
            <p>Defaults follow the source volume; ranges bound both the grid and
               the genetic search. Invalid combinations (e.g. fast ≥ slow) are
               rejected by the strategy's <code>validate()</code> before simulation.</p>
        </div>
        <table class="data">
            <thead><tr><th>Parameter</th><th class="num">Default</th><th class="num">Range</th><th class="num">Step</th></tr></thead>
            <tbody>
{rows}
            </tbody>
        </table>

        <div class="term" style="margin-top: 22px">
            <div class="term-bar"><span class="term-title">reproduce</span><button class="term-copy">copy</button></div>
            <pre><code>{esc(cmd)}</code></pre>
        </div>
    </div>
</section>"""

    # --- results --------------------------------------------------------------
    chart = equity_svg(bt.get("trades", []), bt.get("initialCapital", 10000.0))
    chart_block = ""
    if chart:
        chart_block = (chart + '\n        <p class="table-note">Trade-level equity '
                       '(cumulative closed-trade PnL, default parameters). '
                       'Costs included: 2 bps + 1 tick per side.</p>')

    bt_table = f"""<table class="data">
            <thead><tr><th>Run</th><th>Params</th>{METRICS_HEAD}</tr></thead>
            <tbody>
            <tr><td>backtest, defaults</td><td><code>{esc(params_str(bt.get("params", {})))}</code></td>{metrics_cells(btm)}</tr>
            </tbody>
        </table>"""

    ga_rows = ""
    for row in ga.get("top", [])[:5]:
        ga_rows += (f'            <tr><td class="num">{row["rank"]}</td>'
                    f'<td><code>{esc(params_str(row.get("params", {})))}</code></td>'
                    f'<td class="num">{fmt(row.get("fitness"))}</td>'
                    f'{metrics_cells(row.get("metrics", {}))}</tr>\n')
    ga_block = ""
    if ga_rows:
        ga_block = f"""
        <h3 style="font-size: 15px; font-weight: 600; margin: 26px 0 10px">Genetic search — top 5 of {ga.get("evaluated", "?")} evaluated variants</h3>
        <table class="data">
            <thead><tr><th class="num">#</th><th>Params</th><th class="num">Fitness</th>{METRICS_HEAD}</tr></thead>
            <tbody>
{ga_rows}            </tbody>
        </table>
        <p class="table-note">Fitness = {FITNESS}; population 48, ≤30 generations,
           early stop after 10 stagnant generations. In-sample results —
           treat them as an upper bound, not an expectation.</p>"""

    wfo_rows = ""
    for wr in wfo.get("windowResults", []):
        if wr.get("optimized"):
            wfo_rows += (f'            <tr><td class="num">{wr["window"]}</td>'
                         f'<td class="num">{wr["isBegin"]}..{wr["isEnd"]}</td>'
                         f'<td class="num">{wr["isEnd"]}..{wr["oosEnd"]}</td>'
                         f'<td><code>{esc(params_str(wr.get("params", {})))}</code></td>'
                         f'<td class="num">{fmt(wr.get("isFitness"))}</td>'
                         f'<td class="num">{fmt(wr.get("oosFitness"))}</td>'
                         f'<td class="num">{fmt(wr.get("oosMetrics", {}).get("netProfit"))}</td>'
                         f'<td class="num">{wr.get("oosMetrics", {}).get("numTrades", "—")}</td></tr>\n')
        else:
            wfo_rows += (f'            <tr><td class="num">{wr["window"]}</td>'
                         f'<td class="num">{wr["isBegin"]}..{wr["isEnd"]}</td>'
                         f'<td class="num">{wr["isEnd"]}..{wr["oosEnd"]}</td>'
                         f'<td colspan="5">{esc(wr.get("note", "not optimized"))}</td></tr>\n')
    wfo_block = ""
    if wfo_rows:
        oos = wfo.get("oosOverall", {})
        wfo_block = f"""
        <h3 style="font-size: 15px; font-weight: 600; margin: 26px 0 10px">Walk-forward — 6 rolling windows, 75% in-sample, genetic per window</h3>
        <table class="data">
            <thead><tr><th class="num">#</th><th class="num">IS bars</th><th class="num">OOS bars</th><th>Chosen params</th><th class="num">IS fit</th><th class="num">OOS fit</th><th class="num">OOS NP</th><th class="num">Trades</th></tr></thead>
            <tbody>
{wfo_rows}            </tbody>
        </table>
        <p class="table-note">Stitched out-of-sample equity: net profit {fmt(oos.get("netProfit"))},
           PF {fmt(oos.get("profitFactor"))}, max drawdown {fmt_pct(oos.get("maxDrawdownPct"))},
           {oos.get("numTrades", "—")} trades. Walk-forward efficiency (OOS/IS
           fitness ratio) {fmt(wfe)} — parameters are re-chosen per window from
           in-sample data only.</p>"""

    prev_link = (f'<a href="{prev_next[0][0]}.html">← {esc(prev_next[0][1])}</a>'
                 if prev_next[0] else '<span class="pager-empty"></span>')
    next_link = (f'<a href="{prev_next[1][0]}.html">{esc(prev_next[1][1])} →</a>'
                 if prev_next[1] else '<span class="pager-empty"></span>')

    results = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Results</span>
            <h2>Three passes on the shared benchmark series</h2>
            <p>Every EXPERIMENTS system runs the same protocol: one backtest at
               the volume defaults, a genetic search over the full parameter
               space, and a 6-window walk-forward — all on the same {BENCH_NOTE}.</p>
        </div>
        {bt_table}
        {chart_block}{ga_block}{wfo_block}

        <p class="api-note" style="margin-top: 26px">
            These runs are an engine demonstration on synthetic data, not a
            trading recommendation. The synthetic series has its own drift and
            volatility regimes; absolute numbers will differ on real markets —
            rerun the commands above on your own CSV data
            (<code>--data yourfile.csv</code>) to evaluate the system honestly.
        </p>

        <div class="pager">
            {prev_link}
            <a href="../experiments.html">All experiments</a>
            {next_link}
        </div>
    </div>
</section>"""

    body = "\n".join([head, mech, par, results])
    return page_shell(
        title=f"{title} — EXPERIMENTS · TaCuda Strategy Optimizer",
        description=desc_meta,
        canonical=f"{BASE_URL}/experiments/{name}.html",
        body=body,
        depth=1,
        jsonld_headline=f"{title} — a STAD Club system in the TaCuda optimizer",
    )


# --------------------------------------------------------------- hub page

def hub_page(order: list[str], all_res: dict, summary: dict) -> str:
    n_sys = len(order)
    n_skipped = sum(len(v["skipped"]) for v in VOLUMES.values())

    # Aggregates for the highlight facts.
    best_pf = best_wfe = most_trades = best_oos = None
    for name in order:
        res = all_res[name]
        btm = res.get("bt", {}).get("metrics", {})
        pf, tr = btm.get("profitFactor"), btm.get("numTrades", 0)
        if pf is not None and tr and tr >= 30:
            if best_pf is None or pf > best_pf[1]:
                best_pf = (name, pf)
        if tr and (most_trades is None or tr > most_trades[1]):
            most_trades = (name, tr)
        wfe = res.get("wfo", {}).get("wfe")
        oos_tr = res.get("wfo", {}).get("oosOverall", {}).get("numTrades", 0)
        if wfe is not None and oos_tr >= 20:
            if best_wfe is None or wfe > best_wfe[1]:
                best_wfe = (name, wfe)
        oos_np = res.get("wfo", {}).get("oosOverall", {}).get("netProfit")
        if oos_np is not None and oos_tr >= 20:
            if best_oos is None or oos_np > best_oos[1]:
                best_oos = (name, oos_np)

    def fact(num, text):
        return (f'<div class="fact"><span class="fact-num">{num}</span>'
                f'<span class="fact-text">{text}</span></div>')

    facts = "\n".join([
        fact(f"{n_sys}", "systems implemented and benchmarked, volumes 1–13"),
        fact(esc(META[best_wfe[0]]["title"]) if best_wfe else "—",
             f"best walk-forward efficiency ({fmt(best_wfe[1])}) with ≥20 OOS trades"
             if best_wfe else "no WFE data"),
        fact(esc(META[best_pf[0]]["title"]) if best_pf else "—",
             f"best default-parameter profit factor ({fmt(best_pf[1])}) with ≥30 trades"
             if best_pf else "no PF data"),
        fact(f"{most_trades[1]:,}" if most_trades else "—",
             f"trades from the most active system ({esc(META[most_trades[0]]['title'])})"
             if most_trades else "no trade data"),
    ])

    head = f"""<header class="page-head">
    <div class="wrap">
        <span class="sec-kicker">EXPERIMENTS</span>
        <h1>{n_sys} classic trading systems, rebuilt on the optimizer</h1>
        <p>The EXPERIMENTS pack ports the trading systems of the Omega Research
           <em>System Trading &amp; Development Club</em> volumes 1–13 (1997–2000)
           from TradeStation EasyLanguage to <code>tacuda-opt</code> strategies.
           Every system runs the full pipeline — backtest, genetic optimization
           and walk-forward analysis — with indicators computed by the TaCuda
           GPU library. {n_sys} systems made the cut; the rest of the club
           material (intraday-only, multi-symbol and purely conceptual
           chapters) is documented as skipped, with the reason, volume by
           volume below.</p>

        <div class="env-grid">
            <div class="env-cell">
                <span class="env-k">Systems</span>
                <span class="env-v">{n_sys} implemented</span>
                <span class="env-d">13 source volumes, {n_skipped} skips — every one justified</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Benchmark</span>
                <span class="env-v">3 passes each</span>
                <span class="env-d">backtest · genetic · walk-forward</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Data</span>
                <span class="env-v">{BENCH_BARS} daily bars</span>
                <span class="env-d">synthetic, seed 42, weekends skipped</span>
            </div>
            <div class="env-cell">
                <span class="env-k">Costs</span>
                <span class="env-v">2 bps + 1 tick</span>
                <span class="env-d">per order side, capital 10 000</span>
            </div>
        </div>
    </div>
</header>"""

    method = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Method</span>
            <h2>One execution model, one protocol, no exceptions</h2>
            <p>Porting 1990s EasyLanguage to a modern next-bar-open simulator is
               mostly a discipline problem. These are the pack-wide conventions —
               applied uniformly so systems stay comparable.</p>
        </div>

        <div class="api-rows">
            <div class="api-row">
                <span class="api-name">execution</span>
                <p class="api-desc"><strong>Signals at close, fills at the next open.</strong>
                   The original systems lean on intrabar stop and limit orders; the
                   adaptation confirms the level against the finished bar and fills at
                   the next open (close-confirmation). No look-ahead — a dedicated
                   causality test shifts signals by one bar and requires results to change.</p>
            </div>
            <div class="api-row">
                <span class="api-name">position</span>
                <p class="api-desc"><strong>One unit, both directions.</strong> Pyramiding
                   entries (Stampede's nine units, Luxor's three) collapse to a single
                   unit per side; double-size reversal entries become plain reversals.
                   Dollar-denominated stops are re-expressed as ATR multiples so they
                   survive any price scale.</p>
            </div>
            <div class="api-row">
                <span class="api-name">indicators</span>
                <p class="api-desc"><strong>GPU where it counts.</strong> SMA/EMA/RSI/MACD/
                   ADX/Bollinger/stochastics and 17 further families come from TaCuda
                   device kernels through the optimizer's deduplicating cache; club
                   specials (Accumulation Swing Index, parabolic variants, OBTR,
                   kurtosis) are implemented CPU-side in the strategy pack.</p>
            </div>
            <div class="api-row">
                <span class="api-name">benchmark</span>
                <p class="api-desc"><strong>Three passes per system.</strong> Backtest at
                   volume defaults with the trade list; genetic search (population 48,
                   ≤30 generations, fitness <code>{FITNESS}</code>) over the declared
                   space; walk-forward with 6 rolling windows, 75% in-sample and a
                   genetic optimizer per window. All on the same {BENCH_NOTE}.</p>
            </div>
        </div>

        <p class="api-note">
            Every implementation lives in
            <a href="{GH}/tree/main/tools/optimizer/src/strategy" target="_blank" rel="noopener"><code>tools/optimizer/src/strategy/</code></a>
            and is covered by registry, smoke, warm-up and causality tests
            (127 optimizer tests total). The results below are engine
            demonstrations on synthetic data — not trading advice, and not
            expectations for real markets.
        </p>
    </div>
</section>"""

    highlights = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">At a glance</span>
            <h2>What stood out on the shared series</h2>
            <p>Same data, same costs, same protocol for all {n_sys} systems —
               which makes the outliers meaningful (within the limits of one
               synthetic series).</p>
        </div>
        <div class="bench-grid">
            <div>
                <table class="data">
                    <thead><tr><th>Pass</th><th>What it shows</th></tr></thead>
                    <tbody>
                        <tr><td>backtest</td><td>the system at the volume's published defaults — the honest starting point</td></tr>
                        <tr><td>genetic</td><td>the best the declared parameter space can do in-sample — an upper bound</td></tr>
                        <tr><td>walk-forward</td><td>how much of the optimized edge survives out-of-sample re-optimization — the number to respect</td></tr>
                    </tbody>
                </table>
                <p class="table-note">Systems with WFE near or above 1.0 kept their
                   in-sample character out-of-sample on this series; large negative
                   OOS numbers usually mean the genetic pass overfit a thin edge.</p>
            </div>
            <div class="bench-side">
{facts}
            </div>
        </div>
    </div>
</section>"""

    vol_sections = []
    for vol in sorted(VOLUMES):
        vmeta = VOLUMES[vol]
        names = [n for n in order if META[n]["vol"] == vol]
        if not names:
            continue
        rows = ""
        for n in names:
            m = META[n]
            res = all_res[n]
            btm = res.get("bt", {}).get("metrics", {})
            ga_best = res.get("ga", {}).get("best", {})
            wfo = res.get("wfo", {})
            rows += (
                f'            <tr>'
                f'<td><a href="experiments/{n}.html">{esc(m["title"])}</a></td>'
                f'<td>{esc(CATEGORIES[m["cat"]])}</td>'
                f'<td class="num">{fmt(btm.get("profitFactor"))}</td>'
                f'<td class="num">{btm.get("numTrades", "—")}</td>'
                f'<td class="num">{fmt(ga_best.get("fitness"))}</td>'
                f'<td class="num">{fmt(wfo.get("wfe"))}</td>'
                f'<td class="num">{fmt(wfo.get("oosOverall", {}).get("netProfit"))}</td>'
                f'</tr>\n')
        skipped = ""
        if vmeta["skipped"]:
            items = "; ".join(f"<strong>{esc(s)}</strong> — {esc(r)}"
                              for s, r in vmeta["skipped"])
            skipped = (f'\n        <p class="table-note">Skipped from this '
                       f'volume: {items}.</p>')
        vol_sections.append(f"""<section id="vol{vol}">
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">{esc(vmeta["title"])}</span>
            <h2>{len(names)} system{"s" if len(names) != 1 else ""} from Volume {vol}</h2>
            <p>{esc(vmeta["blurb"])}</p>
        </div>
        <table class="data" data-sortable>
            <thead><tr><th>System</th><th>Category</th><th class="num">BT PF</th><th class="num">BT trades</th><th class="num">GA fitness</th><th class="num">WFE</th><th class="num">OOS NP</th></tr></thead>
            <tbody>
{rows}            </tbody>
        </table>{skipped}
    </div>
</section>""")

    repro = f"""<section>
    <div class="wrap">
        <div class="sec-head">
            <span class="sec-kicker">Reproduce</span>
            <h2>Run the whole pack yourself</h2>
            <p>The benchmark driver ships with the repository; the full
               94-system, three-pass run takes under a minute on an RTX 3090 Ti.</p>
        </div>
        <div class="term">
            <div class="term-bar"><span class="term-title">bench_experiments.py</span><button class="term-copy">copy</button></div>
            <pre><code># build the optimizer, then:
python tools/optimizer/scripts/bench_experiments.py \\
    --bin ./build/tools/optimizer/tacuda-opt \\
    --out exp_results --bars 5000 --seed 42 --backend tacuda

# every strategy is a first-class citizen of the CLI:
./build/tools/optimizer/tacuda-opt list | grep exp
./build/tools/optimizer/tacuda-opt backtest --synthetic 5000 \\
    --strategy exp09_luxor --commission-bps 2 --slippage-ticks 1</code></pre>
        </div>
        <p class="api-note">
            Full adaptation notes per volume:
            <a href="{GH}/blob/main/docs/EXPERIMENTS.md" target="_blank" rel="noopener">docs/EXPERIMENTS.md</a>.
        </p>
    </div>
</section>"""

    body = "\n".join([head, method, highlights] + vol_sections + [repro])
    return page_shell(
        title="EXPERIMENTS — 94 classic trading systems on the TaCuda optimizer",
        description=(f"{n_sys} trading systems from the STAD Club volumes 1-13, "
                     "ported to tacuda-opt and benchmarked with backtest, genetic "
                     "optimization and walk-forward analysis on the GPU."),
        canonical=f"{BASE_URL}/experiments.html",
        body=body,
        depth=0,
        jsonld_headline="EXPERIMENTS: 94 classic trading systems on the TaCuda optimizer",
    )


def write_sitemap(site_dir: str, order: list[str]) -> None:
    urls = [
        ("", "weekly", "1.0"),
        ("benchmark.html", "monthly", "0.8"),
        ("optimizer.html", "monthly", "0.8"),
        ("experiments.html", "weekly", "0.8"),
    ] + [(f"experiments/{n}.html", "monthly", "0.6") for n in order]
    items = "\n".join(
        f"  <url>\n    <loc>{BASE_URL}/{path}</loc>\n"
        f"    <lastmod>{DATE}</lastmod>\n"
        f"    <changefreq>{freq}</changefreq>\n"
        f"    <priority>{prio}</priority>\n  </url>"
        for path, freq, prio in urls)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           f"{items}\n</urlset>\n")
    with open(os.path.join(site_dir, "sitemap.xml"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(xml)


# --------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", ".."))
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    src_dir = os.path.join(repo, "tools", "optimizer", "src", "strategy")
    results_dir = os.path.join(repo, "tools", "optimizer", "bench", "exp_results")
    site_dir = os.path.join(repo, "site")
    out_dir = os.path.join(site_dir, "experiments")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(results_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)

    rules = load_rule_summaries(src_dir)
    order = sorted(META.keys())

    missing = [n for n in order if n not in summary["strategies"]]
    if missing:
        raise SystemExit(f"no benchmark results for: {missing}")
    missing = [n for n in order if n not in rules]
    if missing:
        raise SystemExit(f"no rule comments for: {missing}")
    extra = [n for n in summary["strategies"] if n not in META]
    if extra:
        raise SystemExit(f"strategies without metadata: {extra}")

    all_res = {n: load_results(results_dir, n) for n in order}

    for i, name in enumerate(order):
        prev_n = order[i - 1] if i > 0 else None
        next_n = order[i + 1] if i + 1 < len(order) else None
        prev = (prev_n, META[prev_n]["title"]) if prev_n else None
        nxt = (next_n, META[next_n]["title"]) if next_n else None
        page = strategy_page(
            name, META[name],
            summary["strategies"][name]["params"],
            rules[name], all_res[name], (prev, nxt))
        with open(os.path.join(out_dir, f"{name}.html"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(page)

    hub = hub_page(order, all_res, summary)
    with open(os.path.join(site_dir, "experiments.html"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(hub)

    write_sitemap(site_dir, order)
    print(f"wrote {len(order)} strategy pages + experiments.html + sitemap.xml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
