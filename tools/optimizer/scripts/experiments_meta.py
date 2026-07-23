# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Metadata for the EXPERIMENTS strategy pack (STAD Club volumes 1-13).
#
# Consumed by gen_experiment_pages.py. Each strategy entry carries the display
# title, the original system name from the source volume, a category tag and
# two authored paragraphs: the trading concept, then the adaptation notes for
# the tacuda-opt next-bar-open execution model. The rule-level summary is
# extracted from the C++ implementation comments at generation time.

CATEGORIES = {
    "trend": "Trend following",
    "breakout": "Breakout",
    "pullback": "Trend pullback",
    "reversion": "Mean reversion",
    "pattern": "Price pattern",
    "oscillator": "Oscillator",
    "volume": "Volume flow",
    "volatility": "Volatility",
    "levels": "Support / resistance",
    "calendar": "Calendar",
}

VOLUMES = {
    1: {
        "title": "Volume 1",
        "blurb": "The opening volume: seven classic building blocks — moving-average "
                 "and MACD crossovers, channel and volatility breakouts, floor-trader "
                 "pivots and swing-based support/resistance.",
        "skipped": [
            ("Trend Line Automatic", "relies on chart drawing-object trendlines that have no OHLCV equivalent"),
            ("Spread", "trades the spread of two symbols; the optimizer feeds one series per strategy"),
            ("Pivot Points Intra-Day", "session-open/close logic requires intraday bars"),
        ],
    },
    2: {
        "title": "Volume 2",
        "blurb": "Eight systems around classic indicators pushed one step further: "
                 "ADX differentials, OBV extremes, volume divergence, stochastic "
                 "pins, CCI hooks and Bollinger squeezes.",
        "skipped": [
            ("Opening Gap", "needs intraday bars plus a daily Data2 series for the gap reference"),
        ],
    },
    3: {
        "title": "Volume 3",
        "blurb": "Nine systems leaning on statistics and candles: engulfing patterns, "
                 "Micro-M/W divergences, adaptive oscillator zones, standard-deviation "
                 "volatility and volatility clusters.",
        "skipped": [
            ("Inside Day Breakout Failure", "needs 30-minute bars plus a daily Data2 series"),
        ],
    },
    4: {
        "title": "Volume 4",
        "blurb": "Nine systems including the volume's flagship Momentum-Retracement: "
                 "DMI/ADX timing, Keltner breakouts, linear-regression momentum, "
                 "displaced averages with volume and a kurtosis-based distribution shift.",
        "skipped": [
            ("Spread System Beta Up/Down", "trades a two-symbol spread; one series per strategy here"),
        ],
    },
    5: {
        "title": "Volume 5",
        "blurb": "Seven systems with a retracement flavour: Fibonacci pullbacks, "
                 "oscillator hooks, pivot-low springs, a weekend currency effect and "
                 "displaced standard-deviation bands.",
        "skipped": [
            ("Currency/Bond", "intermarket system driven by a second symbol"),
            ("Intermarket I", "needs a second symbol as the signal series"),
            ("Intermarket II", "needs a second symbol as the signal series"),
        ],
    },
    6: {
        "title": "Volume 6",
        "blurb": "Nine systems, nothing skipped: the Accumulation Swing Index, "
                 "displaced-MA channels with Range Leader bars, Schwager's "
                 "reaction/resumption swing logic, parabolic-with-volume and "
                 "volume-weighted momentum.",
        "skipped": [],
    },
    7: {
        "title": "Volume 7",
        "blurb": "Four single-series systems survive this intermarket-heavy volume: "
                 "SMA derivatives, Art Putt's intra-bar oscillator, ratcheting MA "
                 "support/resistance and a trading-range breakout.",
        "skipped": [
            ("Dow Theory", "needs Industrials, Transports and Utilities as Data2-4"),
            ("Intraday Breakout / Fibonacci Retracement", "time-of-day rules on intraday bars"),
            ("Intermarket S&P/Bonds/Bills", "external Data2 series"),
            ("S&P Implied Volatility", "external implied-volatility series"),
        ],
    },
    8: {
        "title": "Volume 8",
        "blurb": "Four pattern-driven systems sharing a common ATR-stop toolbox: "
                 "Escalator's two-bar reversal, Jack-in-the-Box, Ryan's Hope pullback "
                 "and the Stampede channel breakout.",
        "skipped": [
            ("ActivityBar Price Distribution", "needs intrabar tick-distribution (ActivityBar) data"),
            ("SystemBuilder / 2000i forecasting / Fixed Ratio chapters", "no OHLCV trading rules to port"),
        ],
    },
    9: {
        "title": "Volume 9",
        "blurb": "Eight systems, from DiNapoli double penetration to Luxor and the "
                 "congestion-fading Traffic Jam — all managed by protective, breakeven "
                 "and trailing ATR stops.",
        "skipped": [
            ("International Index Composite", "needs seven world-index data series"),
            ("RadarScreen indicator chapter", "screening indicator, not a trading system"),
        ],
    },
    10: {
        "title": "Volume 10",
        "blurb": "Seven systems including two Joe Krutsinger day-of-week/breakout "
                 "designs, LeBeau's Serendipity trigger, Brower's Reference Deviation "
                 "and the MarketStrength-based Superman.",
        "skipped": [
            ("Gemini", "needs a weekly Data2 series next to the daily one"),
            ("RadarScreen / Serendipity-trigger concept chapters", "no standalone system rules"),
        ],
    },
    11: {
        "title": "Volume 11",
        "blurb": "Seven systems: stacked-average High Five, dip-buying Bull's Eye, "
                 "Night Train with Williams' Bailout exit, a swing-built symmetrical "
                 "triangle, Chande's TrendScore, Lindsay's Trident and an 'Oops' gap fade.",
        "skipped": [
            ("MISER", "needs a bonds Data2 series"),
            ("Two Favorite Exit Strategies / performance-report chapters", "exit and reporting concepts, no standalone system"),
        ],
    },
    12: {
        "title": "Volume 12",
        "blurb": "Nine systems: MACD pullbacks, Stowell zone trades, Wyckoff springs, "
                 "reverse divergence, linear-regression crossings and Walt Downs' "
                 "Tigers in the Mist #2.",
        "skipped": [
            ("Trendiness in the Futures Markets", "research note without trading rules"),
        ],
    },
    13: {
        "title": "Volume 13",
        "blurb": "The club's 'greatest hits' revisions: six systems re-tuned from "
                 "earlier volumes — open-close histogram, DMA with Range Leaders, "
                 "linear-regression momentum, Momentum-Retracement, VolEx and "
                 "volatility clusters.",
        "skipped": [
            ("Luxor (Vol. 13 revision)", "pyramiding variant of the Vol. 9 system already implemented as exp09_luxor"),
            ("Currency/Bonds/Dollar Index, A/D Divergence", "need secondary data series"),
            ("Triple Play (Vol. 13 revision)", "already implemented from Vol. 11"),
            ("Chapters 1-3", "educational material, no system rules"),
        ],
    },
}

META = {
    # ------------------------------------------------------------- volume 1
    "exp01_ma_crossover": {
        "title": "Moving Average Crossover",
        "original": "Moving Average Crossover System",
        "vol": 1, "cat": "trend",
        "desc": [
            "The volume's opener refuses to trade the raw SMA cross — by the time "
            "a fast average overtakes a slow one, the easy part of the move is "
            "often gone. Instead the crossover only arms a breakout order well "
            "beyond the recent extreme, so the market must keep pushing in the "
            "crossover's direction before any position exists.",
            "The adaptation keeps the three-stage structure of the original "
            "EasyLanguage code: an armed entry stop that expires after a fixed "
            "number of bars, a trailing exit at the n-bar extreme, and a re-entry "
            "stop after a stop-out. Intrabar stop fills become close-confirmed "
            "stops filled at the next open, the pack-wide convention.",
        ],
    },
    "exp01_macd_breakout": {
        "title": "MACD Breakout",
        "original": "MACD Breakout System",
        "vol": 1, "cat": "breakout",
        "desc": [
            "A congestion detector built from the MACD histogram: each zero "
            "crossing of the histogram marks a small shift in control between "
            "bulls and bears, and several crossings packed into a short window "
            "mean the market is coiling. The system frames that coil with the "
            "highs and lows printed at the crossings and trades the escape.",
            "Breakout stops sit half an average true range beyond the crossing "
            "channel while the crossings stay young; the trailing exit is the "
            "classic n-bar extreme. The channel is rebuilt bar by bar from a "
            "rolling deque of crossing records, exactly mirroring the original's "
            "arrays.",
        ],
    },
    "exp01_trend_follower": {
        "title": "Trend Follower",
        "original": "Trend Follower System",
        "vol": 1, "cat": "trend",
        "desc": [
            "A ratcheting volatility-band follower in the spirit of a manually "
            "stepped SuperTrend: two bands sit a multiple of average true range "
            "around price, and only a smoothed close (EMA) escaping a band in "
            "the direction of the 4/9/18 SMA stack moves the system into a "
            "position — at which point the bands shift one step in that "
            "direction and the old band becomes the stop.",
            "The band state machine (initialize at the first valid bar, ratchet "
            "on entry, protective stop at the opposite band) is preserved; the "
            "stack filter uses the same fixed 4/9/18 lengths as the source code.",
        ],
    },
    "exp01_sr_levels": {
        "title": "Support & Resistance Levels",
        "original": "Support & Resistance System",
        "vol": 1, "cat": "levels",
        "desc": [
            "Support and resistance are computed the way a chartist would: "
            "average the last few confirmed swing highs into resistance and the "
            "last few swing lows into support, then only trade when the two "
            "levels frame a wide, well-shaped range. Longs come from smoothed "
            "lows reclaiming support, shorts from smoothed highs failing at "
            "resistance.",
            "Swing pivots confirm `strength` bars after the fact — the "
            "implementation only reads them once confirmable with data up to the "
            "current bar, avoiding the look-ahead that naive pivot code hides. "
            "The trailing exit only activates once the trade shows a profit, as "
            "in the original.",
        ],
    },
    "exp01_pivot_points": {
        "title": "Pivot Points",
        "original": "Pivot Points System",
        "vol": 1, "cat": "breakout",
        "desc": [
            "Floor-trader arithmetic on daily bars: pivot = (H+L+C)/3, with the "
            "R2/S2 levels a full range away. A push through R2 while the 4/9/18 "
            "SMA stack is bullish joins the move; a collapse through S2 in a "
            "bearish stack sells it. R2/S2 are far enough from the pivot that "
            "reaching them takes genuine directional pressure.",
            "Levels are computed from the previous bar and armed as stops for "
            "the current bar, matching the original's next-bar orders. Exits are "
            "the volume's signature pair: take the first profitable open, "
            "otherwise trail the n-bar extreme.",
        ],
    },
    "exp01_channel_breakout": {
        "title": "Channel Breakout",
        "original": "Channel Breakout System",
        "vol": 1, "cat": "breakout",
        "desc": [
            "A Donchian breakout with a patience requirement: one poke through "
            "the n-bar channel is noise, so the system counts consecutive "
            "same-side breakouts and only enters on the k-th, provided no "
            "opposite breakout interrupted the sequence. The count resets to one "
            "after entry, keeping the position responsive to follow-through.",
            "Entries fire at the close of the qualifying breakout bar (filled "
            "next open); the exit is the standard n-bar extreme trail. This is "
            "the shortest system in the pack and a useful baseline for the "
            "volume.",
        ],
    },
    "exp01_volex": {
        "title": "VolEx (Volatility Expansion)",
        "original": "VolEx System",
        "vol": 1, "cat": "volatility",
        "desc": [
            "The original volatility-expansion day-trade: from today's open, "
            "place a buy stop a multiple of the 4-bar average range above and a "
            "sell stop a (wider) multiple below. A fill means the day's range "
            "has already expanded beyond its recent norm — the bet is that "
            "expansion begets continuation.",
            "The long and short multipliers are independent parameters, as in "
            "the source. Exits stack three rules: first profitable open "
            "(Williams' Bailout), the n-bar extreme trail, and a stale-position "
            "exit after four bars without progress.",
        ],
    },

    # ------------------------------------------------------------- volume 2
    "exp02_adx_diff": {
        "title": "ADX Differential",
        "original": "ADX Differential System",
        "vol": 2, "cat": "breakout",
        "desc": [
            "ADX measures trend strength, so its bar-to-bar differential "
            "measures trend acceleration. A weighted average of that "
            "differential dropping below zero says the current trend is dying — "
            "which is precisely when this system starts hunting a breakout in "
            "either direction out of the stalling market.",
            "Breakout stops go half an average range beyond the trigger close "
            "and stay armed for a fixed window. The exit begins as a 6-bar "
            "channel stop on the entry bar and hands over to the STAD "
            "converging trail — the volume's shared exit that tightens as the "
            "trade ages.",
        ],
    },
    "exp02_head_shoulders": {
        "title": "Head & Shoulders",
        "original": "Head & Shoulders System",
        "vol": 2, "cat": "pattern",
        "desc": [
            "A fully mechanical head-and-shoulders: three confirmed swing highs "
            "where the middle towers over both shoulders, with the neckline "
            "drawn through the two swing lows between them. A break of the "
            "neckline shorts the completed top; the mirrored pattern in swing "
            "lows buys completed bottoms.",
            "Swing detection uses strength-bar pivots confirmed with available "
            "data only, and patterns expire after `maxage` bars so stale "
            "geometry cannot trigger. The neckline break is a close-confirmed "
            "stop, then the STAD converging trail manages the trade.",
        ],
    },
    "exp02_obv_extremes": {
        "title": "OBV New Highs / New Lows",
        "original": "OBV New Highs/Lows System",
        "vol": 2, "cat": "volume",
        "desc": [
            "On Balance Volume leads price at turning points, the volume "
            "argues: a burst of fresh OBV window-highs shows accumulation worth "
            "joining, while a cluster of OBV window-lows marks the capitulation "
            "the original fades with a double-size contrarian buy. Both cases "
            "resolve to longs — this is a long-only system.",
            "The adaptation counts OBV extremes inside a rolling window "
            "(`occurred_at_least` over new highs/lows) and buys the next open; "
            "the double-size contrarian entry becomes a single unit under the "
            "one-position engine. Exits use the STAD converging trail.",
        ],
    },
    "exp02_ad_divergence": {
        "title": "Advance/Decline Divergence",
        "original": "Advance/Decline Divergence System",
        "vol": 2, "cat": "volume",
        "desc": [
            "The original watched an advance/decline line against price; on a "
            "single series the same role is played by a smoothed volume-flow "
            "line (signed volume by close direction, EMA-smoothed). A new "
            "price low that the flow line refuses to confirm is accumulation — "
            "buy it; a new price high without a flow high is distribution — "
            "sell it.",
            "Divergence is defined bar-exactly as in the source: price takes "
            "out its `lookback`-bar extreme while the flow line stays inside "
            "its own. The STAD converging trail closes the loop.",
        ],
    },
    "exp02_stochastics": {
        "title": "Stochastics",
        "original": "Stochastics System",
        "vol": 2, "cat": "oscillator",
        "desc": [
            "An anti-textbook reading of the stochastic: SlowK pinned beyond "
            "the overbought line for three straight bars is treated as strength "
            "to join, not weakness to fade. The twist is the reversal stop that "
            "converges toward price each bar — when it is hit, the system "
            "flips into a double-size fade of the very trend it was riding.",
            "Under the single-unit engine the double-size flip becomes a plain "
            "reversal, and the fade leg exits on a 5-bar channel break. The "
            "converging-stop arithmetic (step size in average ranges) follows "
            "the original constants.",
        ],
    },
    "exp02_cci_bollinger": {
        "title": "CCI Bollinger Bands",
        "original": "CCI Bollinger Bands System",
        "vol": 2, "cat": "oscillator",
        "desc": [
            "Price outside a Bollinger Band says stretched; a smoothed CCI "
            "hooking back from beyond ±100 says the stretch is snapping. The "
            "band poke sets a hunting mode, the CCI hook arms an entry stop at "
            "the recent 3-bar extreme — so the market must actually turn "
            "before the fade fills.",
            "The hunting mode and the 5-bar life of the armed stop mirror the "
            "source's counters; exits use the converging trail with tightened "
            "constants, the aggressive variant the chapter recommends.",
        ],
    },
    "exp02_pct_change_channel": {
        "title": "Percent Change Channel",
        "original": "Percent Change Channel System",
        "vol": 2, "cat": "reversion",
        "desc": [
            "Rate-of-change as an overbought/oversold gauge: a move beyond "
            "roughly ±2% over the lookback marks an extended market, and the "
            "system fades it — but only via a stop at the recent 3-bar extreme "
            "in the fading direction, so a still-accelerating move cannot fill "
            "the order.",
            "The armed fade expires after three bars, keeping the trade tied "
            "to the specific extension that created it. Exits are quick, as "
            "mean-reversion demands: first profitable open or a 5-bar channel "
            "break.",
        ],
    },
    "exp02_bull_beater": {
        "title": "Bull Beater",
        "original": "Bull Beater System",
        "vol": 2, "cat": "volatility",
        "desc": [
            "A Bollinger squeeze play: when the band width compresses under "
            "1.5 average ranges the market is winding up, and a fast RSI tilt "
            "picks the expected escape direction. Breakout stops at the 5-bar "
            "extremes stay armed for five bars while the squeeze resolves.",
            "The original filtered entries by an intraday session clock; on "
            "daily bars that filter is dropped (noted in the source header). "
            "Exits use the STAD converging trail shared across the volume.",
        ],
    },

    # ------------------------------------------------------------- volume 3
    "exp03_buy_dips": {
        "title": "Buy Dips / Sell Rallies",
        "original": "Buy Dips/Sell Rallies System",
        "vol": 3, "cat": "pullback",
        "desc": [
            "The classic two-clock design: a slow RSI above or below 50 names "
            "the trend, and a fast DMI spread dipping against it names the "
            "dip (or rally) worth buying (or selling). The entry order then "
            "latches onto the dip bar's extreme and waits for price to turn "
            "back with the trend.",
            "The latched stop persists until the opposite setup replaces it, "
            "as in the source — there is no expiry window here. Exits use the "
            "STAD trail; intrabar stops are close-confirmed and filled at the "
            "next open.",
        ],
    },
    "exp03_engulfing": {
        "title": "Bullish & Bearish Engulfing",
        "original": "Bullish & Bearish Engulfing System",
        "vol": 3, "cat": "pattern",
        "desc": [
            "Candlestick logic with a trend veto: a bullish engulfing body "
            "printed against a falling EMA (or bearish engulfing against a "
            "rising one) marks a potential exhaustion of the countertrend "
            "swing. The pattern bar's extreme becomes a three-bar entry stop — "
            "the candle alone never trades.",
            "Body comparisons use the strict engulfing definition from the "
            "chapter (current body swallows the prior body, opposite colors). "
            "The exit trails the best close since entry, offset by half the "
            "4-bar average range.",
        ],
    },
    "exp03_oc_histogram": {
        "title": "Open-Close Histogram",
        "original": "Open-Close Histogram System",
        "vol": 3, "cat": "trend",
        "desc": [
            "Subtract an EMA of opens from an EMA of closes and the sign of "
            "the result tracks who wins the average session — buyers or "
            "sellers. A zero-line cross of this histogram is the volume's "
            "trend-change event, armed as a breakout half an ATR beyond the "
            "crossing bar.",
            "The armed stop lives for ten bars but only while the histogram "
            "keeps its post-cross sign; the opposite cross exits immediately "
            "at the close with a protective stop from the pre-cross bar as "
            "backstop. Volume 13 re-tunes this same engine (exp13_oc_histogram).",
        ],
    },
    "exp03_bollinger": {
        "title": "Bollinger Bands",
        "original": "Bollinger Bands System",
        "vol": 3, "cat": "reversion",
        "desc": [
            "Straight Bollinger mean reversion: two consecutive closes beyond "
            "a band show a stretched market, and the system enters against the "
            "move at the next open, aiming back toward the middle of the "
            "distribution.",
            "The original scaled in with limit orders at deeper band "
            "penetrations; the single-position engine drops the scale-ins and "
            "trades one unit (noted in the code). Exits use the STAD trail.",
        ],
    },
    "exp03_obtr": {
        "title": "On Balance True Range",
        "original": "On Balance True Range System",
        "vol": 3, "cat": "reversion",
        "desc": [
            "The Bollinger dip-buy from the same volume, gated by a volatility "
            "regime filter: On Balance True Range (true range signed by close "
            "direction, accumulated like OBV) must sit under its own EMA — "
            "quiet, orderly selling — before the dip is worth buying. "
            "Long-only by design.",
            "The OBTR line is computed on the CPU exactly as the chapter "
            "defines it; the entry and the STAD-trail exit are shared with the "
            "plain Bollinger system, making the pair a clean A/B test of the "
            "filter.",
        ],
    },
    "exp03_micro_mw": {
        "title": "Micro-M & Micro-W",
        "original": "Micro-M & Micro-W System",
        "vol": 3, "cat": "pattern",
        "desc": [
            "Tiny double-tops and double-bottoms with a divergence engine: a "
            "price/RSI swing divergence (price makes the extreme, the "
            "oscillator refuses) followed within a few bars by a two-bar close "
            "hook forms the Micro-M top or Micro-W bottom.",
            "The hook arms a 4-bar-extreme entry stop rather than trading "
            "immediately, filtering out divergences that never turn. Swings "
            "confirm strength bars late, look-ahead-free; exits use the STAD "
            "trail.",
        ],
    },
    "exp03_var_zones": {
        "title": "Variable Zones",
        "original": "Variable Zones System",
        "vol": 3, "cat": "oscillator",
        "desc": [
            "Fixed 30/70 oscillator bands assume every market distributes RSI "
            "the same way; this system rebuilds the bands from the "
            "oscillator's own confirmed swing history, averaging its recent "
            "swing highs into the top zone and swing lows into the bottom "
            "zone.",
            "Trades fire when the RSI crosses back out of an adaptive zone — "
            "up through the bottom band buys, down through the top band "
            "sells. Zone updates use only confirmed swings; the STAD trail "
            "manages exits.",
        ],
    },
    "exp03_stddev_vol": {
        "title": "Standard Deviation Volatility",
        "original": "Standard Deviation Volatility System",
        "vol": 3, "cat": "volatility",
        "desc": [
            "A always-in-the-hunt volatility system: every bar, entry stops "
            "sit a fraction of the Bollinger band width above and below the "
            "close. Whichever side fills first defines the position — the "
            "band width scales the trigger distance to current volatility "
            "automatically.",
            "Exits take the first profitable open; otherwise the position "
            "simply flips when the opposite stop fills. This is the most "
            "trade-frequent system in the volume and a good stress test of "
            "the cost model.",
        ],
    },
    "exp03_vol_clusters": {
        "title": "Volatility Clusters",
        "original": "Volatility Clusters System",
        "vol": 3, "cat": "volatility",
        "desc": [
            "Volatility arrives in clusters: a close-to-close jump or a bar "
            "range beyond mean + k standard deviations flags the start of "
            "one. The system brackets the flag bar with stops an average "
            "range away on both sides and lets the cluster pick the "
            "direction.",
            "Statistics are population standard deviations over the rolling "
            "window, as EasyLanguage's StdDev computes them. Exits at the "
            "first profitable open keep the holding period near one bar. "
            "Volume 13 revisits this engine with an ATR stop (exp13_vol_clusters).",
        ],
    },

    # ------------------------------------------------------------- volume 4
    "exp04_dmi_adx": {
        "title": "DMI-ADX",
        "original": "DMI-ADX System",
        "vol": 4, "cat": "trend",
        "desc": [
            "A three-way DMI conversation: ADX crossing above the opposite DI "
            "line while DI+ leads DI- means the trend is strong and directional "
            "agreement is complete. That setup bar's extreme becomes the entry "
            "stop, cancelled if ADX slips back under the line before the fill.",
            "Exits listen to the same indicator family: a protective stop at "
            "the setup bar's far extreme, a close-out when ADX peaks and turns "
            "(trend exhaustion), and another when the DI lines cross against "
            "the position.",
        ],
    },
    "exp04_keltner": {
        "title": "Keltner Channel",
        "original": "Keltner Channel System",
        "vol": 4, "cat": "breakout",
        "desc": [
            "Chester Keltner's channel — an EMA of typical price banded by "
            "average true range — read as a breakout instrument: a close "
            "outside the channel arms a stop half a channel-width beyond that "
            "bar's extreme, demanding follow-through past an already unusual "
            "close.",
            "The armed stop lives five bars. Exits either mean-revert (a close "
            "back through the midline leaves at that close) or trail the 4-bar "
            "extreme, whichever comes first.",
        ],
    },
    "exp04_linreg_mom": {
        "title": "Linear Regression & Momentum",
        "original": "Linear Regression & Momentum System",
        "vol": 4, "cat": "pullback",
        "desc": [
            "Two lenses on the same trend: a smoothed linear-regression "
            "endpoint gives the direction, and a momentum oscillator curling "
            "up from below zero times the entry into the dip. The entry stop "
            "sits halfway between the setup high and the regression line — "
            "inside the pullback, not chasing the high.",
            "An ATR protective stop guards the entry; the exit is a market "
            "order when the regression line turns and the 4-bar extreme "
            "breaks. Volume 13 re-tunes this design (exp13_linreg_momentum).",
        ],
    },
    "exp04_hpi_breakout": {
        "title": "Herrick Payoff & Channel Breakout",
        "original": "Herrick Payoff Index & Channel Breakout System",
        "vol": 4, "cat": "volume",
        "desc": [
            "A channel breakout that must be paid for: the close-confirmed "
            "n-bar breakout only arms when a Herrick-Payoff-style money-flow "
            "line agrees with the direction. True HPI needs open interest; on "
            "OHLCV data the adaptation uses mid-price change times volume, "
            "smoothed — the chapter's own fallback.",
            "Entry stops rest at the setup bar's extreme for five bars. The "
            "trade exits on the 4-bar extreme trail or immediately when the "
            "flow line flips sign against the position.",
        ],
    },
    "exp04_consec_close": {
        "title": "Consecutive Close",
        "original": "Consecutive Close System",
        "vol": 4, "cat": "pattern",
        "desc": [
            "Three up closes in a row prove demand; the first down close after "
            "them is treated as the pullback to join, not the reversal to "
            "fear. The next open plus a quarter-ATR becomes a breakout stop — "
            "the market must resume upward before the order fills.",
            "The mirrored short side uses three down closes and an up close. "
            "Exits trail the 10-bar extreme, with the original's scale-out "
            "converted to a single-unit two-strike exit once price clears the "
            "pattern extreme.",
        ],
    },
    "exp04_mom_retrace": {
        "title": "Momentum-Retracement",
        "original": "Momentum-Retracement System",
        "vol": 4, "cat": "pullback",
        "desc": [
            "The volume's flagship and its most elaborate machine: a strong "
            "trend must first exist (EMA channel direction, MACD above its "
            "signal, RSI and DMI-spread confirmation), then price must "
            "retrace in a controlled three-way pattern before a stop entry "
            "arms a third of the way back into the prior swing.",
            "Risk management is layered exactly as the chapter describes: an "
            "ATR protective stop, a swing/EMA trailing stop, and a "
            "momentum-death close-out when the indicators that built the "
            "setup die. Long and short sides are fully symmetric. Volume 13 "
            "ships a re-tuned MRTS revision (exp13_mom_retrace).",
        ],
    },
    "exp04_three_ema": {
        "title": "Three EMA Crossover",
        "original": "Three EMA Crossover System",
        "vol": 4, "cat": "trend",
        "desc": [
            "A triple-EMA alignment system: the fast average crossing the "
            "medium one signals, and the slow average vetoes any signal "
            "against the larger tide. Entries fill at the next open — no "
            "breakout confirmation here, the alignment itself is the edge.",
            "The exit mirrors the entry: losing the fast/medium alignment "
            "leaves the trade, with a quarter-step converging trail as the "
            "disaster backstop.",
        ],
    },
    "exp04_dma_volume": {
        "title": "Displaced MA & Volume",
        "original": "Displaced MA & Volume System",
        "vol": 4, "cat": "volume",
        "desc": [
            "A displaced moving average removes the average's lag-flutter by "
            "shifting it forward in time; a cross of price through it only "
            "counts here when volume runs above its own displaced average — "
            "participation must confirm the technical event.",
            "The confirmed cross arms a next-open breakout stop 20% of ATR "
            "beyond the open. Exits trail the 5-bar extreme and close at "
            "market on the reverse cross.",
        ],
    },
    "exp04_price_dist": {
        "title": "Price Distribution",
        "original": "Price Distribution System",
        "vol": 4, "cat": "volatility",
        "desc": [
            "The statistical odd one out: rolling kurtosis of closes measures "
            "whether the market trades in a peaked, choppy distribution or a "
            "flat, directional one. Kurtosis crossing under zero marks the "
            "regime change from chop to trend; the rolling skew picks the "
            "direction to trade it.",
            "Both moments are computed in double precision on the CPU over "
            "the rolling window. Exits use the STAD converging trail at full "
            "range — the system aims to hold through the whole directional "
            "phase.",
        ],
    },

    # ------------------------------------------------------------- volume 5
    "exp05_fib_retrace": {
        "title": "Fibonacci .382 Retracement",
        "original": "Fibonacci .382 Retracement System",
        "vol": 5, "cat": "pullback",
        "desc": [
            "The 38.2% retracement of the 50-bar range is the chapter's "
            "definition of a healthy pullback: deep enough to shake out weak "
            "hands, shallow enough to keep the trend intact. Reaching it "
            "opens a 24-bar window in which the system waits for the trend "
            "to prove it is resuming.",
            "Resumption is a hook (swing low) in the 15-bar EMA or a close "
            "back across it, entering at the next open. The original risked "
            "a fixed dollar amount per trade; the adaptation exposes that as "
            "a percent trailing stop from the best close instead.",
        ],
    },
    "exp05_hilo_osc": {
        "title": "High/Low Oscillator",
        "original": "High/Low Oscillator System",
        "vol": 5, "cat": "oscillator",
        "desc": [
            "An oscillator built from the channel extremes themselves: the "
            "n-bar high (or low) minus its own EMA measures how hard price "
            "presses against the edge of its recent range. A slow close EMA "
            "names the trend; the oscillator curling back toward it times "
            "the entry.",
            "Exits combine a percent trailing stop off the close with "
            "oscillator zero-cross timing exits, so a fading press against "
            "the extremes releases the trade before the trailing stop must.",
        ],
    },
    "exp05_stoch_obos": {
        "title": "Stochastic Overbought/Oversold",
        "original": "Stochastic Overbought/Oversold System",
        "vol": 5, "cat": "oscillator",
        "desc": [
            "A patient stochastic-hook system: SlowK crossing SlowD inside "
            "the oversold band is only a candidate — the entry is a stop at "
            "the two-bar extreme, armed for six bars, so the market must "
            "actually lift off the low before the long exists.",
            "The setup bar's far extreme is the protective stop from the "
            "moment of fill. Opposite hooks occurring outside the bands exit "
            "at the open, the chapter's asymmetric exit rule preserved as-is.",
        ],
    },
    "exp05_pivot_sr": {
        "title": "Pivot Point Support & Resistance",
        "original": "Pivot Point Support & Resistance System",
        "vol": 5, "cat": "levels",
        "desc": [
            "A spring off broken support: a pivot low that is also the "
            "channel low gets undercut — trapping breakout sellers — and a "
            "wide-range reversal bar reclaims it. The entry stop above that "
            "reversal bar catches the short squeeze, filtered by an EMA so "
            "only with-trend springs trade.",
            "Risk management is staged: initial stop under the reversal low, "
            "breakeven once the trade earns twice its risk, profit-taking at "
            "the 44-bar extreme. The 24-bar setup window matches the source.",
        ],
    },
    "exp05_currency_4day": {
        "title": "Currency Four-Day",
        "original": "Currency Four-Day System",
        "vol": 5, "cat": "calendar",
        "desc": [
            "A weekend-momentum effect in currencies: whatever way a 50-bar "
            "RSI leans on Friday's close, the system trades that way into "
            "Monday's open and is out by Tuesday. The edge, per the chapter, "
            "is that weekend position adjustment continues Friday's tilt "
            "through Monday.",
            "Day-of-week logic uses the bar timestamps directly (the "
            "synthetic benchmark series skips weekends like a real daily "
            "feed). Monday touching Friday's extreme takes profit at the "
            "same exit bar — collapsed into the one-bar hold under next-open "
            "execution.",
        ],
    },
    "exp05_chan_pullback": {
        "title": "Channel Breakout with Pullback",
        "original": "Channel Breakout with Pullback System",
        "vol": 5, "cat": "pullback",
        "desc": [
            "Breakouts fail because everyone sees them; this system lets the "
            "breakout happen, then buys the first 4-bar pullback extreme "
            "printed within ten bars of it. The failed-breakout crowd's "
            "stop-loss selling becomes the system's entry liquidity.",
            "Entry fills at the next open once the pullback extreme prints. "
            "The exit is a snap to the opposite 8-bar extreme — wide enough "
            "to survive the pullback's noise, tight enough to cap a real "
            "reversal.",
        ],
    },
    "exp05_disp_stddev": {
        "title": "Displaced Standard Deviation Bands",
        "original": "Displaced Standard Deviation Bands System",
        "vol": 5, "cat": "volatility",
        "desc": [
            "A pure stop-and-reverse machine: a 3-bar moving average "
            "displaced 16 bars forward, plus and minus two current standard "
            "deviations, forms a pair of bands that both always carry live "
            "stops. The displacement makes the bands a forecast of where "
            "price should be if nothing changes; two sigmas of violation "
            "flips the book.",
            "There are no separate exits — the opposite band is the exit and "
            "the next entry. Always in the market once warmed up.",
        ],
    },

    # ------------------------------------------------------------- volume 6
    "exp06_asi": {
        "title": "Accumulation Swing Index",
        "original": "Accumulation Swing Index System",
        "vol": 6, "cat": "breakout",
        "desc": [
            "Wilder's Accumulation Swing Index compresses each bar's "
            "open/high/low/close relationship into one number and "
            "accumulates it into a price-like line. The system compares ASI "
            "readings at successive price swing highs: a fresh swing high "
            "whose ASI beats the previous swing's reading is real strength, "
            "armed as a two-bar-high breakout for five bars.",
            "The ASI recursion (Wilder's K/R/M arithmetic) is implemented on "
            "the CPU to the letter. Initial stop at the two-bar extreme, "
            "then the trail steps to the most recent strength-1 swing.",
        ],
    },
    "exp06_mac_adx": {
        "title": "ADX & Moving-Average Channel",
        "original": "ADX & Moving-Average Channel System",
        "vol": 6, "cat": "trend",
        "desc": [
            "An EMA-of-highs / EMA-of-lows channel defines the trend's "
            "envelope; a close beyond it with ADX rising means the escape "
            "has strength behind it. The entry stop half a channel-height "
            "beyond that close is deliberately short-lived — two bars — "
            "because a real escape should not dawdle.",
            "The exit is symmetric and simple: a close back through the "
            "opposite channel line ends the trade. No trailing stops; the "
            "channel is the whole risk model.",
        ],
    },
    "exp06_dma_chan": {
        "title": "Displaced MA Channel & Range Leaders",
        "original": "Displaced MA Channel & Range Leaders System",
        "vol": 6, "cat": "trend",
        "desc": [
            "A Range Leader is a bar whose midpoint clears the previous "
            "bar's extreme on expanding range — a bar that leads, not "
            "follows. One of those closing outside a displaced-MA channel "
            "enters at that close, betting the leadership persists.",
            "The exit evolves with the trade: it trails the displaced "
            "median-price average early, then hugs the channel's own side "
            "once the trade matures. Volume 13's revision re-tunes this "
            "pairing (exp13_dma_range_leaders).",
        ],
    },
    "exp06_ma_pairs": {
        "title": "Four Sets of MA Crossover",
        "original": "Four Sets of Moving Average Crossover System",
        "vol": 6, "cat": "trend",
        "desc": [
            "Why should the average pair that gets you into longs also get "
            "you out of shorts? This system unbundles the roles: one SMA "
            "pair arms long entries at the bar high, a second pair exits "
            "longs on its own crossover, and two more pairs do the same for "
            "the short side — four independently tuned clocks.",
            "The eight lengths make a big, honest optimization surface — a "
            "good subject for the genetic search, and a caution about "
            "degrees of freedom.",
        ],
    },
    "exp06_cci_avg": {
        "title": "CCI Average",
        "original": "CCI Average System",
        "vol": 6, "cat": "oscillator",
        "desc": [
            "A smoothed CCI escaping its ±100 band flags an unusual "
            "commodity move; the system arms a 9-bar channel breakout stop "
            "for four bars in that direction, demanding price confirm what "
            "the oscillator claims.",
            "The exit is this chapter's invention: a channel trail whose "
            "lookback shrinks by one bar every time the trade prints a new "
            "extreme — the better the trade goes, the tighter it is held.",
        ],
    },
    "exp06_mrtr": {
        "title": "Minor Reaction / Trend Resumption",
        "original": "Minor Reaction/Trend Resumption System",
        "vol": 6, "cat": "pullback",
        "desc": [
            "Jack Schwager's swing framework mechanized: a channel breakout "
            "names the trend, four nested reaction bars (each holding inside "
            "the prior's extreme) define a orderly pullback, and two thrust "
            "bars breaking back with the trend re-arm the entry.",
            "The reaction extreme is the protective stop — if the pullback "
            "deepens past its own definition, the premise is dead. All swing "
            "counting is causal, bar-close information only.",
        ],
    },
    "exp06_first_pullback": {
        "title": "First Pullback",
        "original": "First Pullback System",
        "vol": 6, "cat": "pullback",
        "desc": [
            "Linda Raschke's dictum that the first pullback after a trend "
            "change is the safest trade, made mechanical: the MACD signal "
            "line's zero cross dates the trend change, and the first "
            "MACD/signal cross against the new trend is that first "
            "pullback, armed as a volatility-stop entry above the close.",
            "Exits stack the pullback extreme (premise failure), a "
            "volatility stop (noise failure) and the trend flipping back "
            "(regime failure). One trade per trend leg, as the name "
            "promises.",
        ],
    },
    "exp06_parabolic_vol": {
        "title": "Parabolic & Volume",
        "original": "Parabolic & Volume System",
        "vol": 6, "cat": "trend",
        "desc": [
            "Wilder's Parabolic SAR flips constantly in quiet markets; this "
            "system only honors flips that arrive on above-average volume — "
            "conviction flips. The entry stop sits a fraction of the "
            "old-to-new SAR distance beyond the flip close, armed four "
            "bars.",
            "Once filled, the SAR itself trails the trade, accelerating as "
            "the move extends. The SAR recursion runs on the CPU with the "
            "chapter's acceleration constants as parameters.",
        ],
    },
    "exp06_vwm": {
        "title": "Volume-Weighted Momentum",
        "original": "Volume-Weighted Momentum System",
        "vol": 6, "cat": "volume",
        "desc": [
            "Momentum times volume, EMA-smoothed: a move on volume counts, "
            "a move without it does not. The line crossing zero arms a "
            "quarter-ATR breakout above the crossing close for five bars, "
            "with an ATR stop under the setup bar.",
            "The opposite zero cross exits at the next open — the signal "
            "that built the trade is also the only signal trusted to end "
            "it.",
        ],
    },

    # ------------------------------------------------------------- volume 7
    "exp07_deriv_ma": {
        "title": "Derivative Moving Average",
        "original": "Derivative Moving Average System",
        "vol": 7, "cat": "trend",
        "desc": [
            "Calculus on a moving average: the first discrete derivative "
            "gives the average's slope, the second gives its acceleration. "
            "The second derivative crossing zero (deceleration ending) arms "
            "the setup; the first derivative confirming through zero in the "
            "same direction fires a stop entry one tick beyond the close.",
            "The system is stop-and-reverse — the mirrored setup is the "
            "exit. Derivatives are simple bar differences of the SMA, as "
            "the chapter defines them.",
        ],
    },
    "exp07_sr_osc": {
        "title": "Short-Term Support/Resistance",
        "original": "Short-Term Support/Resistance System (Art Putt)",
        "vol": 7, "cat": "oscillator",
        "desc": [
            "Art Putt's intra-bar strength oscillator: "
            "((TrueHigh−Open)+(Close−TrueLow)) / (2·TrueRange) measures how "
            "much of each bar's journey favored the bulls. Dipping under "
            "the buy level arms a long; re-crossing the level enters at the "
            "next open — buying the first sign of recovery, not the dip "
            "itself.",
            "Exits watch the oscillator's own geography: breaking the "
            "pre-entry extreme abandons the premise, and a 2-bar extreme "
            "stop activates once the oscillator reaches the far band.",
        ],
    },
    "exp07_ma_suppres": {
        "title": "Moving Average Support/Resistance",
        "original": "Moving Average Support/Resistance System",
        "vol": 7, "cat": "levels",
        "desc": [
            "Every close-through-SMA flip freezes the extreme price tracked "
            "during the previous regime, turning the average's history into "
            "a ladder of support/resistance levels. A close through the "
            "frozen level — proof the new regime has absorbed the old one's "
            "supply — enters with a stop at the bar extreme.",
            "Risk runs on two anchors: an entry-anchored protective ATR "
            "stop and a position-extreme ATR trail, whichever bites first.",
        ],
    },
    "exp07_tr_breakout": {
        "title": "Trading Range Breakout",
        "original": "Trading Range Breakout System",
        "vol": 7, "cat": "breakout",
        "desc": [
            "A congestion-quality gauge: bars that overlap heavily "
            "accumulate a 'no-trade' area, and only a dense enough range "
            "qualifies as a coil worth trading. A true-range expansion bar "
            "closing beyond that range is the ignition — entry on that "
            "close.",
            "The opposite range edge is the protective stop (the coil "
            "should not be re-entered), with an ATR trail off the position "
            "extreme managing the winner.",
        ],
    },

    # ------------------------------------------------------------- volume 8
    "exp08_escalator": {
        "title": "Escalator",
        "original": "Escalator System",
        "vol": 8, "cat": "pattern",
        "desc": [
            "A two-bar reversal in a filtered trend: a weak close (bottom of "
            "the range) followed by a strong close (top of the range) with "
            "both moving averages pointing up is the escalator step. Entry "
            "one tick beyond the two-bar extreme, on the very next bar only "
            "— this pattern does not age well.",
            "The exit is a fixed bracket, unusual for the pack: stop at the "
            "pattern's other extreme, limit target at twice the risk. "
            "Close-confirmation approximates both intrabar orders.",
        ],
    },
    "exp08_jack_in_box": {
        "title": "Jack-in-the-Box",
        "original": "Jack-in-the-Box System",
        "vol": 8, "cat": "pattern",
        "desc": [
            "The box is the downtrend's lowest bar; the jack pops when a "
            "later bar's low prints above that bar's high — the first "
            "geometric proof the decline has ended. An ATR-offset breakout "
            "entry stays armed for five bars (mirrored for tops).",
            "The exit is the volume's modified parabolic: seeded a multiple "
            "of ATR(3) beyond the entry bar, accelerating with new extremes "
            "— shared machinery with Vol. 10's Going In Style and Vol. 12's "
            "Springs.",
        ],
    },
    "exp08_ryans_hope": {
        "title": "Ryan's Hope",
        "original": "Ryan's Hope System",
        "vol": 8, "cat": "pullback",
        "desc": [
            "A pure pullback-in-trend: the SMA must be rising (trend), the "
            "close must undercut the close of Length2 bars ago (pullback), "
            "and the longer trend must remain intact. Entry at the next "
            "open — the discount is the edge, no breakout confirmation "
            "needed.",
            "Management is the volume's shared four-stop ATR suite: "
            "protective, breakeven, trailing and big-profit stops, each an "
            "ATR multiple parameter.",
        ],
    },
    "exp08_stampede": {
        "title": "Stampede",
        "original": "Stampede System",
        "vol": 8, "cat": "breakout",
        "desc": [
            "The volume's brute-force entry: a stop one tick beyond the "
            "n-bar extreme, always armed, both sides. Everything "
            "interesting lives in the exits — the same four-ATR-stop suite "
            "as Ryan's Hope, which turns a naive breakout into a managed "
            "campaign.",
            "The original pyramids up to nine units as the move extends; "
            "the single-position engine trades one unit per side, noted in "
            "the code and the docs.",
        ],
    },

    # ------------------------------------------------------------- volume 9
    "exp09_double_fun": {
        "title": "Double Your Fun",
        "original": "Double Your Fun System",
        "vol": 9, "cat": "pattern",
        "desc": [
            "DiNapoli-style double penetration of a displaced moving "
            "average: price crosses above the DMA, gets pulled back "
            "through it quickly, then crosses above again — the second "
            "penetration proves the first was no accident. Entry one tick "
            "above the second-cross bar, armed four bars.",
            "The exit trails the n-bar extreme, letting the "
            "double-penetration thesis run as far as the swing allows.",
        ],
    },
    "exp09_luxor": {
        "title": "Luxor",
        "original": "Luxor System",
        "vol": 9, "cat": "trend",
        "desc": [
            "The volume's flagship: while the fast triangular moving "
            "average rides above the slow one, a stop order rides one tick "
            "above every bar's high — each bar is a fresh chance to join "
            "the trend at proof-of-strength prices. The slow TMA itself is "
            "the trailing exit.",
            "The original pyramids up to three units on ADX-confirmed "
            "pullbacks; the adaptation trades one unit (Vol. 13 ships the "
            "pyramiding revision, skipped here as a duplicate engine). "
            "Triangular MAs are double-smoothed SMAs, computed exactly.",
        ],
    },
    "exp09_no_hurry": {
        "title": "No Hurry",
        "original": "No Hurry System",
        "vol": 9, "cat": "breakout",
        "desc": [
            "A channel breakout that waits n bars after the signal before "
            "entering — if the breakout was real, being late costs little; "
            "if it was noise, the delay filters it out entirely. Entries "
            "alternate long/short, never re-entering the same side twice "
            "in a row.",
            "Protective, breakeven and trailing ATR stops manage the "
            "position — the volume's common risk grammar.",
        ],
    },
    "exp09_obv_revisited": {
        "title": "OBV Revisited",
        "original": "OBV Revisited System",
        "vol": 9, "cat": "volume",
        "desc": [
            "OBV rebuilt with a graded brush: instead of all-or-nothing "
            "volume by close direction, each bar contributes volume scaled "
            "by where the close sits between open and range extremes — a "
            "weighted OBV. Crossing its SMA arms a one-tick breakout of "
            "the crossover bar.",
            "The opposite cross exits at the close; protective, breakeven "
            "and trailing ATR stops guard the trade meanwhile.",
        ],
    },
    "exp09_red_rover": {
        "title": "Red Rover",
        "original": "Red Rover System",
        "vol": 9, "cat": "levels",
        "desc": [
            "Support and resistance from the weighted close (WC = "
            "(H+L+2C)/4): 2·WC−Low projects resistance, 2·WC−High projects "
            "support — one-bar pivot arithmetic. Stop-and-reverse breakouts "
            "through the projected lines keep the system almost always in "
            "the market.",
            "An ATR profit target takes the quick win when offered; "
            "otherwise the opposite projected line is both protective stop "
            "and reversal entry.",
        ],
    },
    "exp09_skinny_dipper": {
        "title": "Skinny Dipper",
        "original": "Skinny Dipper System",
        "vol": 9, "cat": "pullback",
        "desc": [
            "A 'naked close' — one below the prior bar's low — against a "
            "rising double-EMA trend is the skinny dip: an overdone "
            "one-bar flush inside a healthy trend. The system arms a stop "
            "beyond the naked-close bar for exactly one bar; either the "
            "market snaps back immediately or there is no trade.",
            "Protective and breakeven ATR stops plus an ATR profit target "
            "manage the trade — mean-reversion entry, trend-following "
            "management.",
        ],
    },
    "exp09_swinger": {
        "title": "Swinger",
        "original": "Swinger System",
        "vol": 9, "cat": "oscillator",
        "desc": [
            "A price-oscillator system that buys weakness in uptrends: "
            "with the SMA trend filter up, a rising oscillator still below "
            "zero buys the next open — momentum recovering before it turns "
            "positive is the earliest with-trend entry the chapter could "
            "define.",
            "When momentum fades an n-bar extreme trailing stop activates; "
            "a protective ATR stop is live from entry. Mirrored short "
            "side.",
        ],
    },
    "exp09_traffic_jam": {
        "title": "Traffic Jam",
        "original": "Traffic Jam System",
        "vol": 9, "cat": "reversion",
        "desc": [
            "Congestion fading: ADX below its threshold and falling "
            "certifies a trendless market, where three consecutive down "
            "closes are noise to buy, not weakness to respect (up closes "
            "mirror short). Entry at that third close, filled next open.",
            "The exit is time-based — out at the close a fixed number of "
            "bars later — with a protective ATR stop for the times the jam "
            "breaks into a trend against the position.",
        ],
    },

    # ------------------------------------------------------------ volume 10
    "exp10_monday_trade": {
        "title": "JK Monday We Trade",
        "original": "JK Monday We Trade System",
        "vol": 10, "cat": "calendar",
        "desc": [
            "Joe Krutsinger's minimalist weekday system: on Friday's close, "
            "compare the close to its 2-bar SMA — above buys Monday's open, "
            "below sells it short. One bar later the trade is gone. The "
            "chapter's point is how much structure a system can shed and "
            "still test positive.",
            "Under next-open execution the dollar money-management stop and "
            "the exit-on-close collapse into a clean one-bar hold; "
            "day-of-week comes from the bar timestamps.",
        ],
    },
    "exp10_serendipity": {
        "title": "Serendipity",
        "original": "Serendipity System",
        "vol": 10, "cat": "oscillator",
        "desc": [
            "Chuck LeBeau's counterintuitive trigger: with RSI overbought, "
            "buy a stop a fraction of ATR above the SETUP bar's open — not "
            "the entry bar's. Because the reference is the older open, the "
            "setup day's own rally already counts toward the trigger, "
            "getting the fill earlier in fast markets. Overbought here "
            "means strength to join, LeBeau's signature reading.",
            "Bracket exits: protective ATR stop, breakeven shift, ATR "
            "profit target — the chapter's exact trio, each as an ATR "
            "multiple parameter.",
        ],
    },
    "exp10_eight_ball": {
        "title": "Eight Ball",
        "original": "Eight Ball System",
        "vol": 10, "cat": "pullback",
        "desc": [
            "A two-speed EMA pullback: the slow average must be rising "
            "(trend up) while the fast one dips (pullback in progress). "
            "The entry is an ATR-trigger breakout above the dip, so the "
            "pullback must already be resolving before the fill.",
            "Four ATR stops manage the trade — protective, trailing, "
            "volatility and a big-profit n-bar trail that only activates "
            "after an outsized gain, the volume's shared suite.",
        ],
    },
    "exp10_go_in_style": {
        "title": "Going In Style",
        "original": "Going In Style System",
        "vol": 10, "cat": "breakout",
        "desc": [
            "A channel breakout with a volatility offset: a fresh n-bar "
            "extreme arms a stop at the close plus trigger·ATR for the "
            "next bar only — the breakout must extend immediately or the "
            "order dies. This one-bar patience is the 'style'.",
            "A modified parabolic seeded an ATR(3) multiple off the entry "
            "bar trails the winner, the same machinery as "
            "Jack-in-the-Box.",
        ],
    },
    "exp10_store": {
        "title": "Who's Minding the Store",
        "original": "Who's Minding the Store System",
        "vol": 10, "cat": "pattern",
        "desc": [
            "A Key Reversal with a mood filter: a new n-bar low that "
            "closes up (sellers exhausted intraday) while the slow "
            "stochastic is oversold. The reversal alone is not tradeable — "
            "the system arms a 4-bar-high breakout for a few bars and "
            "demands the market follow through.",
            "The initial stop sits at the n-bar extreme that defined the "
            "reversal; trailing and big-profit ATR stops take over as the "
            "trade matures.",
        ],
    },
    "exp10_ref_dev": {
        "title": "Reference Deviation",
        "original": "Reference Deviation System",
        "vol": 10, "cat": "oscillator",
        "desc": [
            "Bill Brower's RDV oscillator: 100 · sum(C−SMA) / sum|C−SMA| "
            "over the same window — signed deviation divided by total "
            "deviation, a self-normalizing trend gauge bounded at ±100. "
            "Above the threshold is a trend worth holding long, below the "
            "negative threshold short.",
            "The zero cross exits, making the system nearly always in the "
            "market with a neutral gap around zero. RDV is computed on the "
            "CPU from an SMA request in a few lines.",
        ],
    },
    "exp10_superman": {
        "title": "Superman",
        "original": "Superman System",
        "vol": 10, "cat": "oscillator",
        "desc": [
            "MarketStrength: the net n-bar close change divided by the sum "
            "of same-signed moves — how much of the recent action pushed "
            "one way. Pinned at ±95 with older momentum decelerating, the "
            "market is at maximum one-sidedness; the system arms an n-bar "
            "extreme breakout in that direction.",
            "The trail follows the m-bar extreme with a risk-multiple "
            "profit target on top — strength entries, quick-profit "
            "management.",
        ],
    },

    # ------------------------------------------------------------ volume 11
    "exp11_high_five": {
        "title": "High Five",
        "original": "High Five System",
        "vol": 11, "cat": "trend",
        "desc": [
            "Five stacked SMAs, each double the length of the previous, "
            "all rising at once — five timeframes agreeing on the trend. "
            "The entry stop at the close plus a fraction of the bar's "
            "range only fills if the agreement continues into the next "
            "bar.",
            "The exit inverts the entry's logic: three or more averages "
            "turning against the position dissolves the consensus and the "
            "trade, with protective and breakeven ATR stops as backstops.",
        ],
    },
    "exp11_bullseye": {
        "title": "Hit the Bull's Eye",
        "original": "Hit the Bull's Eye System",
        "vol": 11, "cat": "pullback",
        "desc": [
            "A limit-order dip-buy in an uptrend: a fresh n-bar high "
            "closing above the high two bars back, with a rising EMA, "
            "projects a limit buy at the midpoint of that earlier bar — "
            "the bull's eye. The system pays a discount for strength "
            "instead of chasing it.",
            "Limit fills are modeled close-confirmed like the pack's stop "
            "fills. The profit target is a multiple of the setup-bar "
            "excursion, with a protective ATR stop underneath.",
        ],
    },
    "exp11_night_train": {
        "title": "JK Night Train",
        "original": "JK Night Train System",
        "vol": 11, "cat": "pattern",
        "desc": [
            "Krutsinger again, one rule per side: a close under the close "
            "two bars back arms a stop at the current bar's high — "
            "weakness that immediately reverses is strength. Exit at the "
            "first profitable open, Larry Williams' Bailout, plus a "
            "protective ATR stop.",
            "The system is deliberately skeletal; the chapter uses it to "
            "argue that exits, not entries, decide profitability. A good "
            "candidate for watching the ATR-stop parameter dominate the "
            "optimization.",
        ],
    },
    "exp11_sym_triangle": {
        "title": "Symmetrical Triangle",
        "original": "Symmetrical Triangle System",
        "vol": 11, "cat": "pattern",
        "desc": [
            "Four alternating strength-1 swings, each inside the last, "
            "squeeze price into a symmetrical triangle; a DMI comparison "
            "confirms which way the larger trend leans. Entry breaks "
            "through the triangle's resistance (or support) trendline — "
            "computed as the line through the two relevant pivots.",
            "The stop goes past the last pivot, shifts to breakeven at a "
            "risk multiple, and a measured-move target (triangle height "
            "projected from the break) takes profits.",
        ],
    },
    "exp11_trendscore": {
        "title": "TrendScore",
        "original": "TrendScore System",
        "vol": 11, "cat": "trend",
        "desc": [
            "Tushar Chande's TrendScore: compare today's close against "
            "each of the closes lookback+1 through lookback+10 bars back, "
            "scoring ±1 per comparison — a bounded, outlier-proof trend "
            "census. The score above its SMA with price above its own SMA "
            "buys the next open.",
            "Either condition failing exits; protective, breakeven and "
            "trailing ATR stops cap the damage when the census flips "
            "abruptly.",
        ],
    },
    "exp11_triple_play": {
        "title": "Triple Play",
        "original": "Triple Play System",
        "vol": 11, "cat": "pattern",
        "desc": [
            "Lindsay's Trident with Bollinger confirmation: swing low "
            "outside the lower band (point 1), swing high outside the "
            "upper band (point 2), then a retracement swing low back "
            "inside the bands at 38–62% of the swing (point 3). The entry "
            "stop a quarter-swing above point 3 trades the projected "
            "second leg.",
            "Management follows the Trident script: stop past point 3, "
            "breakeven after target 1, a trailing quarter-drop stop once "
            "price reaches point 2's level, and a measured target 2. The "
            "original trades two units; this adaptation trades one.",
        ],
    },
    "exp11_win_gapper": {
        "title": "Win One for the Gapper",
        "original": "Win One for the Gapper System",
        "vol": 11, "cat": "pattern",
        "desc": [
            "Larry Williams' 'Oops' fade pointed with the trend: in an "
            "ADX-strong uptrend (ADX above 25 and rising, DMI+ leading), a "
            "gap-down open that recovers to the previous bar's low traps "
            "the gap sellers — buy their covering. Mirrored for gap-up "
            "opens in downtrends.",
            "Daily-bar gaps are read from open versus prior extreme "
            "exactly as the source defines them; the four common ATR stops "
            "manage the trade.",
        ],
    },

    # ------------------------------------------------------------ volume 12
    "exp12_first_prize": {
        "title": "First Prize",
        "original": "First Prize System",
        "vol": 12, "cat": "pullback",
        "desc": [
            "The first MACD pullback of a new up leg: the signal line "
            "above zero dates the leg, MACD dipping under the signal once "
            "— and only once — is the pullback, and a rising EMA vets the "
            "trend. Entry on a return to the setup bar's high.",
            "Seven exits manage the trade (protective, breakeven, "
            "trailing, volatility, big-profit, countertrend and "
            "trend-reversal stops) — the most heavily managed system in "
            "the pack, and a showcase for fitness functions that punish "
            "drawdown.",
        ],
    },
    "exp12_in_the_zone": {
        "title": "In The Zone",
        "original": "In The Zone System",
        "vol": 12, "cat": "levels",
        "desc": [
            "Joseph Stowell's support-zone trade: a thrust close above the "
            "high of bar 3 proves demand, then a close back inside the "
            "support zone (under the previous close) offers the discount. "
            "The entry stop at the setup bar's high requires the zone to "
            "hold and lift.",
            "The protective stop under the setup bar frames the zone as "
            "risk; breakeven arrives at a risk multiple and an ATR profit "
            "target above the thrust bar completes the bracket.",
        ],
    },
    "exp12_jailbreak": {
        "title": "Jailbreak",
        "original": "Jailbreak System",
        "vol": 12, "cat": "breakout",
        "desc": [
            "A Donchian breakout with decoupled clocks: enter on the "
            "Length1-bar extreme, exit on the shorter of the two channel "
            "lengths — the asymmetry lets winners breathe while losers "
            "leave on the fast clock. An ATR protective stop caps the gap "
            "risk between the two.",
            "Three parameters, honest surface, and the pack's cleanest "
            "demonstration that exit lengths matter as much as entry "
            "lengths.",
        ],
    },
    "exp12_lr_crossing": {
        "title": "LR Crossing",
        "original": "LR Crossing System",
        "vol": 12, "cat": "trend",
        "desc": [
            "Fernando Zayas' regression crossover: two linear-regression "
            "endpoints, fast and slow, replace moving averages — "
            "regression endpoints hug turns tighter than SMAs of the same "
            "length. The cross reverses the position, always in the "
            "market.",
            "The original's dollar-based stop and target are exposed as "
            "optional ATR multiples, with 0 disabling them to match the "
            "EasyLanguage defaults exactly.",
        ],
    },
    "exp12_ntw": {
        "title": "Not That Way, The Other Way",
        "original": "Not That Way The Other Way System",
        "vol": 12, "cat": "oscillator",
        "desc": [
            "Reverse divergence, the less famous sibling: price puts in a "
            "LOWER swing high while MACD makes a HIGHER high — momentum "
            "leads price down, per the chapter, and the system shorts the "
            "failure. A MACD cross below its signal then arms a stop below "
            "the setup close, scaled by the pivot spread. The long side "
            "mirrors.",
            "The pivot-scaled protective stop, breakeven shift and "
            "measured profit target all key off the divergence geometry "
            "itself.",
        ],
    },
    "exp12_kinda_shifty": {
        "title": "Seems Kinda Shifty",
        "original": "Seems Kinda Shifty System",
        "vol": 12, "cat": "trend",
        "desc": [
            "A dual-SMA system whose exit migrates: crossings under the "
            "fast average at successively lower levels keep the exit at "
            "the fast SMA (tight), while higher-level crossings shift it "
            "back to a close through the slow SMA (loose). The exit "
            "adapts to how orderly the trend is behaving.",
            "Entries are plain dual-SMA crossovers; all the chapter's "
            "invention is in the shifting exit state machine, ported "
            "as-is.",
        ],
    },
    "exp12_springs": {
        "title": "Springs & Upthrusts",
        "original": "Springs & Upthrusts System",
        "vol": 12, "cat": "pattern",
        "desc": [
            "Wyckoff's spring: a wide-range bar violates a confirmed swing "
            "low but closes back above it near the top of its range — the "
            "breakdown found no sellers. Buy a push beyond a fraction of "
            "the spring bar's range; upthrusts mirror the logic at swing "
            "highs.",
            "The stop under the violated swing frames the trade; the "
            "modified parabolic (the Vol. 8/10 machinery) trails the "
            "recovery.",
        ],
    },
    "exp12_three_musketeers": {
        "title": "Three Musketeers",
        "original": "Three Musketeers System",
        "vol": 12, "cat": "oscillator",
        "desc": [
            "Stochastic position plus a micro-dip: Slow %D above 50 keeps "
            "the market's medium pulse bullish while a dip under the low "
            "of bar 3 supplies the entry discount. The stop at the setup "
            "bar's high demands an immediate recovery.",
            "The bracket is risk-defined from the setup bar: stop under "
            "it, breakeven after an ATR move, profit target at a multiple "
            "of the initial risk.",
        ],
    },
    "exp12_mist2": {
        "title": "Tigers in the Mist #2",
        "original": "Tigers in the Mist #2 System",
        "vol": 12, "cat": "pattern",
        "desc": [
            "Walt Downs' failed-gap pattern: after an n-bar breakout bar, "
            "the next bar gaps beyond that bar's extreme — and closes back "
            "through it. The gap crowd is trapped; enter on a return to "
            "the failed bar's extreme and trade their unwind.",
            "The protective stop at the signal bar's opposite extreme "
            "bounds the pattern; the modified parabolic trail manages the "
            "escape.",
        ],
    },

    # ------------------------------------------------------------ volume 13
    "exp13_oc_histogram": {
        "title": "Open-Close Histogram (Vol. 13)",
        "original": "Open-Close Histogram System, Vol. 13 revision",
        "vol": 13, "cat": "trend",
        "desc": [
            "The club's revision of the Vol. 3 histogram: EMA(close) minus "
            "EMA(open) still gauges who wins the average session, but the "
            "entry now adds a fraction-of-ATR offset above the crossover "
            "bar's high and a protective stop below the crossover bar — "
            "the re-tuned risk framing is the whole point of the revisit.",
            "An opposite zero-cross still exits at that close. Comparing "
            "this against exp03_oc_histogram on the same data shows what "
            "five volumes of club feedback changed.",
        ],
    },
    "exp13_dma_range_leaders": {
        "title": "DMA & Range Leaders (Vol. 13)",
        "original": "Displaced MA & Range Leaders System, Vol. 13 revision",
        "vol": 13, "cat": "trend",
        "desc": [
            "The Vol. 6 pairing revisited: a Range Leader bar (wide range, "
            "midpoint above the previous high) closing above the displaced "
            "high average buys at that bar's high. The revision replaces "
            "the original's dollar stops with scalable risk.",
            "In this adaptation those dollar stops become ATR "
            "protective/breakeven/trailing stops — the pack-wide convention "
            "for dollar-denominated risk, keeping the system meaningful "
            "across price scales.",
        ],
    },
    "exp13_linreg_momentum": {
        "title": "Linear Regression & Momentum (Vol. 13)",
        "original": "Linear Regression & Momentum System, Vol. 13 revision",
        "vol": 13, "cat": "pullback",
        "desc": [
            "The Vol. 4 regression-pullback engine, re-tuned: an EMA of the "
            "linear-regression endpoint must rise with the close above it, "
            "momentum must be negative but turning up, and the entry stop "
            "sits an ATR offset above the setup bar for a few bars.",
            "The revision swaps the original exit pair for the common "
            "five-stop ATR suite (protective, breakeven, trailing, "
            "volatility, big-profit), making risk uniform with the rest of "
            "the late volumes.",
        ],
    },
    "exp13_mom_retrace": {
        "title": "Momentum Retracement (Vol. 13)",
        "original": "Momentum Retracement System (MRTS), Vol. 13 revision",
        "vol": 13, "cat": "pullback",
        "desc": [
            "The club's final word on its flagship: the EMA channel and "
            "MACD signal still define the trend and RSI/DMI-spread momentum "
            "still arms the retracement check, but the Vol. 13 revision "
            "tightens the retracement window around the trend extreme and "
            "enters on a two-bar-high stop.",
            "The ATR stop suite plus a trend-failure market exit manage the "
            "position. Set against exp04_mom_retrace, this is the pack's "
            "best before/after study of a system evolving through club "
            "feedback.",
        ],
    },
    "exp13_volex": {
        "title": "VolEx (Vol. 13)",
        "original": "VolEx System, Cruz/Williams/Wright Vol. 13 revision",
        "vol": 13, "cat": "volatility",
        "desc": [
            "The Vol. 1 volatility expansion with a decade of club edits: "
            "stops still bracket the open at multiples of the four-bar "
            "average range (short factor wider), but the exit stack now "
            "adds a protective ATR stop to the profitable-open bailout, "
            "four-bar trail and stale-trade exit.",
            "The four-rule exit interplay is the interesting optimization "
            "surface here; the entry has no free parameters beyond the two "
            "multipliers.",
        ],
    },
    "exp13_vol_clusters": {
        "title": "Volatility Clusters (Vol. 13)",
        "original": "Volatility Clusters System, Vol. 13 revision",
        "vol": 13, "cat": "volatility",
        "desc": [
            "The Vol. 3 cluster detector, hardened: a close-to-close move "
            "or range beyond mean + N standard deviations still brackets "
            "the flag bar with breakout stops at prior three-bar "
            "average-range offsets, but entries may now reverse an open "
            "position and a protective ATR stop backs the "
            "first-profitable-open exit.",
            "Population standard deviations over the rolling window match "
            "EasyLanguage's StdDev, as in the Vol. 3 original — the two "
            "make a clean revision pair on identical data.",
        ],
    },
}
