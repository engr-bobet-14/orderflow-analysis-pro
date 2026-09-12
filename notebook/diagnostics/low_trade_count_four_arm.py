"""
Four-arm ablation for the "only ~0-1 trades over 32 days" defect diagnosed
against notebook/02_cross_validation_tuning.ipynb's run_backtest.

READ-ONLY DIAGNOSTIC. Never imports/monkeypatches orderflow_system on disk.
Does not change any threshold, any InstrumentConfig value, or any strategy
logic. The two "fixes" below exist only as local monkeypatches on a
SignalAggregator *instance* inside this script, applied to reproduce a
counterfactual for measurement purposes -- they are not applied to the repo.

Arms (identical 30,102-candle dataset, identical BASELINE_PARAMS, identical
InstrumentConfig -- nothing but the two flags below ever changes):
  1. neither fix              -- real wall-clock cooldown, stuck-phase bug present
  2. state-machine fix only   -- real wall-clock cooldown, stuck-phase bug patched
  3. candle-time fix only     -- historical-candle-time cooldown, stuck-phase bug present
  4. both fixes                -- historical-candle-time cooldown, stuck-phase bug patched

Bug #1 (stuck phase): SignalAggregator._handle_absorption_at_level calls
trade.advance_to_absorption(signal) -- moving the active trade's phase to
TradePhase.ABSORPTION_DETECTED -- before checking the composite-score gate
a few lines later. On a score miss it returns None without reverting the
phase. process_signal's routing (aggregator.py) has branches for WATCHING,
POSITION_OPEN, and BREAK_EVEN/TRAILING -- none for ABSORPTION_DETECTED --
so every subsequent signal for that instrument silently falls through to
the function's final `return None` for the rest of the run.

Bug #2 (wall-clock cooldown): process_signal's cooldown check computes
`now_ms = int(time.time() * 1000)` -- real wall-clock time -- and compares
it against `signal_cooldown_seconds`. A fast in-process replay over 30,102
candles finishes in a few seconds of real time, so after the first signal
that updates `_last_signal_time`, the cooldown blocks essentially every
later signal for the rest of the run regardless of simulated market time
elapsed. notebook 01's own `HistoricalClock` adapter (Section "Historical
Replay Clock Adapter (Backtest Only)") already fixes this for its own
replay helpers; run_backtest in notebook 02 never adopted it.

Run: .venv_orderflow/bin/python3 notebook/diagnostics/low_trade_count_four_arm.py
Requires the candle cache notebook/.cv_cache/candles_*.pkl to already exist
(built by running notebook 02's Section 2/3 cells once) -- this script does
not re-scan the raw parquet trade feed itself.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from orderflow_system.analytics.delta import DeltaEngine
from orderflow_system.analytics.footprint import FootprintEngine
from orderflow_system.analytics.volume_profile import VolumeProfileEngine
from orderflow_system.patterns.absorption import AbsorptionDetector
from orderflow_system.patterns.initiative import InitiativeDetector
from orderflow_system.patterns.exhaustion import ExhaustionDetector
from orderflow_system.patterns.divergence import DivergenceDetector
from orderflow_system.signals.profile_framing import ProfileFramingEngine
from orderflow_system.signals.aggregator import SignalAggregator
from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.models import SignalType, TradePhase
import orderflow_system.signals.aggregator as agg_module

TICK_SIZE = 0.01
SYMBOL = "BTCUSDT"
CV_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cv_cache"


def load_candles() -> list:
    candidates = sorted(CV_CACHE_DIR.glob("candles_*.pkl"))
    if not candidates:
        raise SystemExit(
            f"No cached candles found under {CV_CACHE_DIR}. Run notebook "
            "02_cross_validation_tuning.ipynb's Section 2/3 cells once to "
            "build the cache, then rerun this script."
        )
    with open(candidates[-1], "rb") as f:
        candles, _omitted = pickle.load(f)
    return candles


def build_daily_profiles(candles: list, config) -> dict:
    candles_by_day = defaultdict(list)
    for c in candles:
        day = dt.datetime.fromtimestamp(c.timestamp_ms / 1000, tz=dt.timezone.utc).date()
        candles_by_day[day].append(c)
    vp_engine = VolumeProfileEngine(config.volume_profile)
    daily_profiles = {}
    for day, day_candles in sorted(candles_by_day.items()):
        p = vp_engine.compute_from_candles(day_candles, session_date=str(day))
        if p.total_volume > 0:
            daily_profiles[day] = p
    return daily_profiles


CONFIG = get_btcusd_config()

BASELINE_PARAMS = {
    "price_proximity_pct": 0.002,
    "min_composite_score": 40.0,
    "signal_cooldown_seconds": 30.0,
    "min_aggressive_volume": float(CONFIG.absorption.min_aggressive_volume),
    "absorption_max_price_displacement_ticks": float(CONFIG.absorption.max_price_displacement_ticks),
    "big_trade_filter": float(CONFIG.absorption.big_trade_filter),
    "min_delta_threshold": float(CONFIG.initiative.min_delta_threshold),
    "initiative_min_price_displacement_ticks": float(CONFIG.initiative.min_price_displacement_ticks),
    "volume_decline_pct": float(CONFIG.exhaustion.volume_decline_pct),
    "lookback_bars": int(CONFIG.divergence.lookback_bars),
    "delta_failure_pct": float(CONFIG.divergence.delta_failure_pct),
}


def build_instrument_config(params: dict):
    base = get_btcusd_config()
    absorption = replace(
        base.absorption,
        min_aggressive_volume=params["min_aggressive_volume"],
        max_price_displacement_ticks=params["absorption_max_price_displacement_ticks"],
        big_trade_filter=params["big_trade_filter"],
    )
    initiative = replace(
        base.initiative,
        min_delta_threshold=params["min_delta_threshold"],
        min_price_displacement_ticks=params["initiative_min_price_displacement_ticks"],
    )
    exhaustion = replace(base.exhaustion, volume_decline_pct=params["volume_decline_pct"])
    divergence = replace(
        base.divergence,
        lookback_bars=params["lookback_bars"],
        delta_failure_pct=params["delta_failure_pct"],
    )
    return replace(base, absorption=absorption, initiative=initiative, exhaustion=exhaustion, divergence=divergence)


class _HistoricalClock:
    """Backtest-only stand-in for time.time(): reports the current candle's
    historical timestamp (seconds) instead of the real wall clock. Mirrors
    notebook 01's HistoricalClock adapter. Never imported by orderflow_system."""

    def __init__(self):
        self.now_ms = 0

    def time(self) -> float:
        return self.now_ms / 1000.0


def run_arm(
    candles: list,
    daily_profiles: dict,
    instrument_config,
    params: dict,
    fix_stuck_phase: bool,
    use_historical_clock: bool,
) -> dict:
    de = DeltaEngine(tick_size=TICK_SIZE)
    fe = FootprintEngine(tick_size=TICK_SIZE)
    absorption_d = AbsorptionDetector(instrument_config.absorption, tick_size=TICK_SIZE)
    initiative_d = InitiativeDetector(instrument_config.initiative, tick_size=TICK_SIZE)
    exhaustion_d = ExhaustionDetector(instrument_config.exhaustion)
    divergence_d = DivergenceDetector(instrument_config.divergence)
    framing = ProfileFramingEngine()
    agg = SignalAggregator(
        min_composite_score=params["min_composite_score"],
        signal_cooldown_seconds=params["signal_cooldown_seconds"],
        price_proximity_pct=params["price_proximity_pct"],
    )

    if fix_stuck_phase:
        # Diagnostic-only counterfactual: route ABSORPTION_DETECTED the same
        # way WATCHING is routed, so a score-miss can be retried on the next
        # absorption signal instead of permanently wedging the trade.
        # Patches this SignalAggregator *instance* only -- never written to
        # orderflow_system/signals/aggregator.py.
        orig_process = agg.process_signal

        def patched_process_signal(instrument, signal, bias, current_price, recent_candles):
            active_trade = agg._active_trades.get(instrument)
            if active_trade is not None and active_trade.phase == TradePhase.ABSORPTION_DETECTED:
                if signal.signal_type == SignalType.ABSORPTION:
                    return agg._handle_absorption_at_level(
                        instrument, signal, bias, active_trade, current_price,
                        int(agg_module.time.time() * 1000),
                    )
                return None
            return orig_process(instrument, signal, bias, current_price, recent_candles)

        agg.process_signal = patched_process_signal

    raw_signals = []
    raw_by_type = Counter()
    actions = []
    recent = []
    current_day = None
    current_bias = None

    hc = _HistoricalClock()
    clock_patch = patch("orderflow_system.signals.aggregator.time.time", hc.time) if use_historical_clock else None
    if clock_patch:
        clock_patch.start()
    try:
        for c in candles:
            if use_historical_clock:
                hc.now_ms = c.timestamp_ms

            day = dt.datetime.fromtimestamp(c.timestamp_ms / 1000, tz=dt.timezone.utc).date()
            if day != current_day:
                prev_day, current_day = current_day, day
                if prev_day is not None and prev_day in daily_profiles:
                    framing.add_profile(daily_profiles[prev_day])
                    current_bias = framing.analyze(current_price=c.open)
                    for level in current_bias.qualified_levels:
                        if level.strength >= 50:
                            agg.set_watching(SYMBOL, level, level.direction)

            d = de.compute_from_candle(c)
            fp_bar = fe.build_from_candle(c)
            sig_list = [
                absorption_d.check_candle(c, fp_bar, d, c.close),
                initiative_d.check_candle(c, d, fp_bar),
                exhaustion_d.check_candle(c, d, de, fp_bar, recent),
                divergence_d.check_candle(c, de),
            ]
            signals = [s for s in sig_list if s is not None]
            for s in signals:
                raw_by_type[s.signal_type.value] += 1
            raw_signals.extend(signals)

            for sig in signals:
                out = agg.process_signal(
                    instrument=SYMBOL, signal=sig, bias=current_bias,
                    current_price=c.close, recent_candles=recent[-5:],
                )
                if out is not None:
                    actions.append((c.timestamp_ms, out))

            recent.append(c)
            if len(recent) > 20:
                recent = recent[-20:]
    finally:
        if clock_patch:
            clock_patch.stop()

    action_types = Counter(a.action for _, a in actions)

    # Pair enter -> next exit, same convention as
    # notebook 02's simulate_trades_risk_based: a trade still open at the
    # end of the sample is reported separately, not counted as completed.
    completed = 0
    open_trade = False
    for _, a in actions:
        if a.action == "enter" and not open_trade:
            open_trade = True
        elif a.action == "exit" and open_trade:
            completed += 1
            open_trade = False

    return {
        "raw_signals": len(raw_signals),
        "raw_by_type": dict(raw_by_type),
        "actions": len(actions),
        "action_types": dict(action_types),
        "entries": action_types.get("enter", 0),
        "exits": action_types.get("exit", 0),
        "completed_trades": completed,
        "open_at_end": open_trade,
    }


ARMS = [
    ("1_neither_fix", False, False),
    ("2_state_machine_fix_only", True, False),
    ("3_candle_time_fix_only", False, True),
    ("4_both_fixes", True, True),
]


def main():
    candles = load_candles()
    daily_profiles = build_daily_profiles(candles, CONFIG)
    instrument_config = build_instrument_config(BASELINE_PARAMS)

    n_days = len({
        dt.datetime.fromtimestamp(c.timestamp_ms / 1000, tz=dt.timezone.utc).date()
        for c in candles
    })
    print(f"{len(candles):,} candles across {n_days} calendar days\n")

    results = {}
    header = f"{'arm':30s} {'raw_signals':>11s} {'actions':>8s} {'entries':>8s} {'exits':>7s} {'completed':>10s} {'open_end':>9s}"
    print(header)
    print("-" * len(header))
    for name, fix_stuck_phase, use_historical_clock in ARMS:
        r = run_arm(candles, daily_profiles, instrument_config, BASELINE_PARAMS, fix_stuck_phase, use_historical_clock)
        results[name] = r
        print(
            f"{name:30s} {r['raw_signals']:11d} {r['actions']:8d} {r['entries']:8d} "
            f"{r['exits']:7d} {r['completed_trades']:10d} {str(r['open_at_end']):>9s}"
        )

    print()
    for name, r in results.items():
        print(f"{name}: raw_by_type={r['raw_by_type']}  action_types={r['action_types']}")

    out_path = Path(__file__).resolve().parent / "low_trade_count_four_arm_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_candles": len(candles),
                "n_days": n_days,
                "baseline_params": BASELINE_PARAMS,
                "arms": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
