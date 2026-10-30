"""
End-to-end validation of the info-bar pipeline against the real external dataset.

    python -m orderflow_system.validate_infobars --bar-type all --start 2023-01-01 --end 2023-04-01
    python -m orderflow_system.validate_infobars --bar-type fixed_dollar_imbalance          # whole history

For each bar type it checks, on the REAL files (read-only):
  * schema + per-row integrity (strict reader: any bad row raises)
  * chronological order, tie handling, data-outage bars
  * pipeline runs to completion
  * determinism: a second replay yields an identical digest
  * no look-ahead: replaying only the first 40% of bars reproduces the same output
    for those bars (digest of the prefix == digest of the same prefix in the full run)
  * the pipeline emitted no price-level (footprint) data
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
import json
import sys
import time
from typing import Optional

from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.infobar_feed import InfoBarReader, open_bars
from orderflow_system.data.models import BarType, Side, SignalType
from orderflow_system.pipeline import InfoBarPipeline, ReplayDigest
from orderflow_system.ledger import TradeLedger


class LedgerAudit:
    """Audit actual execution snapshots while each closing bar is available."""

    def __init__(self, price_proximity_pct: float):
        self.ledger = TradeLedger()
        self.price_proximity_pct = price_proximity_pct
        self.counters = Counter()
        self.violations = Counter()
        self.examples = []

    def _check(self, condition, name, bar):
        if not condition:
            self.violations[name] += 1
            if len(self.examples) < 20:
                self.examples.append({"bar_seq": bar.seq, "violation": name})

    def process_result(self, result):
        bar = result.bar
        if bar.data_gap:
            self._check(not result.signals, "gap_detector_signal", bar)
        for aggregate in result.aggregated:
            self.counters["aggregates"] += 1
            execution = aggregate.execution
            self._check(aggregate.bar_seq == bar.seq == execution.bar_seq, "aggregate_bar_seq", bar)
            self._check(execution.timestamp_ms == bar.timestamp_ms, "aggregate_timestamp", bar)
            self._check(aggregate.direction == execution.direction, "aggregate_direction", bar)
            if bar.data_gap:
                self._check(aggregate.action in ("stop", "target"), "gap_signal_driven_action", bar)
                if aggregate.action in ("stop", "target"):
                    self.counters["protective_gap_exits"] += 1
            if aggregate.action in ("break_even", "trail"):
                self.counters["management_actions"] += 1
                valid = (execution.stop_loss < bar.close if execution.direction == Side.BUY
                         else execution.stop_loss > bar.close)
                self._check(valid, "invalid_management_stop", bar)
            if aggregate.action == "enter":
                absorption = [signal for signal in aggregate.signals
                              if signal.signal_type == SignalType.ABSORPTION]
                self._check(bool(absorption) and all(signal.direction == execution.direction for signal in absorption),
                            "entry_signal_direction", bar)
        for record in self.ledger.process_result(result):
            execution = record.execution
            long = execution.direction == Side.BUY
            if record.action == "enter":
                self.counters["entries"] += 1
                level = execution.qualified_level
                self._check(level is not None and level.direction == execution.direction, "entry_level_direction", bar)
                if level is not None:
                    distance = abs(execution.entry_price - level.price) / max(execution.entry_price, 1.0)
                    self._check(distance < self.price_proximity_pct, "entry_proximity", bar)
                valid = (execution.stop_loss < execution.entry_price < execution.take_profit if long
                         else execution.take_profit < execution.entry_price < execution.stop_loss)
                self._check(valid, "entry_sl_tp", bar)
                self._check(execution.fill_price == bar.close == execution.entry_price, "entry_fill", bar)
                continue
            self.counters["closes"] += 1
            self.counters[record.action] += 1
            fill = execution.fill_price
            self._check(fill is not None, "missing_close_fill", bar)
            if fill is None:
                continue
            expected_pnl = fill - execution.entry_price if long else execution.entry_price - fill
            self._check(execution.pnl is not None and math.isclose(execution.pnl, expected_pnl, abs_tol=1e-10),
                        "close_pnl", bar)
            self._check(execution.exit_reason == record.action, "close_reason", bar)
            if record.action == "exit":
                self._check(fill == bar.close, "pattern_exit_fill", bar)
            elif record.action == "stop":
                expected = min(execution.stop_loss, bar.open) if long else max(execution.stop_loss, bar.open)
                self._check(fill == expected, "stop_fill", bar)
                self._check(bar.low <= fill <= bar.high, "stop_outside_bar", bar)
            elif record.action == "target":
                self._check(fill == execution.take_profit, "target_fill", bar)
                gap_through = bar.open > execution.take_profit if long else bar.open < execution.take_profit
                if gap_through:
                    self.counters["target_gap_through"] += 1
                self._check(bar.low <= fill <= bar.high or gap_through, "target_outside_bar", bar)

    def report(self):
        counters = {name: self.counters[name] for name in (
            "aggregates", "entries", "closes", "stop", "target", "exit",
            "management_actions", "protective_gap_exits", "target_gap_through")}
        return {"counters": counters, "violations": dict(self.violations),
                "violation_count": sum(self.violations.values()), "examples": self.examples,
                "ok": not self.violations}


def check_start_independence(info):
    """Compare the first post-outage bar with three starts and an unbounded start."""
    missing = info.missing_hours_ms
    if not missing:
        return {"ok": True, "skipped": "no outage reported"}
    first = missing[0]
    end_of_outage = first + 3_600_000
    for hour in missing[1:]:
        if hour != end_of_outage:
            break
        end_of_outage += 3_600_000
    baseline = next((bar for bar in InfoBarReader(info) if bar.timestamp_ms >= end_of_outage), None)
    if baseline is None:
        return {"ok": False, "error": "no bar after first reported outage"}
    starts = (first - 1_800_000, first, end_of_outage)
    samples = []
    for start in starts:
        found = next((bar for bar in InfoBarReader(info, start_ms=start) if bar.seq == baseline.seq), None)
        samples.append({"start_ms": start, "duration_ms": found.duration_ms if found else None,
                        "data_gap": found.data_gap if found else None})
    return {"ok": all((sample["duration_ms"], sample["data_gap"]) ==
                      (baseline.duration_ms, baseline.data_gap) for sample in samples),
            "bar_seq": baseline.seq, "duration_ms": baseline.duration_ms,
            "data_gap": baseline.data_gap, "starts": samples}


def _replay(root, symbol, bar_type, start, end, limit: Optional[int] = None, snapshot_at: Optional[int] = None):
    """Replay once. Returns (reader, pipeline, final digest, price-level count, seconds, digest after `snapshot_at` bars)."""
    reader = open_bars(root, symbol, bar_type, start=start, end=end, strict=True)
    pipe = InfoBarPipeline(get_btcusd_config(), bar_type)
    digest = ReplayDigest()
    snapshot = None
    price_levels = 0
    pipe.ledger_audit = LedgerAudit(pipe.aggregator.price_proximity_pct)
    t0 = time.time()
    for bar in reader:
        if limit is not None and pipe.bars_processed >= limit:
            break
        res = pipe.process_bar(bar)
        digest.update(res)
        pipe.ledger_audit.process_result(res)
        price_levels += res.footprint.has_price_levels or bool(res.delta.horizontal_delta)
        if snapshot_at is not None and pipe.bars_processed == snapshot_at:
            snapshot = digest.hexdigest()
    return reader, pipe, digest, price_levels, time.time() - t0, snapshot


def validate(root: str, symbol: str, bar_type: BarType, start: Optional[str], end: Optional[str]) -> dict:
    reader, pipe, digest, price_levels, secs, _ = _replay(root, symbol, bar_type, start, end)
    n = pipe.bars_processed
    rep = reader.report
    k = max(1, int(n * 0.4))

    # Second full replay: must be identical (determinism); also record where the first k bars stood.
    _, _, digest2, _, _, prefix_in_full = _replay(root, symbol, bar_type, start, end, snapshot_at=k)
    # Replay of ONLY the first k bars: must equal the same bars inside the full run (no look-ahead).
    _, _, d_prefix, _, _, _ = _replay(root, symbol, bar_type, start, end, limit=k)

    ledger_audit = pipe.ledger_audit.report()
    start_independence = check_start_independence(reader.info)
    checks = {
        "trade_ledger_valid": ledger_audit["ok"],
        "window_start_independent": start_independence["ok"],
        "pipeline_completed": n > 0 and n == rep.rows_yielded,
        "no_rows_rejected": rep.rows_rejected == 0,
        "deterministic_replay": digest.hexdigest() == digest2.hexdigest(),
        "no_lookahead_prefix_invariant": d_prefix.hexdigest() == prefix_in_full,
        "no_price_level_data_fabricated": price_levels == 0,
    }
    stats = pipe.stats
    return {
        "bar_type": bar_type.value,
        "dataset": str(reader.info.parquet_path),
        "bars": n,
        "dataset_id": reader.info.dataset_id,
        "ledger_audit": ledger_audit,
        "start_independence": start_independence,
        "first_close_utc": _iso(rep.first_ts_ms),
        "last_close_utc": _iso(rep.last_ts_ms),
        "tied_timestamps": rep.tied_timestamps,
        "max_gap_hours": round(rep.max_gap_ms / 3.6e6, 2),
        "bars_spanning_missing_hours": rep.bars_spanning_missing_hours,
        "sessions_closed": stats["sessions_closed"],
        "signals": stats["signals"],
        "actions": stats["actions"],
        "gap_bars_skipped": stats["gap_bars_skipped"],
        "digest": digest.hexdigest()[:16],
        "bars_per_second": round(n / secs) if secs else None,
        "checks": checks,
        "ok": all(checks.values()),
    }


def _iso(ms):
    from datetime import datetime, timezone
    return None if ms is None else datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bar-type", default="all", help="a bar type or 'all'")
    ap.add_argument("--root", default="/Volumes/Market-Data/info-bars")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args(argv)

    types = list(BarType) if args.bar_type == "all" else [BarType(args.bar_type)]
    ok = True
    for bt in types:
        out = validate(args.root, args.symbol, bt, args.start, args.end)
        print(json.dumps(out, indent=2))
        ok &= out["ok"]
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
