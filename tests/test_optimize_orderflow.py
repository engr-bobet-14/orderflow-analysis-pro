"""Frozen work-order proofs. All generated data/output lives under tmp_path."""
from dataclasses import replace
from datetime import date
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import make_rows, write_dataset
from scripts import optimize_orderflow as opt
from test_pipeline import synth_bars
from orderflow_system.config.settings import INFO_BARS, get_btcusd_config
from orderflow_system.data.infobar_feed import open_bars
from orderflow_system.data.models import BarType, Side
from orderflow_system.pipeline import InfoBarPipeline

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/optimize_orderflow.py"


@pytest.fixture(autouse=True)
def neutral_environment(monkeypatch):
    for name in ("START_DATE", "END_DATE", "ORDERFLOW_TRADES_ROOT"):
        monkeypatch.delenv(name, raising=False)


def cli_args(tmp_path, n=120):
    source = tmp_path / "source"
    write_dataset(source, make_rows(n, step_s=3600))
    return ["--bar-type", "fixed_dollar_imbalance", "--start-date", "2023-01-01",
            "--end-date", "2023-02-01", "--n-trials", "2", "--n-splits", "3",
            "--data-root", str(source), "--study-db", str(tmp_path / "s.db"),
            "--cache-dir", str(tmp_path / "cache"), "--output", str(tmp_path / "r.json")]


def args_for(tmp_path, **kw):
    args = opt.build_parser().parse_args(cli_args(tmp_path))
    args.symbol = "BTCUSDT"
    args.bar_type = "fixed_dollar_imbalance"
    args.space = dict(opt.SEARCH_SPACE_BOUNDS)
    for name, value in kw.items():
        setattr(args, name, value)
    return args


def entry_of(results):
    studies = results["symbols"]["BTCUSDT"]["fixed_dollar_imbalance"]
    return next(iter(studies.values()))


def run_cli(argv, **kw):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(SCRIPT), *argv], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=90, **kw)


@pytest.mark.parametrize("argv", [
    ["--symbol", "ETHUSDT"], ["--symbol", "x/y"], ["--bar-type", "junk"],
    ["--n-trials", "0"], ["--n-jobs", "-2"], ["--n-splits", "1"],
    ["--embargo", "nan"], ["--holdout-fraction", "1"], ["--fee-bps", "inf"],
    ["--slippage-bps", "-1"], ["--seed", "-1"], ["--slippage-bps", "10000"],
    ["--start-date", "2023-02-01", "--end-date", "2023-01-01"],
    ["--start-date", "not-a-day"], ["--evaluate-holdout", "--holdout-fraction", "0"],
    ["--risk-per-trade", "1"], ["--capital", "1000"],
])
def test_cli_errors_before_data_work(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        opt.main(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_help_defaults_and_env(monkeypatch):
    assert run_cli(["--help"]).returncode == 0
    p = opt.build_parser()
    args = p.parse_args([])
    assert args.bar_type == [b.value for b in BarType]
    assert args.symbol == ["BTCUSDT"]
    assert len(opt.SEARCH_SPACE_BOUNDS) == 6
    assert not any("risk" in k or "capital" in k for k in opt.SEARCH_SPACE_BOUNDS)
    monkeypatch.setenv("START_DATE", "2023-01-02")
    monkeypatch.setenv("END_DATE", "2023-02-03")
    args = opt.build_parser().parse_args([])
    assert args.start_date == date(2023, 1, 2) and args.end_date == date(2023, 2, 3)


@pytest.mark.parametrize("override", [
    {"risk": {"low": 1, "high": 2}}, {"lookback_bars": {"low": 1.5, "high": 3}},
    {"min_composite_score": {"low": 20, "high": 10}},
    {"volume_decline_pct": {"low": .1, "high": 2}},
    {"price_proximity_pct": {"low": 0, "high": 2}},
    {"lookback_bars": {"low": True, "high": 3}},
    {"lookback_bars": [1, 3]}, {"lookback_bars": {"low": 2, "high": float("nan")}},
])
def test_space_validation(tmp_path, override):
    path = tmp_path / "space.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        opt.load_space(path)


def notebook_namespace():
    """Execute definition cells verbatim; skip notebook I/O and baseline cell setup."""
    nb = json.loads((ROOT / "notebook/02_cross_validation_tuning.ipynb").read_text())
    ns = {"CONFIG": get_btcusd_config(), "pipeline": InfoBarPipeline(get_btcusd_config()),
          "INFO_BARS": INFO_BARS, "get_btcusd_config": get_btcusd_config,
          "InfoBarPipeline": InfoBarPipeline, "TradeLedger": opt.TradeLedger, "Path": Path}
    for index in range(15, 21):
        exec(compile("".join(nb["cells"][index]["source"]), f"notebook-cell-{index}", "exec"), ns)
    return ns


def params():
    return {"price_proximity_pct": .01, "min_composite_score": 10,
            "signal_cooldown_seconds": 5, "volume_decline_pct": .3,
            "lookback_bars": 5, "delta_failure_pct": .7}


def test_sizing_and_costs_match_notebook(tmp_path):
    ns = notebook_namespace()
    # Synthetic immutable snapshots make all sizing cases explicit.
    entries, closes = [], []
    for i, (entry, stop, exit_price, side) in enumerate([
        (100., 95., 110., Side.BUY), (100., 99.999, 90., Side.BUY),
        (100., 105., 90., Side.SELL), (100., 100., 110., Side.SELL),
    ]):
        opened = SimpleNamespace(trade_id=str(i), bar_seq=2*i, timestamp_ms=1672531200000+i*86400000,
                                 direction=side, fill_price=entry, stop_loss=stop, take_profit=120.)
        closed = SimpleNamespace(trade_id=str(i), bar_seq=2*i+1, timestamp_ms=opened.timestamp_ms+60000,
                                 fill_price=exit_price, exit_reason="target", pnl=exit_price-entry)
        entries.append(SimpleNamespace(execution=opened))
        closes.append(SimpleNamespace(execution=closed))
    ledger = SimpleNamespace(entries=entries, closes=closes)
    actual = opt.simulate_trades_risk_based(ledger)
    pd.testing.assert_frame_equal(actual, ns["simulate_trades_risk_based"](ledger))
    assert (actual.position_notional <= actual.equity_before).all()
    assert actual.iloc[0].dollar_risk == 50
    assert actual.iloc[1].stop_loss_pct == .0005
    assert actual.iloc[1].position_notional == actual.iloc[1].equity_before
    assert (actual.fee > 0).all() and (actual.slippage > 0).all()


def trade_table(pnls, days):
    return pd.DataFrame({"net_pnl": pnls, "exit_time": pd.to_datetime(days, utc=True)})


def test_day_one_loss_and_trailing_flat_days():
    import quantstats as qs
    trades = trade_table([-4000., 50.], ["2023-01-01", "2023-01-02"])
    start, end = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-01-05", tz="UTC")
    returns = opt.daily_returns(trades, start, end)
    reference = pd.Series([5000., 1000., 1050., 1050., 1050., 1050.],
                          index=pd.date_range("2022-12-31", "2023-01-05", tz="UTC")).pct_change().fillna(0)
    pd.testing.assert_series_equal(returns, reference)
    assert returns.loc[start] == -.8
    metrics = opt.compute_financial_metrics(trades, start_ts=start, end_ts=end)
    assert metrics["max_drawdown_pct"] == -80
    assert metrics["sharpe_ratio"] == qs.stats.sharpe(reference, periods=365)
    assert metrics["sortino_ratio"] == qs.stats.sortino(reference, periods=365)
    # Parity only with identical last observation day, as the work order requires.
    short = opt.compute_financial_metrics(trades, start_ts=start, end_ts=start + pd.Timedelta(days=1))
    legacy = notebook_namespace()["compute_financial_metrics"](trades, start_ts=start-pd.Timedelta(days=1))
    assert short == legacy


def test_all_winning_metrics_serialize():
    trades = trade_table([10., 20.], ["2023-01-01", "2023-01-02"])
    metrics = opt.compute_financial_metrics(trades, start_ts=pd.Timestamp("2023-01-01", tz="UTC"),
                                           end_ts=pd.Timestamp("2023-01-02", tz="UTC"))
    metrics["profit_factor"] = opt.compute_profit_factor(trades)
    assert metrics["profit_factor"] is None
    safe = opt.json_safe(metrics)
    assert safe["sortino_ratio"] is None and safe["sortino_ratio_status"] == "non_finite"
    json.dumps(safe, allow_nan=False)


def assert_prefix_parity(bars, args):
    p = params()
    folds = opt.build_cv_folds(bars, 3, .02)
    ends = {int(f["test_idx"].max()) for f in folds}
    whole = opt.run_backtest(bars, opt.build_instrument_config(p), p["price_proximity_pct"],
                             bars[0].bar_type, prefix_ends=ends)
    all_trades = opt.simulate_trades_risk_based(whole["ledger"])
    assert len(all_trades) > 0, "parity must exercise completed trades"
    ns = notebook_namespace()
    notebook_folds = ns["build_cv_folds"](bars, 3, .02)
    for fold, nf in zip(folds, notebook_folds):
        for key in fold:
            np.testing.assert_array_equal(fold[key], nf[key])
        end, start = int(fold["test_idx"].max()), fold["score_start_idx"]
        prefix = opt.run_backtest(bars[:end+1], opt.build_instrument_config(p), p["price_proximity_pct"], bars[0].bar_type)
        assert prefix["replay_digest"] == whole["prefix_digests"][end]
        prefix_trades = opt.simulate_trades_risk_based(prefix["ledger"])
        selected = all_trades.loc[all_trades.exit_seq <= bars[end].seq].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, prefix_trades, check_dtype=not selected.empty)
        actual = opt.evaluate_fold(fold, bars, p, args, all_trades)
        reference = ns["evaluate_fold"](nf, bars, p)
        for key in ("n_trades", "total_pnl", "win_rate_pct", "n_context_bars", "n_test_bars", "n_scored_bars", "account_blown"):
            assert actual[key] == reference[key]
        notebook_replay = ns["run_backtest"](bars[:end+1], ns["build_instrument_config"](p), p["price_proximity_pct"])
        pd.testing.assert_frame_equal(prefix_trades, ns["simulate_trades_risk_based"](notebook_replay["ledger"]))
        scored = prefix_trades.loc[prefix_trades.exit_seq.between(bars[start].seq, bars[end].seq)]
        end_day = pd.Timestamp(bars[end].timestamp_ms, unit="ms", tz="UTC").floor("D")
        if len(scored) and scored.exit_time.max().floor("D") == end_day:
            shifted = pd.Timestamp(bars[start].timestamp_ms, unit="ms", tz="UTC") - pd.Timedelta(days=1)
            metrics = ns["compute_financial_metrics"](scored, start_ts=shifted)
            for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct"):
                assert actual[key] == pytest.approx(metrics[key], nan_ok=True)
        assert all(fold["context_idx"] < fold["test_idx"].min())
        assert end - start + 1 == actual["n_scored_bars"]
    np.testing.assert_array_equal(np.concatenate([f["test_idx"] for f in folds]), np.arange(len(bars)))


def test_single_pass_notebook_parity(tmp_path):
    assert_prefix_parity(synth_bars(6000, seed=3), args_for(tmp_path))


@pytest.mark.real_data
@pytest.mark.skipif(not Path(INFO_BARS.root, "BTCUSDT").is_dir(), reason="real dataset not mounted")
def test_real_prefix_and_notebook_parity(tmp_path):
    bars = list(open_bars(INFO_BARS.root, "BTCUSDT", BarType.DOLLAR_IMBALANCE, "2023-06-01", "2023-06-04"))
    assert_prefix_parity(bars, args_for(tmp_path))


def test_fold_validation_and_holdout_rounding(tmp_path):
    bars = synth_bars(100)
    args = args_for(tmp_path, holdout_fraction=.001)
    with pytest.raises(ValueError, match=r"N=100.*holdout bars=0.*blackout bars=2"):
        opt.split_timeline(bars, args)
    args.holdout_fraction, args.embargo = .02, .03
    with pytest.raises(ValueError, match="holdout bars=2.*blackout bars=3"):
        opt.split_timeline(bars, args)
    with pytest.raises(ValueError, match="consumed"):
        opt.build_cv_folds(bars, 5, .2)
    with pytest.raises(ValueError, match="strictly increasing"):
        opt.build_cv_folds([bars[0], bars[0]], 2, 0)


class CaptureTrial:
    def __init__(self):
        self.user_attrs = {}
    def suggest_int(self, name, *args, **kw):
        return params()[name]
    def suggest_float(self, name, *args, **kw):
        return params()[name]
    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_no_future_or_holdout_replay(tmp_path, monkeypatch):
    bars = synth_bars(4500, seed=3)
    args = args_for(tmp_path)
    n_search, folds = opt.split_timeline(bars, args)
    original = opt.run_backtest
    seen = []
    def replay(segment, *a, **kw):
        seen.append(len(segment))
        return original(segment, *a, **kw)
    monkeypatch.setattr(opt, "run_backtest", replay)
    a, b = CaptureTrial(), CaptureTrial()
    opt.objective(a, bars[:n_search], folds, args)
    mutated = bars[:n_search] + [replace(bar, close=1e9, high=1e9) for bar in bars[n_search:]]
    opt.objective(b, mutated[:n_search], folds, args)
    assert seen == [n_search, n_search] and a.user_attrs == b.user_attrs
    fold = folds[0]
    end = int(fold["test_idx"].max())
    wild = bars[:end+1] + [replace(bar, close=1e9, high=1e9) for bar in bars[end+1:]]
    assert opt.evaluate_fold(fold, bars, params(), args) == opt.evaluate_fold(fold, wild, params(), args)
    c = CaptureTrial()
    opt.objective(c, wild[:n_search], folds, args)
    assert c.user_attrs["fold_results"][0] == a.user_attrs["fold_results"][0]
    assert c.user_attrs["fold_results"][1:] != a.user_attrs["fold_results"][1:]
    assert a.user_attrs["total_trades"] > 0


def test_optimizer_entry_points_replay_search_only(tmp_path, monkeypatch):
    args = args_for(tmp_path, n_trials=1, n_jobs=1)
    bars = synth_bars(300)
    path = tmp_path/"timeline.bin"
    opt.write_bar_stream(path, bars)
    metadata = {"fingerprint": "b"*64}
    n_search, _ = opt.split_timeline(bars, args)
    replayed = []
    original = opt.run_backtest
    def replay(segment, *a, **kw):
        replayed.append(len(segment))
        assert len(segment) == n_search, "optimizer replay included holdout bars"
        return original(segment, *a, **kw)
    monkeypatch.setattr(opt, "run_backtest", replay)
    opt.optimize_study(args, path, metadata)
    assert replayed == [n_search]
    # Exercise the worker entry point and its real optimize_loop in-process so
    # the same replay spy can observe the worker's independently opened memmap.
    worker_args = args_for(tmp_path/"worker", n_trials=1, n_jobs=2)
    name, identity = opt.study_identity(worker_args, metadata)
    monkeypatch.setattr(opt.signal, "signal", lambda *a: None)
    opt.worker_main(worker_args, name, identity, path, n_search, os.getppid(), 1)
    assert replayed == [n_search, n_search]


def test_cache_memmap_coverage_and_resume(tmp_path):
    args = args_for(tmp_path)
    path, metadata = opt.build_cache(args, args.symbol, args.bar_type)
    bars = opt.CachedBars(path, args.bar_type)
    assert len(bars) == metadata["n_bars"] == 120
    assert metadata["observed_days"] == 5
    reader = list(open_bars(args.data_root, args.symbol, args.bar_type))
    assert list(bars) == reader
    stamp = path.stat().st_mtime_ns
    assert opt.build_cache(args, args.symbol, args.bar_type) == (path, metadata)
    assert path.stat().st_mtime_ns == stamp
    assert isinstance(bars.array, np.memmap)


def write_ticks(root, day, rows, filename="part.parquet", hourly=False):
    directory = root / f"date={day}"
    directory /= "hour=00/symbol=BTCUSDT" if hourly else "symbol=BTCUSDT"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename + ".ready" if hourly else filename)
    columns = ["trade_id", "trade_time_ms", "price", "quantity", "quote_quantity", "is_buyer_maker"]
    pq.write_table(pa.Table.from_pylist([dict(zip(columns, row)) for row in rows]), path)
    return path


def sample_ticks(day="2023-06-10"):
    t = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    return [(1, t+1000, 100., 2., 200., False), (2, t+59000, 110., 1., 110., True),
            (3, t+61000, 105., 3., 315., False), (4, t+7200000, 120., 2., 240., True)]


def test_minute_builder_ohlc_duplicates_gap_and_close_time(tmp_path):
    root = tmp_path / "ticks"
    rows = sample_ticks()
    write_ticks(root, "2023-06-10", rows)
    original = list(opt.minute_rows(root, "BTCUSDT", date(2023, 6, 10), date(2023, 6, 11), tmp_path/"cache"))
    write_ticks(root, "2023-06-10", rows, "copy.parquet", hourly=True)
    duplicate = list(opt.minute_rows(root, "BTCUSDT", date(2023, 6, 10), date(2023, 6, 11), tmp_path/"cache"))
    assert original == duplicate and len(original) == 3
    bar = original[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (100, 110, 100, 110)
    assert (bar.volume, bar.buy_volume, bar.sell_volume, bar.seq, bar.tick_count) == (3, 2, 1, 2, 2)
    assert bar.timestamp_ms == rows[0][1] - 1000 + 60000
    assert bar.dollar_value == 310
    args = args_for(tmp_path, trades_root=root, start_date=date(2023, 6, 10), end_date=date(2023, 6, 10))
    path, meta = opt.build_cache(args, "BTCUSDT", "time_1m")
    assembled = list(opt.CachedBars(path, "time_1m"))
    assert assembled[-1].data_gap and assembled[-1].duration_ms > 3600000
    assert assembled[0].duration_ms is None
    assert meta["observed_days"] == 1
    conflict = [(*rows[0][:2], 101., *rows[0][3:])]
    write_ticks(root, "2023-06-10", conflict, "conflict.parquet")
    with pytest.raises(ValueError, match="Conflicting duplicate.*2023-06-10"):
        list(opt.minute_rows(root, "BTCUSDT", date(2023, 6, 10), date(2023, 6, 11), tmp_path/"cache"))


def test_minute_month_stages_range_reuse_and_builder_invalidation(tmp_path, monkeypatch):
    root = tmp_path / "ticks"
    # Include trade near UTC month boundary: its bar is known at July 1 midnight.
    june = sample_ticks("2023-06-01") + [(i+10, row[1], *row[2:]) for i, row in enumerate(sample_ticks("2023-06-10"))]
    write_ticks(root, "2023-06-01", june[:4])
    write_ticks(root, "2023-06-10", june[4:])
    args = args_for(tmp_path, trades_root=root, start_date=date(2023, 6, 10), end_date=date(2023, 6, 14))
    opt.build_cache(args, "BTCUSDT", "time_1m")
    stages = {p: p.stat().st_mtime_ns for p in args.cache_dir.glob("minute-*.bin")}
    args.start_date, args.end_date = date(2023, 6, 1), date(2023, 6, 30)
    path, metadata = opt.build_cache(args, "BTCUSDT", "time_1m")
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stages.items())
    args.cache_dir = tmp_path / "fresh"
    fresh, fm = opt.build_cache(args, "BTCUSDT", "time_1m")
    assert path.read_bytes() == fresh.read_bytes() and metadata["fingerprint"] == fm["fingerprint"]
    monkeypatch.setattr(opt, "DATA_PROC_FP", "changed-builder-source")
    changed, cm = opt.build_cache(args, "BTCUSDT", "time_1m")
    assert cm["fingerprint"] != fm["fingerprint"]
    assert changed != fresh and len(list(args.cache_dir.glob("minute-*.bin"))) == 4
    manifests = [json.loads(p.read_text()) for p in args.cache_dir.glob("minute-*.json")]
    assert {m["data_proc_fp"] for m in manifests} == {fm["data_proc_fp"], "changed-builder-source"}


def test_month_close_exact_utc_boundary(tmp_path):
    root = tmp_path / "ticks"
    t = int(pd.Timestamp("2023-06-30T23:59:30Z").timestamp()*1000)
    write_ticks(root, "2023-06-30", [(1, t, 100., 1., 100., False)])
    write_ticks(root, "2023-07-01", [(2, t+60000, 101., 1., 101., False)])
    args = args_for(tmp_path, trades_root=root, start_date=date(2023, 7, 1), end_date=date(2023, 7, 1))
    path, _ = opt.build_cache(args, "BTCUSDT", "time_1m")
    bars = list(opt.CachedBars(path, "time_1m"))
    assert len(bars) == 2 and bars[0].timestamp_ms == t+30000


def test_mount_guard_never_creates_source(tmp_path):
    args = args_for(tmp_path, trades_root=tmp_path/"absent")
    with pytest.raises(ValueError, match="volume mounted"):
        opt.build_cache(args, "BTCUSDT", "time_1m")
    assert not args.trades_root.exists() and not args.cache_dir.exists()


def test_minute_nonpositive_volume_refused(tmp_path):
    root = tmp_path / "ticks"
    row = sample_ticks()[0]
    write_ticks(root, "2023-06-10", [(*row[:3], 0., *row[4:])])
    with pytest.raises(ValueError, match="Invalid.*volume.*2023-06-10"):
        list(opt.minute_rows(root, "BTCUSDT", date(2023,6,10), date(2023,6,11), tmp_path/"cache"))


def test_minute_duckdb_spills_under_cache_dir(tmp_path, monkeypatch):
    import duckdb
    root = tmp_path/"ticks"
    write_ticks(root, "2023-06-10", sample_ticks())
    args = args_for(tmp_path, trades_root=root, start_date=date(2023,6,10), end_date=date(2023,6,10))
    connect = duckdb.connect
    settings = []
    executed = []
    class SpyConnection:
        def __enter__(self):
            self.connection = connect()
            return self
        def __exit__(self, *exc):
            self.connection.close()
        def execute(self, query, *a, **kw):
            executed.append(query)
            result = self.connection.execute(query, *a, **kw)
            if query.startswith("SET temp_directory"):
                directory, memory = self.connection.execute(
                    "SELECT current_setting('temp_directory'), current_setting('memory_limit')").fetchone()
                assert Path(directory).is_dir()
                settings.append((Path(directory), memory))
            return result
    monkeypatch.setattr(duckdb, "connect", SpyConnection)
    opt.build_cache(args, "BTCUSDT", "time_1m")
    assert settings
    assert "SET memory_limit='512MB'" in executed
    for directory, _ in settings:
        assert directory.parent == args.cache_dir
        assert not directory.exists(), "connection spill directory was not cleaned up"


def test_artifact_path_spellings_share_all_locks(tmp_path, monkeypatch):
    argv = cli_args(tmp_path)
    monkeypatch.chdir(tmp_path)
    original = opt.file_lock
    paths = []
    def lock(path, **kw):
        paths.append(Path(path))
        return original(path, **kw)
    monkeypatch.setattr(opt, "file_lock", lock)
    locks = []
    for prefix in ("state/", str(tmp_path/"state")+"/", "./state/"):
        paths.clear()
        opt.main(argv + ["--study-db", prefix+"s.db", "--output", prefix+"r.json",
                         "--cache-dir", prefix+"cache", "--n-trials", "1"])
        assert all(path.is_absolute() for path in paths)
        locks.append(set(paths))
    assert locks[0] == locks[1] == locks[2]


def test_resume_topup_zero_parallel_and_identity(tmp_path):
    argv = cli_args(tmp_path)
    first = entry_of(opt.main(argv))
    second = entry_of(opt.main(argv + ["--n-jobs", "2"]))
    assert first["trial_counts"] == second["trial_counts"] and second["trial_counts"]["COMPLETE"] == 2
    assert first["best_params"] == second["best_params"]
    third = entry_of(opt.main(argv + ["--n-trials", "3"]))
    assert third["trial_counts"]["COMPLETE"] == 3
    changed = entry_of(opt.main(argv + ["--fee-bps", "6"]))
    assert first["study_name"] != changed["study_name"]
    assert len(json.loads((tmp_path/"r.json").read_text())["symbols"]["BTCUSDT"]["fixed_dollar_imbalance"]) == 2


def test_execution_source_and_defaults_identity(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    meta = {"fingerprint": "a"*64}
    name, identity = opt.study_identity(args, meta)
    monkeypatch.setattr(opt, "STRATEGY_FP", "changed")
    assert opt.study_identity(args, meta)[0] != name
    monkeypatch.setattr(opt, "STRATEGY_FP", identity["strategy_fp"])
    newroot = tmp_path/"copy"
    shutil.copytree(ROOT/"orderflow_system", newroot/"orderflow_system")
    monkeypatch.setattr(opt, "ROOT", newroot)
    data, strategy = opt.source_fingerprints()
    assert strategy == opt.STRATEGY_FP and data == opt.DATA_PROC_FP
    path = newroot/"orderflow_system/data/infobar_feed.py"
    path.write_bytes(path.read_bytes()+b"\n# test reader semantics change\n")
    assert opt.source_fingerprints()[0] != data and opt.source_fingerprints()[1] != strategy


def test_version_and_identity_mismatch_refused(tmp_path):
    args = args_for(tmp_path)
    name, identity = opt.study_identity(args, {"fingerprint": "a"*64})
    study = opt.open_study(args, name, identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        opt.open_study(args, name, {**identity, "seed": -1})
    versions = opt.library_versions()
    study.set_user_attr("versions", {**versions, "quantstats": "99.0.0"})
    with pytest.raises(ValueError, match="major.minor changed"):
        opt.open_study(args, name, identity)


def test_sequential_proposals_reproducible_across_resume(tmp_path):
    import optuna
    a, b = tmp_path/"a", tmp_path/"b"
    aa, bb = cli_args(a), cli_args(b)
    opt.main(aa)
    opt.main(aa + ["--n-trials", "4"])
    opt.main(bb + ["--n-trials", "4"])
    def trials(path):
        summary, = optuna.get_all_study_summaries(storage=f"sqlite:///{path}")
        return optuna.load_study(study_name=summary.study_name, storage=f"sqlite:///{path}").trials
    ta, tb = trials(a/"s.db"), trials(b/"s.db")
    assert [t.params for t in ta] == [t.params for t in tb]
    assert [t.value for t in ta] == [t.value for t in tb]


def test_interleaved_worker_seed_streams_do_not_repeat_params(tmp_path, monkeypatch):
    import optuna
    args = args_for(tmp_path, n_jobs=2)
    seeds = []
    sampler = optuna.samplers.TPESampler
    def record_sampler(*a, **kw):
        seeds.append(kw["seed"])
        return sampler(*a, **kw)
    monkeypatch.setattr(optuna.samplers, "TPESampler", record_sampler)
    name, identity = opt.study_identity(args, {"fingerprint": "a"*64})
    opt.open_study(args, name, identity, worker_index=1)
    assert seeds[-1] == args.seed + 1_000_003
    study = optuna.create_study(direction="maximize", sampler=sampler(seed=args.seed))
    # Interleaving worker 1 at len=1 and worker 0 at len=2 reproduced the old
    # equal seed. Only proposal generation is needed for this regression.
    def proposal(trial, *a):
        opt.suggest_params(trial, args.space)
        return 0.0
    monkeypatch.setattr(opt, "objective", proposal)
    seeds.clear()
    for i in range(10):
        args.worker_index, args.n_trials = i % 2, i + 1
        opt.optimize_loop(study, [], [], args)
    proposed = [tuple(sorted(t.params.items())) for t in study.trials]
    assert len(proposed) == len(set(proposed)) == 10
    assert len(set(seeds)) == 10
    assert seeds[::2] == [args.seed+i for i in range(0, 10, 2)]


def test_parallel_spawn_and_concurrent_db_initialization(tmp_path):
    argv = cli_args(tmp_path)
    result = run_cli(argv + ["--n-jobs", "2"])
    assert result.returncode == 0, result.stdout+result.stderr
    data = entry_of(json.loads((tmp_path/"r.json").read_text()))
    assert 2 <= data["trial_counts"]["COMPLETE"] <= 3
    # Two distinct studies racing schema creation in a completely absent DB.
    freshdb = str(tmp_path/"fresh.db")
    base = [sys.executable, str(SCRIPT), *argv, "--study-db", freshdb, "--n-trials", "1"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    a = subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    b = subprocess.Popen(base + ["--fee-bps", "6"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    ao, ae = a.communicate(timeout=90)
    bo, be = b.communicate(timeout=90)
    assert a.returncode == b.returncode == 0, ao+ae+bo+be


def test_worker_failure_propagates_and_failed_trials_remain(tmp_path):
    args = args_for(tmp_path, n_jobs=2)
    path, meta = opt.build_cache(args, args.symbol, args.bar_type)
    # Inject a genuine objective failure in spawned workers, after trial admission.
    args.space = {"missing_required_params": (1, 2, "int")}
    with pytest.raises(RuntimeError, match="worker failure"):
        opt.optimize_study(args, path, meta)
    name, identity = opt.study_identity(args, meta)
    study = opt.open_study(args, name, identity)
    assert len(study.trials) >= 1
    assert all(t.state.name == "FAIL" for t in study.trials)


def test_default_all_six_studies_and_time_cli(tmp_path):
    argv = cli_args(tmp_path)
    for kind in BarType:
        write_dataset(tmp_path/"source", make_rows(120, step_s=3600), bar_type=kind)
    results = opt.main(argv + ["--bar-type", "all", "--n-trials", "1"])
    entries = results["symbols"]["BTCUSDT"]
    assert set(entries) == {b.value for b in BarType}
    assert all(next(iter(studies.values()))["trial_counts"]["COMPLETE"] == 1 for studies in entries.values())
    root = tmp_path/"ticks"
    t = sample_ticks()[0][1] - 1000
    write_ticks(root, "2023-06-10", [(i+1, t+i*60000, 100.+i, 1., 100.+i, bool(i%2)) for i in range(30)])
    output = opt.main(argv + ["--bar-type", "time_1m", "--trades-root", str(root),
                             "--start-date", "2023-06-10", "--end-date", "2023-06-10", "--n-trials", "1"])
    entry = next(iter(output["symbols"]["BTCUSDT"]["time_1m"].values()))
    assert entry["dataset"]["n_bars"] == 30


def test_same_study_lock_and_orphan_worker_lock(tmp_path):
    args = args_for(tmp_path)
    path, meta = opt.build_cache(args, args.symbol, args.bar_type)
    name, _ = opt.study_identity(args, meta)
    lock = opt.study_lock_path(args.study_db, name)
    with opt.file_lock(lock):
        with pytest.raises(RuntimeError, match="Same study already running"):
            opt.optimize_study(args, path, meta)
    with opt.file_lock(str(lock)+".workers", shared=True):
        with pytest.raises(RuntimeError, match="orphaned workers still running"):
            opt.optimize_study(args, path, meta)


def test_parent_death_workers_exit_and_resume(tmp_path):
    argv = cli_args(tmp_path, n=12000) + ["--n-jobs", "2", "--n-trials", "9999"]
    # Ensure enough data survives the requested range for a long, observable trial.
    args = opt.build_parser().parse_args(argv)
    args.symbol, args.bar_type = "BTCUSDT", "fixed_dollar_imbalance"
    args.space = dict(opt.SEARCH_SPACE_BOUNDS)
    path, meta = opt.build_cache(args, args.symbol, args.bar_type)
    name, _ = opt.study_identity(args, meta)
    lock = opt.study_lock_path(args.study_db, name)
    log = (tmp_path/"parent.log").open("w")
    parent = subprocess.Popen([sys.executable, str(SCRIPT), *argv], cwd=ROOT,
                              env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+45
        busy = False
        while time.monotonic() < deadline:
            with Path(str(lock)+".workers").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy = True
            if busy:
                break
            assert parent.poll() is None
            time.sleep(.02)
        assert busy, "workers did not acquire lifetime locks"
        parent.kill()
        parent.wait(timeout=10)
        args.n_trials = 1
        # A just-spawned worker may already have noticed the dead parent. In that
        # case clean admission is safe; otherwise the lifetime lock must refuse.
        try:
            result = opt.optimize_study(args, path, meta)
        except RuntimeError as exc:
            assert "orphaned workers still running" in str(exc)
            while True:
                try:
                    with opt.file_lock(str(lock)+".workers", nonblocking=True):
                        break
                except RuntimeError:
                    assert time.monotonic() < deadline, "workers did not exit after parent death"
                    time.sleep(.05)
            result = opt.optimize_study(args, path, meta)
        assert result["trial_counts"]["COMPLETE"] >= 1
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        log.close()


def test_dsr_raw_trial_dispersion_and_gate_wiring(tmp_path, monkeypatch):
    import optuna
    from importlib import import_module
    leakage_aware_hpo = import_module("RiskLabAI.backtest.validation.leakage_aware_hpo")
    study = optuna.create_study(direction="maximize")
    def add(raw):
        study.add_trial(optuna.trial.create_trial(value=-5., user_attrs={"raw_daily_sharpe": raw}))
    add(.1)
    returns = np.array([0., .01, -.02, .03, -.01])
    assert opt.deflated_gate(study, returns)["status"] == "indeterminate"
    add(.1)
    assert opt.deflated_gate(study, returns)["passes"] is None
    add(None)
    add(.3)
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL,
                                            user_attrs={"raw_daily_sharpe": .2}))
    seen = {}
    def gate(r, n_trials, trial_sharpe_std):
        seen.update(n=n_trials, std=trial_sharpe_std)
        np.testing.assert_array_equal(r, returns)
        return {"passes": True, "deflated_sharpe": .9}
    monkeypatch.setattr(leakage_aware_hpo, "deflated_sharpe_gate", gate)
    actual = opt.deflated_gate(study, returns)
    assert actual["passes"] and seen["n"] == 4 and actual["n_dispersion_trials"] == 3
    assert seen["std"] == np.std([.1, .1, .3], ddof=1)


def test_objective_raw_sharpe_excludes_penalties(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    bars = synth_bars(30)
    folds = opt.build_cv_folds(bars, 3, .02)
    results = iter([{"fold_sharpe": -5., "sharpe_ratio": 40., "penalized": True, "n_trades": 1, "n_scored_bars": 10},
                    {"fold_sharpe": 2., "sharpe_ratio": 2., "penalized": False, "n_trades": 3, "n_scored_bars": 10},
                    {"fold_sharpe": 4., "sharpe_ratio": 4., "penalized": False, "n_trades": 3, "n_scored_bars": 10}])
    monkeypatch.setattr(opt, "evaluate_fold", lambda *a, **kw: next(results))
    trial = CaptureTrial()
    assert opt.objective(trial, bars, folds, args) == pytest.approx(1/3)
    assert trial.user_attrs["raw_daily_sharpe"] == pytest.approx(3/np.sqrt(365))


def test_holdout_all_winning_and_history_merge(tmp_path, monkeypatch):
    import optuna
    args = args_for(tmp_path)
    bars = synth_bars(100)
    n_search = 80
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=1., params=params(),
        distributions={k: optuna.distributions.IntDistribution(5,30) if k=="lookback_bars" else
                       optuna.distributions.FloatDistribution(opt.SEARCH_SPACE_BOUNDS[k][0], opt.SEARCH_SPACE_BOUNDS[k][1],
                                                             log=opt.SEARCH_SPACE_BOUNDS[k][2]=="float_log") for k in params()},
        user_attrs={"n_penalized_folds": 0, "raw_daily_sharpe": .1}))
    full = trade_table([10.,20.], [pd.Timestamp(bars[i].timestamp_ms, unit="ms", tz="UTC") for i in (85,95)])
    full["exit_seq"] = [bars[i].seq for i in (85,95)]
    full["equity_after"] = [5010.,5030.]
    monkeypatch.setattr(opt, "simulate_trades_risk_based", lambda *a, **kw: full)
    result = opt.evaluate_holdout(study, bars, n_search, args)
    assert result["profit_factor"] is None and result["profit_factor_status"] == "no_losses"
    assert result["bar_construction_independent"] is False
    assert result["deflated_sharpe_gate"]["passes"] is None
    json.dumps(result, allow_nan=False)
    entry = {"symbol": "BTCUSDT", "bar_type": args.bar_type, "study_name": "study",
             "holdout_history": {"holdout::1": result}}
    opt.save_results(args.output, entry)
    entry["holdout_history"] = {"holdout::2": result}
    opt.save_results(args.output, entry)
    merged = opt.save_results(args.output, entry)
    assert len(merged["symbols"]["BTCUSDT"][args.bar_type]["study"]["holdout_history"]) == 2


@pytest.mark.parametrize("overrides, reason", [
    ({"n_trades": 1}, "trade count"),
    ({"account_blown": True}, "account blown"),
    ({"sharpe_ratio": float("nan")}, "non-finite"),
    ({"sharpe_ratio": float("inf")}, "non-finite"),
])
def test_ineligible_holdout_gate_is_indeterminate(tmp_path, monkeypatch, overrides, reason):
    import optuna
    from importlib import import_module
    hpo = import_module("RiskLabAI.backtest.validation.leakage_aware_hpo")
    args = args_for(tmp_path)
    bars = synth_bars(100)
    study = optuna.create_study(direction="maximize")
    distributions = {k: optuna.distributions.IntDistribution(5,30) if k=="lookback_bars" else
                     optuna.distributions.FloatDistribution(lo, hi, log=kind=="float_log")
                     for k, (lo, hi, kind) in opt.SEARCH_SPACE_BOUNDS.items()}
    for raw in (.1, .3):
        study.add_trial(optuna.trial.create_trial(value=1., params=params(), distributions=distributions,
                        user_attrs={"n_penalized_folds": 0, "raw_daily_sharpe": raw}))
    trades = trade_table([10., -5.], [pd.Timestamp(bars[i].timestamp_ms, unit="ms", tz="UTC") for i in (85,95)])
    metrics = {"n_trades": 2, "sharpe_ratio": 1., "account_blown": False, **overrides}
    monkeypatch.setattr(opt, "run_backtest", lambda *a, **kw: {"ledger": None, "replay_digest": "digest"})
    monkeypatch.setattr(opt, "simulate_trades_risk_based", lambda *a, **kw: trades)
    monkeypatch.setattr(opt, "score_window", lambda *a, **kw: (metrics, trades))
    def unexpected_gate(*a, **kw):
        pytest.fail("RiskLabAI gate must not run for an ineligible holdout")
    monkeypatch.setattr(hpo, "deflated_sharpe_gate", unexpected_gate)
    gate = opt.evaluate_holdout(study, bars, 80, args)["deflated_sharpe_gate"]
    assert gate["passes"] is None and gate["status"] == "indeterminate"
    assert reason in gate["reason"]
    assert gate["n_trials"] == gate["n_dispersion_trials"] == 2
    assert gate["trial_sharpe_std"] > 0


def test_concurrent_holdout_records_and_json_history(tmp_path):
    import optuna
    argv = cli_args(tmp_path)
    first = entry_of(opt.main(argv))
    # Each invocation must leave an independent user-attr record even when no
    # candidate can pass min-trades; two concurrent attempts retry admission.
    code = '''
import sys, time
from scripts import optimize_orderflow as o
for attempt in range(100):
    try:
        o.main(sys.argv[1:])
        break
    except SystemExit as exc:
        if exc.code != 2: raise
        time.sleep(.05)
else: raise RuntimeError('admission did not free')
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    ps = [subprocess.Popen([sys.executable, "-c", code, *argv, "--evaluate-holdout"], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for _ in range(2)]
    for p in ps:
        out, err = p.communicate(timeout=90)
        assert p.returncode == 0, out+err
    study = optuna.load_study(study_name=first["study_name"], storage=f"sqlite:///{tmp_path/'s.db'}")
    history = {k:v for k,v in study.user_attrs.items() if k.startswith("holdout::")}
    assert len(history) == 2
    persisted = entry_of(json.loads((tmp_path/"r.json").read_text()))
    assert persisted["holdout_history"] == history


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(128, ["git"]), OSError("git unavailable")])
def test_git_failure_still_persists_results(tmp_path, monkeypatch, error):
    def unavailable(*a, **kw):
        raise error
    monkeypatch.setattr(subprocess, "check_output", unavailable)
    result = entry_of(opt.main(cli_args(tmp_path)))
    persisted = entry_of(json.loads((tmp_path / "r.json").read_text()))
    assert result["trial_counts"]["COMPLETE"] == 2
    assert persisted["git"] == result["git"] == {"sha": None, "dirty": None, "error": str(error)}


def test_repeated_holdout_warns_and_records_counts(tmp_path, monkeypatch, capsys):
    import optuna
    original = opt.objective
    def eligible(trial, *a):
        value = original(trial, *a)
        trial.set_user_attr("n_penalized_folds", 0)
        return value
    monkeypatch.setattr(opt, "objective", eligible)
    argv = cli_args(tmp_path) + ["--evaluate-holdout"]
    first = entry_of(opt.main(argv))
    first_output = capsys.readouterr().out
    study = optuna.load_study(study_name=first["study_name"], storage=f"sqlite:///{tmp_path/'s.db'}")
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    second = entry_of(opt.main(argv + ["--n-trials", "3"]))
    second_output = capsys.readouterr().out
    history = second["holdout_history"]
    assert len(history) == 2
    assert all(history[key] == value for key, value in first["holdout_history"].items())
    records = sorted(history.values(), key=lambda r: r["prior_evaluations"])
    assert [r["prior_evaluations"] for r in records] == [0, 1]
    assert [r["trials_at_evaluation"] for r in records] == [2, 3]
    assert all(r["status"] == "evaluated" and r["trial_number"] in (0, 1, 3) for r in records)
    assert "repeated" not in first_output.lower()
    assert "warning" in second_output.lower() and "repeated" in second_output.lower()
    assert "holdout independence" in second_output.lower()
    persisted = entry_of(json.loads((tmp_path / "r.json").read_text()))
    assert persisted["holdout_history"] == history
    stored = optuna.load_study(study_name=first["study_name"], storage=f"sqlite:///{tmp_path/'s.db'}")
    assert {key: value for key, value in stored.user_attrs.items() if key.startswith("holdout::")} == history


def test_info_study_without_duckdb_metadata(tmp_path, monkeypatch):
    from importlib import metadata
    original = metadata.version
    seen = []
    def without_duckdb(name):
        seen.append(name)
        if name == "duckdb":
            raise metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(metadata, "version", without_duckdb)
    argv = cli_args(tmp_path)
    first = entry_of(opt.main(argv))
    resumed = entry_of(opt.main(argv + ["--n-trials", "3"]))
    assert first["study_name"] == resumed["study_name"]
    assert resumed["trial_counts"]["COMPLETE"] == 3
    assert "duckdb" not in seen and "duckdb" not in resumed["versions"]
    assert {"optuna", "RiskLabAI", "quantstats", "pandas", "numpy"} <= resumed["versions"].keys()


@pytest.mark.parametrize("n_penalized, degenerate", [(3, True), (0, False)])
def test_best_degenerate_flag_and_warning(tmp_path, capsys, n_penalized, degenerate):
    import optuna
    args = args_for(tmp_path)
    study = optuna.create_study(direction="maximize")
    study.set_user_attr("versions", {})
    attrs = {"n_scored_bars": 10, "total_trades": 0, "n_penalized_folds": n_penalized}
    study.add_trial(optuna.trial.create_trial(value=-5., user_attrs=attrs))
    study.add_trial(optuna.trial.create_trial(value=-6., user_attrs={**attrs, "n_penalized_folds": 0}))
    result = opt.study_result(study, args, {}, {})
    assert result["best_degenerate"] is degenerate
    assert result["best_trial_number"] == 0 and result["best_objective"] == -5.
    output = capsys.readouterr().out.lower()
    assert ("warning" in output and "all folds penalised" in output) is degenerate
    opt.save_results(args.output, result)
    assert entry_of(json.loads(args.output.read_text()))["best_degenerate"] is degenerate


@pytest.mark.parametrize("value", ["all", "all,time_1m", "fixed_tick_run,all", "all,unknown"])
def test_bar_type_all_token_expansion(value, capsys):
    parser = opt.build_parser()
    if "unknown" in value:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--bar-type", value])
        assert exc.value.code == 2 and "unknown bar type" in capsys.readouterr().err
        return
    info = [bar.value for bar in BarType]
    expected = {"all": info, "all,time_1m": info + ["time_1m"],
                "fixed_tick_run,all": ["fixed_tick_run"] + [bar for bar in info if bar != "fixed_tick_run"]}
    actual = parser.parse_args(["--bar-type", value]).bar_type
    assert actual == expected[value] and len(actual) == len(set(actual))
