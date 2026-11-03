"""Tiny paired trades/Info-Bars dataset in the builder's `--paired` on-disk format (src/paired.py).

Trades are written as daily source partitions; bars are cut at chosen ordinals and filled exactly as
RiskLabAI does (tick rule with initial rule 0, sell = volume - buy, dollar = sum(price * qty)), so the
alignment verifier can be tested against a stream whose truth is known.
"""
import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from orderflow_system.data import paired_dataset as pdx

SYMBOL = "BTCUSDT"
STAMP = "20261103T000000Z"
DAY0 = date(2023, 1, 1)


def day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def make_trades(days=3, per_day=600, seed=7, gap_hour=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames, next_id = [], 1
    for k in range(days):
        lo = day_ms(DAY0 + timedelta(days=k))
        t = np.sort(rng.integers(lo, lo + 86_400_000, per_day))
        if gap_hour is not None and k == 1:
            hour = lo + gap_hour * 3_600_000
            t = t[(t < hour) | (t >= hour + 3_600_000)]
        t[3] = t[2]                                   # same-ms tie, ordered by trade_id
        n = len(t)
        frames.append(pd.DataFrame({
            "trade_id": np.arange(next_id, next_id + n), "trade_time_ms": t,
            "price": np.round(20000 + np.cumsum(rng.choice([-0.5, 0.0, 0.5], n)), 2),
            "quantity": np.round(rng.integers(1, 2000, n) / 1000, 3),
            "is_buyer_maker": rng.random(n) < 0.5,
        }))
        next_id += n
    return pd.concat(frames, ignore_index=True)


def tick_rule(price: np.ndarray) -> np.ndarray:
    diff = np.diff(price, prepend=price[0])
    return pd.Series(np.sign(diff)).replace(0.0, np.nan).ffill().fillna(0.0).to_numpy()


def bar_table(trades: pd.DataFrame, ends: list[int]) -> pa.Table:
    """Bars ending at 1-based ordinals `ends` (Tick Number), RiskLabAI field semantics."""
    p, q = trades.price.to_numpy(), trades.quantity.to_numpy()
    rule = tick_rule(p)
    rows, prev = [], 0
    for e in ends:
        s = slice(prev, e)
        vol, buy = float(np.sum(q[s])), float(np.sum(np.where(rule[s] == 1, q[s], 0.0)))
        last = min(e, len(p)) - 1                       # an end past the stream (corrupt-bar tests) uses the last trade
        rows.append({"Date Time": int(trades.trade_time_ms.iloc[last]) * 1000, "Tick Number": e,
                     "Open": p[prev], "High": p[s].max(), "Low": p[s].min(), "Close": p[last],
                     "Cumulative Volume": vol, "Cumulative Buy Volume": buy,
                     "Cumulative Sell Volume": vol - buy, "Cumulative Ticks": e - prev,
                     "Cumulative Dollar Value": float(np.sum(p[s] * q[s])), "Threshold": 1.0})
        prev = e
    df = pd.DataFrame(rows)
    return pa.table({
        "Date Time": pa.array(df["Date Time"], type=pa.timestamp("us", tz="UTC")),
        "Tick Number": pa.array(df["Tick Number"], type=pa.int64()),
        **{c: pa.array(df[c], type=pa.float64()) for c in ("Open", "High", "Low", "Close", "Cumulative Volume",
                                                            "Cumulative Buy Volume", "Cumulative Sell Volume")},
        "Cumulative Ticks": pa.array(df["Cumulative Ticks"], type=pa.int64()),
        "Cumulative Dollar Value": pa.array(df["Cumulative Dollar Value"], type=pa.float64()),
        "Threshold": pa.array(df["Threshold"], type=pa.float64()),
    })


def canonical(trades: pd.DataFrame) -> np.ndarray:
    rows = np.empty(len(trades), dtype=pdx.ROW_DTYPE)
    rows["time_ms"] = trades.trade_time_ms
    rows["trade_id"] = trades.trade_id
    rows["price_ticks"] = np.rint(trades.price * pdx.PRICE_SCALE)
    rows["qty_lots"] = np.rint(trades.quantity * pdx.QTY_SCALE)
    rows["is_buyer_maker"] = trades.is_buyer_maker
    return rows


def hours_frame(trades: pd.DataFrame, first: date, last: date) -> pd.DataFrame:
    rows = canonical(trades)
    grid = np.arange(day_ms(first), day_ms(last + timedelta(days=1)), 3_600_000, dtype=np.int64)
    hour = rows["time_ms"] // 3_600_000 * 3_600_000
    n, lots, digest = [], [], []
    for h in grid:
        part = rows[hour == h]
        n.append(len(part))
        lots.append(int(part["qty_lots"].sum()))
        digest.append(hashlib.sha256(part.tobytes()).hexdigest())
    n = np.array(n, dtype=np.int64)
    return pd.DataFrame({"hour_ms": grid, "first_ordinal": np.cumsum(n) - n + 1, "n_trades": n,
                         "sum_qty_lots": np.array(lots, dtype=np.int64), "digest": digest})


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_paired(root: Path, trades: pd.DataFrame, ends_by_type: dict[str, list[int]],
                 days: int = 3, missing_hours=()) -> tuple[Path, str]:
    """Write sources + a published paired dataset under `root`; returns (dataset_dir, root_hash)."""
    source = root / "futures_trades"
    stamps = pd.to_datetime(trades.trade_time_ms, unit="ms", utc=True)
    records = []
    for d, part in trades.groupby(stamps.dt.date):
        folder = source / f"date={d.isoformat()}" / f"symbol={SYMBOL}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "part-0.parquet"
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), path)
        st = path.stat()
        records.append({"path": path.relative_to(source).as_posix(), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                        "sha256": sha(path), "row_count": len(part), "min_time": int(part.trade_time_ms.min()),
                        "max_time": int(part.trade_time_ms.max())})
    last = DAY0 + timedelta(days=days - 1)
    staging = root / "paired" / ".staging"
    bars = []
    for bt, ends in ends_by_type.items():
        d = staging / SYMBOL / bt
        d.mkdir(parents=True)
        pq.write_table(bar_table(trades, ends), d / f"{STAMP}.parquet", row_group_size=7)
        (d / f"{STAMP}.json").write_text(json.dumps({
            "symbol": SYMBOL, "config": {"bar_type": bt}, "n_ticks": len(trades),
            "missing_tick_hours": list(missing_hours), "requested_range": [DAY0.isoformat(), last.isoformat()]}))
        bars.append({"bar_type": bt, "parquet": f"{SYMBOL}/{bt}/{STAMP}.parquet", "config": {"bar_type": bt}})
    manifest = {"format": pdx.FORMAT, "symbol": SYMBOL, "stamp": STAMP, "n_trades": len(trades),
                "requested_range": [DAY0.isoformat(), last.isoformat()], "row_format": pdx.ROW_FORMAT,
                "source_root": str(source), "files": records, "bars": bars,
                "builder": {"git_sha": "f" * 40, "git_dirty": False}}
    (staging / pdx.TRADE_MANIFEST).write_bytes(canonical_json(manifest))
    pq.write_table(pa.Table.from_pandas(hours_frame(trades, DAY0, last), preserve_index=False),
                   staging / pdx.HOURS_FILE)
    artifacts = [{"path": p.relative_to(staging).as_posix(), "size": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
    data = canonical_json({"format": pdx.FORMAT, "artifacts": artifacts})
    (staging / pdx.DATASET_MANIFEST).write_bytes(data)
    root_hash = hashlib.sha256(data).hexdigest()
    final = root / "paired" / f"{STAMP}-{root_hash[:12]}"
    os.rename(staging, final)
    return final, root_hash


def rewrite_artifact(dataset: Path, rel: str, new_bytes: bytes, *, consistent: bool) -> str:
    """Replace an artifact; with consistent=True also rewrite the manifest so its hashes match (new root)."""
    (dataset / rel).write_bytes(new_bytes)
    if not consistent:
        return ""
    manifest = json.loads((dataset / pdx.DATASET_MANIFEST).read_bytes())
    for a in manifest["artifacts"]:
        if a["path"] == rel:
            a["size"], a["sha256"] = len(new_bytes), hashlib.sha256(new_bytes).hexdigest()
    data = canonical_json(manifest)
    (dataset / pdx.DATASET_MANIFEST).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
