"""Shared fixtures: build tiny synthetic info-bar parquet datasets in the real on-disk layout."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orderflow_system.data.models import BarType

STAMP = "20260101T000000Z"


def make_rows(n=50, t0="2023-01-01T00:00:00", step_s=30, price0=20000.0, seed_delta=0.3):
    """Deterministic synthetic bars: random-walk-ish prices, alternating pressure."""
    t0_ms = int(datetime.fromisoformat(t0).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows, price, seq = [], price0, 0
    for i in range(n):
        o = price
        c = o + ((-1) ** (i // 3)) * (2.0 + (i % 5))
        h, l = max(o, c) + 1.0, min(o, c) - 1.0
        vol = 10.0 + (i % 7)
        buy = vol * (0.5 + seed_delta * ((-1) ** (i // 3)) * 0.5)
        nt = 100 + i
        seq += nt
        rows.append(dict(
            ts=t0_ms + i * step_s * 1000, seq=seq, o=o, h=h, l=l, c=c,
            vol=vol, buy=buy, sell=vol - buy, nt=nt, dv=vol * (h + l) / 2, th=1000.0,
        ))
        price = c
    return rows


def write_dataset(root: Path, rows, bar_type=BarType.DOLLAR_IMBALANCE, symbol="BTCUSDT",
                  stamp=STAMP, missing_hours=(), row_group_size=20):
    d = root / symbol / bar_type.value
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "Date Time": pa.array([r["ts"] * 1000 for r in rows], type=pa.timestamp("us", tz="UTC")),
        "Tick Number": pa.array([r["seq"] for r in rows], type=pa.int64()),
        "Open": [r["o"] for r in rows], "High": [r["h"] for r in rows],
        "Low": [r["l"] for r in rows], "Close": [r["c"] for r in rows],
        "Cumulative Volume": [r["vol"] for r in rows],
        "Cumulative Buy Volume": [r["buy"] for r in rows],
        "Cumulative Sell Volume": [r["sell"] for r in rows],
        "Cumulative Ticks": pa.array([r["nt"] for r in rows], type=pa.int64()),
        "Cumulative Dollar Value": [r["dv"] for r in rows],
        "Threshold": [r["th"] for r in rows],
    })
    pq.write_table(table, d / f"{stamp}.parquet", row_group_size=row_group_size)
    (d / f"{stamp}.json").write_text(json.dumps({
        "symbol": symbol, "config": {"bar_type": bar_type.value},
        "missing_tick_hours": list(missing_hours), "n_bars": len(rows),
    }))
    return d


@pytest.fixture
def dataset_root(tmp_path):
    write_dataset(tmp_path, make_rows(100))
    return tmp_path
