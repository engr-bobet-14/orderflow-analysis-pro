"""
Paired trades/Info-Bars dataset (built by information-driven-bars `run_info_bars.py --final --paired`).

Layout of one published dataset `<paired root>/<stamp>-<root hash[:12]>/`:
    <SYMBOL>/<bar_type>/<stamp>.parquet + .json   bars + sidecars (the InfoBarReader layout)
    trade_manifest.json                           stream contract, source files + SHA-256, provenance
    hours.parquet                                 per UTC hour: first_ordinal, n_trades, sum_qty_lots, digest
    dataset_manifest.json                         size + SHA-256 of every artifact above

The root hash (SHA-256 of dataset_manifest.json's bytes) is pinned OUTSIDE the dataset in the tracked
registry `config/paired_datasets.json`, added by a reviewed commit. `open_registered` refuses an
unregistered id, a root mismatch, any artifact whose size/hash differs, and any unlisted or missing
file, all before a single bar is read. Nothing is ever repaired.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orderflow_system.data.infobar_feed import DataIntegrityError

FORMAT = "paired-info-bars/1"
DATASET_MANIFEST = "dataset_manifest.json"
TRADE_MANIFEST = "trade_manifest.json"
HOURS_FILE = "hours.parquet"
PRICE_SCALE = 100
QTY_SCALE = 1000
# Canonical row hashed per hour; must equal the builder's src/paired.py ROW_DTYPE byte for byte.
ROW_DTYPE = np.dtype([("time_ms", "<i8"), ("trade_id", "<i8"), ("price_ticks", "<i8"),
                      ("qty_lots", "<i8"), ("is_buyer_maker", "u1")])
ROW_FORMAT = {"fields": [[n, ROW_DTYPE.fields[n][0].str] for n in ROW_DTYPE.names],
              "price_scale": PRICE_SCALE, "qty_scale": QTY_SCALE, "byte_order": "little", "packed": True}
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "paired_datasets.json"


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def pinned_root(dataset_id: str, registry_path: Path = REGISTRY_PATH) -> str:
    registry = json.loads(Path(registry_path).read_text())
    if dataset_id not in registry:
        raise DataIntegrityError(f"paired dataset {dataset_id!r} is not registered in {registry_path}")
    return registry[dataset_id]


@dataclass
class PairedDataset:
    directory: Path
    root_hash: str
    manifest: dict                       # trade_manifest.json
    hours: pd.DataFrame
    _stats: dict = field(default_factory=dict, repr=False)

    @property
    def dataset_id(self) -> str:
        return self.directory.name

    @property
    def bars_root(self) -> Path:
        """Root for find_dataset/open_bars: `<bars_root>/<SYMBOL>/<bar_type>/<stamp>.parquet`."""
        return self.directory

    @property
    def bar_types(self) -> list[str]:
        return sorted(b["bar_type"] for b in self.manifest["bars"])

    def bar_path(self, bar_type: str) -> Path:
        (entry,) = [b for b in self.manifest["bars"] if b["bar_type"] == bar_type]
        return self.directory / entry["parquet"]

    @classmethod
    def open(cls, directory: Path, expected_root: str) -> "PairedDataset":
        directory = Path(directory)
        data = (directory / DATASET_MANIFEST).read_bytes()
        root = hashlib.sha256(data).hexdigest()
        if root != expected_root:
            raise DataIntegrityError(f"{directory.name}: root hash {root} != pinned {expected_root}")
        manifest = json.loads(data)
        if manifest.get("format") != FORMAT:
            raise DataIntegrityError(f"{directory.name}: unknown dataset format {manifest.get('format')!r}")
        listed = {a["path"]: a for a in manifest["artifacts"]}
        present = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                   if p.is_file() and p.name != DATASET_MANIFEST}
        if present - set(listed):
            raise DataIntegrityError(f"{directory.name}: unlisted file(s) {sorted(present - set(listed))}")
        stats = {}
        for rel, a in listed.items():
            p = directory / rel
            if not p.is_file():
                raise DataIntegrityError(f"{directory.name}: listed artifact missing: {rel}")
            st = p.stat()
            if st.st_size != a["size"] or sha256_file(p) != a["sha256"]:
                raise DataIntegrityError(f"{directory.name}: {rel} does not match the dataset manifest")
            stats[rel] = (st.st_size, st.st_mtime_ns)
        trade_manifest = json.loads((directory / TRADE_MANIFEST).read_bytes())
        if trade_manifest.get("row_format") != ROW_FORMAT:
            raise DataIntegrityError(f"{directory.name}: unsupported row format {trade_manifest.get('row_format')}")
        hours = pq.read_table(directory / HOURS_FILE).to_pandas()
        return cls(directory, root, trade_manifest, hours, stats)

    def recheck(self) -> None:
        """End-of-run guard: every verified artifact still has the size/mtime seen at open."""
        for rel, seen in self._stats.items():
            st = (self.directory / rel).stat()
            if (st.st_size, st.st_mtime_ns) != seen:
                raise DataIntegrityError(f"{self.dataset_id}: {rel} changed after verification")


def open_registered(paired_root: Path, dataset_id: str, registry_path: Path = REGISTRY_PATH) -> PairedDataset:
    return PairedDataset.open(Path(paired_root) / dataset_id, pinned_root(dataset_id, registry_path))
