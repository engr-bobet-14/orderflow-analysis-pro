"""Verify a paired trades/Info-Bars dataset end to end (read-only) and write its alignment report.

1. Integrity: root hash vs the pinned value, every artifact's size + SHA-256, no unlisted/missing file.
2. Full-history alignment over every bar type (orderflow_system/data/trade_alignment.py): source-file
   SHA-256, per-hour digests, every bar rebuilt from its trades, tails. Fails fast on the first mismatch.

The pinned root comes from orderflow_system/config/paired_datasets.json. Before a dataset is registered,
pass the root hash printed by the builder with --expected-root (the report records which was used).

    python scripts/verify_paired_dataset.py --paired-root <root> --dataset-id <id> [--expected-root <sha256>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orderflow_system.data.paired_dataset import PairedDataset, pinned_root  # noqa: E402
from orderflow_system.data.trade_alignment import verify_alignment  # noqa: E402

CERT_DIR = ROOT / "output" / "certificates"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paired-root", type=Path, required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--expected-root", help="root hash to pin when the dataset is not registered yet")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, help="report path (default: output/certificates/alignment-<id>.json)")
    args = ap.parse_args(argv)
    pin = args.expected_root or pinned_root(args.dataset_id)
    t0 = time.time()
    ds = PairedDataset.open(args.paired_root / args.dataset_id, pin)
    report = verify_alignment(ds, threads=args.threads)
    report["pin_source"] = "--expected-root" if args.expected_root else "registry"
    report["seconds"] = round(time.time() - t0, 1)
    out = args.out or CERT_DIR / f"alignment-{ds.dataset_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rss"}, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
