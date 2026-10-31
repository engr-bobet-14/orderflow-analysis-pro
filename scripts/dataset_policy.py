"""Dataset-policy boundaries from CLAUDE.md. Pure functions, no I/O.

2019-2023 development/HPO/CV, 2024-2025 out-of-sample validation only, 2026 final test.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

DEV_START = date(2019, 1, 1)
DEV_END = date(2023, 12, 31)
VALIDATION_START = date(2024, 1, 1)
VALIDATION_END = date(2025, 12, 31)
FINAL_TEST_START = date(2026, 1, 1)


def ensure_dev_window(start, end):
    if start > end or start < DEV_START or end > DEV_END:
        raise ValueError(f"analysis window must lie within {DEV_START}..{DEV_END} (dataset policy)")


def ensure_validation_window(start, end):
    if start > end or start < VALIDATION_START or end > VALIDATION_END:
        raise ValueError(f"validation window must lie within {VALIDATION_START}..{VALIDATION_END}")


def _utc_ms(day):
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def window_slice(timestamps_ms, start, end):
    """(first, last_exclusive) of bars with start 00:00Z <= close time < (end + 1 day) 00:00Z."""
    low, high = _utc_ms(start), _utc_ms(end + timedelta(days=1))
    stamps = np.asarray(timestamps_ms)
    return int(np.searchsorted(stamps, low, side="left")), int(np.searchsorted(stamps, high, side="left"))
