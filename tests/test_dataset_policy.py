"""Dataset-policy boundaries (CLAUDE.md): 2019-2023 dev, 2024-2025 validation, 2026 final test."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import dataset_policy as policy


def ms(text):
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def test_dev_window_guard():
    policy.ensure_dev_window(date(2019, 9, 10), date(2023, 12, 31))
    for start, end in [(date(2019, 9, 10), date(2024, 1, 1)),
                       (date(2018, 12, 31), date(2023, 12, 31)),
                       (date(2023, 6, 1), date(2023, 5, 1))]:
        with pytest.raises(ValueError, match="dataset policy"):
            policy.ensure_dev_window(start, end)


@pytest.mark.parametrize("start, end", [
    (date(2023, 12, 31), date(2024, 6, 1)),
    (date(2024, 1, 1), date(2026, 1, 1)),
    (date(2026, 1, 1), date(2026, 6, 1)),
])
def test_validation_window_guard_refuses_outside_policy(start, end):
    with pytest.raises(ValueError, match="validation window must lie within 2024-01-01..2025-12-31"):
        policy.ensure_validation_window(start, end)


def test_validation_window_guard_accepts_full_policy_window():
    policy.ensure_validation_window(date(2024, 1, 1), date(2025, 12, 31))


def test_window_slice_boundaries():
    stamps = np.array([ms("2023-12-31T23:59:59.999"), ms("2024-01-01T00:00:00.000"),
                       ms("2025-12-31T23:59:59.999"), ms("2026-01-01T00:00:00.000")])
    assert policy.window_slice(stamps, date(2024, 1, 1), date(2025, 12, 31)) == (1, 3)
    assert policy.window_slice(stamps, date(2019, 9, 10), date(2023, 12, 31)) == (0, 1)
