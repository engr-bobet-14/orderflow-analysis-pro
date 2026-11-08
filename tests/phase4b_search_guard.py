"""Pytest plugin for the Phase 4b real-data proof (amendment A1). Load with ``-p tests.phase4b_search_guard``.

Every info-bar reader must be bounded and end no later than 2023-04-29T00:00Z, before the search horizon
H = 2023-04-29T03:32:55.751Z. Bounded trade-cache builds must end by then as well. The checks run before
any read, so a refused read touches no bytes. Not intercepted (disclosed in amendment A1 §7): the production
cache's integrity decoding (#6), PairedDataset.open's artifact hashing and hours index, trade queries inside an
admitted bounded build, and whole row groups that straddle an admitted end.
"""
from datetime import datetime, timezone

import pytest

from scripts import phase4b_proof as proof

LIMIT_MS = int(datetime.fromisoformat(proof.LAST_SAFE_EXCLUSIVE_END + 'T00:00:00+00:00').timestamp() * 1000)


def _refuse(what, value):
    raise AssertionError(f'{what} reads past the Phase 4b search horizon: {value!r}')


def install(monkeypatch):
    import sys
    import orderflow_system.data.infobar_feed as feed
    from scripts import trade_feature_cache as tfc
    original = feed.InfoBarReader

    class GuardedReader(original):
        def __init__(self, info, start_ms=None, end_ms=None, strict=True):
            if end_ms is None or end_ms > LIMIT_MS:
                _refuse('InfoBarReader', end_ms)
            super().__init__(info, start_ms, end_ms, strict)

    for module in list(sys.modules.values()):
        if getattr(module, 'InfoBarReader', None) is original:
            monkeypatch.setattr(module, 'InfoBarReader', GuardedReader)
    build = tfc.build

    def guarded_build(ds, destination, *, start=None, end=None, **kwargs):
        if end is None or str(end)[:10] > proof.LAST_SAFE_EXCLUSIVE_END:
            _refuse('trade cache build', end)
        return build(ds, destination, start=start, end=end, **kwargs)
    monkeypatch.setattr(tfc, 'build', guarded_build)


@pytest.fixture(autouse=True)
def phase4b_search_only(monkeypatch):
    install(monkeypatch)
