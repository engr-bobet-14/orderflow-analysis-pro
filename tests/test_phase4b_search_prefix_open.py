"""SearchPrefix.open with the production cache options used by S1 and the AC1 proof (found by the first AC1 run)."""
from types import SimpleNamespace

import pytest

from scripts import phase4_studies as p4


class FakeBars(list):
    pass


@pytest.fixture
def captured(monkeypatch):
    from scripts import optimize_orderflow as opt, trade_feature_cache as tfc
    seen = {}

    def bars(path, bar_type):
        seen['bars'] = (path, bar_type)
        return FakeBars(range(10))

    def updates(path, bars, **options):
        seen['updates'] = (path, options)
        return list(range(len(bars)))
    monkeypatch.setattr(opt, 'CachedBars', bars)
    monkeypatch.setattr(tfc, 'CachedTradeUpdates', updates)
    monkeypatch.setattr(opt, 'split_timeline', lambda bars, args: (8, []))
    return seen


def test_production_call_signature_reaches_the_cache_identity_check(captured):
    # Exactly the call made by phase4b_campaign (S1) and phase4b_regression._prefix (AC1).
    prefix = p4.SearchPrefix.open('bars.bin', 'fixed_tick_imbalance', 'lane', SimpleNamespace(),
                                  production=True, dataset_id='ds', bar_type='fixed_tick_imbalance')
    assert captured['bars'] == ('bars.bin', 'fixed_tick_imbalance')
    assert captured['updates'][1] == {'production': True, 'dataset_id': 'ds', 'bar_type': 'fixed_tick_imbalance'}
    assert len(prefix.bars) == 8 and len(prefix.trade_updates) == 8


def test_regression_prefix_opens_the_production_lane(captured, monkeypatch, tmp_path):
    from scripts import phase4b_regression as regression, trade_feature_cache as tfc
    monkeypatch.setattr(tfc, 'generation_directory', lambda root, ds: tmp_path)
    monkeypatch.setattr(tfc, 'production_generation', lambda root: {'lanes': {'fixed_volume_imbalance': str(tmp_path)}})
    prefix = regression._prefix('fixed_volume_imbalance')
    assert captured['updates'][1]['bar_type'] == 'fixed_volume_imbalance'
    assert captured['updates'][1]['production'] is True
    assert len(prefix.bars) == 8
