"""Amendment A1: search-only replacements for real-data tests that read Phase 4b pseudo-holdout dates.

Windows end before H (scripts/phase4b_proof.py). The proof command loads tests/phase4b_search_guard.py, which
refuses any unbounded or late bar reader and bounded trade build before it reads. Gaps G1-G4: amendment A1.
"""
from pathlib import Path

import pytest

from orderflow_system.config.settings import TradeFlowConfig
from orderflow_system.data.models import BarType

INFO_ROOT = '/Volumes/Market-Data/info-bars'
DATASET = '20261103T011635Z-d3a7b9603b37'
pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not Path(INFO_ROOT, 'BTCUSDT').is_dir(), reason='info-bar dataset not mounted')]


@pytest.fixture(autouse=True)
def only_under_the_search_guard(request):
    if not request.config.pluginmanager.hasplugin('tests.phase4b_search_guard'):
        pytest.skip('A1 replacements run only via scripts/phase4b_proof.real_data_selection() (guard plugin)')


needs_paired = pytest.mark.skipif(not (Path(TradeFlowConfig().paired_root) / DATASET).is_dir(),
                                  reason='certified paired dataset not mounted')


# A1-1 (replaces test_real_dataset.py::test_pipeline_on_real_bars, 2023-11-19..24). validate() also runs
# check_start_independence, which ignores the window and reads unbounded up to the dataset's first outage
# (after H). So the remaining validate() checks are composed here from its own replay helper. Gap G1:
# start independence across a real outage is not exercised (no outage exists before H in any bar type).
@pytest.mark.parametrize('bar_type', list(BarType), ids=lambda b: b.value)
def test_a1_1_pipeline_on_real_search_bars(bar_type):
    from orderflow_system.validate_infobars import _replay
    start, end = '2023-03-19', '2023-03-24'
    reader, pipe, digest, price_levels, _, _ = _replay(INFO_ROOT, 'BTCUSDT', bar_type, start, end)
    n = pipe.bars_processed
    k = max(1, int(n * 0.4))
    _, _, digest2, _, _, prefix_in_full = _replay(INFO_ROOT, 'BTCUSDT', bar_type, start, end, snapshot_at=k)
    _, _, d_prefix, _, _, _ = _replay(INFO_ROOT, 'BTCUSDT', bar_type, start, end, limit=k)
    rep = reader.report
    checks = {'trade_ledger_valid': pipe.ledger_audit.report()['ok'],
              'pipeline_completed': n > 0 and n == rep.rows_yielded,
              'no_rows_rejected': rep.rows_rejected == 0,
              'deterministic_replay': digest.hexdigest() == digest2.hexdigest(),
              'no_lookahead_prefix_invariant': d_prefix.hexdigest() == prefix_in_full,
              'no_price_level_data_fabricated': price_levels == 0}
    assert n > 500
    assert all(checks.values()), checks


# A1-2 (replaces the tick-run distant-absorption regression, 2023-11-19..24). Gap G2: the reviewed bar
# seq 4304211644 is not exercised; the proximity invariant is.
def test_a1_2_tick_run_entries_stay_within_proximity():
    from orderflow_system.config.settings import get_btcusd_config
    from orderflow_system.data.infobar_feed import open_bars
    from orderflow_system.ledger import TradeLedger
    from orderflow_system.pipeline import InfoBarPipeline
    pipe = InfoBarPipeline(get_btcusd_config(), BarType.TICK_RUN)
    ledger = TradeLedger()
    for bar in open_bars(INFO_ROOT, 'BTCUSDT', BarType.TICK_RUN, start='2023-03-19', end='2023-03-24'):
        ledger.process_result(pipe.process_bar(bar))
    assert ledger.entries, 'the replay must exercise entry proximity'
    for record in ledger.entries:
        execution = record.execution
        assert abs(execution.entry_price - execution.qualified_level.price) / execution.entry_price < .002


# A1-3 (replaces test_optimize_orderflow.py::test_real_prefix_and_notebook_parity, 2023-06-01..04).
def test_a1_3_real_prefix_and_notebook_parity(tmp_path):
    from orderflow_system.config.settings import INFO_BARS
    from orderflow_system.data.infobar_feed import open_bars
    from tests.test_optimize_orderflow import args_for, assert_prefix_parity
    bars = list(open_bars(INFO_BARS.root, 'BTCUSDT', BarType.DOLLAR_IMBALANCE, '2023-03-01', '2023-03-04'))
    assert_prefix_parity(bars, args_for(tmp_path))


# A1-4 (replaces test_phase4_real.py::test_bounded_real_cache_and_live[2023-10-16-2023-10-17]): the same
# assertions, called unchanged, on the latest whole search day.
@needs_paired
@pytest.mark.parametrize('start,end', [('2023-04-17', '2023-04-18')])
def test_a1_4_bounded_real_cache_and_live(start, end):
    from tests.test_phase4_real import test_bounded_real_cache_and_live as bounded
    bounded(start, end)
