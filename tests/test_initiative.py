"""Participation baselines include known rates and retain them across outages."""

from orderflow_system.config.settings import get_btcusd_config
from orderflow_system.data.models import BarType, InfoBar, SignalType
from orderflow_system.pipeline import InfoBarPipeline


def test_post_gap_initiative_compares_against_prior_known_rates_only():
    pipe = InfoBarPipeline(get_btcusd_config(), BarType.DOLLAR_IMBALANCE)

    def bar(seq, volume=1.0, duration_ms=1000, gap=False):
        return InfoBar(timestamp_ms=1672531200000 + seq * 1000, seq=seq,
                       open=100, high=101.1, low=99.9, close=101, volume=volume,
                       buy_volume=volume * .9, sell_volume=volume * .1,
                       dollar_value=volume * 100.5, duration_ms=duration_ms, data_gap=gap)

    pipe.process_bar(bar(0, volume=1e6, duration_ms=None))
    assert pipe.initiative._avg_rate_window == []
    for seq in range(1, 52):
        pipe.process_bar(bar(seq))
    assert pipe.initiative._avg_rate_window == [1.0] * 50
    for seq in range(52, 55):
        result = pipe.process_bar(bar(seq, volume=1e9, gap=True))
        assert not result.signals
        assert pipe.initiative._avg_rate_window == [1.0] * 50
    result = pipe.process_bar(bar(55, volume=4))
    initiative = next(signal for signal in result.signals if signal.signal_type == SignalType.INITIATIVE)
    assert initiative.details["avg_rate"] == 1.0
    assert initiative.details["vol_acceleration"] == 4.0
    assert pipe.initiative._avg_rate_window == [1.0] * 49 + [4.0]
