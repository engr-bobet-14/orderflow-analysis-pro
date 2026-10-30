"""Merged profile framing respects the instrument's configured area and window."""

from orderflow_system.analytics.volume_profile import VolumeProfileEngine
from orderflow_system.config.settings import VolumeProfileConfig, get_btcusd_config
from orderflow_system.data.models import VolumeProfileResult
from orderflow_system.pipeline import InfoBarPipeline
from orderflow_system.signals.profile_framing import ProfileFramingEngine, LevelType


def test_merged_levels_use_configured_area_and_two_profile_window():
    config = VolumeProfileConfig(value_area_pct=0.9, merge_max_days=2)
    framing = ProfileFramingEngine(config)
    profiles = [VolumeProfileResult(session_date=f"2023-01-0{i + 1}", poc=100, val=90, vah=110,
                                   volume_at_price=volume, total_volume=sum(volume.values()))
                for i, volume in enumerate(({90: 10000, 100: 1, 110: 1},
                                            {90: 2, 100: 5, 110: 2},
                                            {90: 2, 100: 5, 110: 2}))]
    for profile in profiles:
        framing.add_profile(profile)
    bias = framing.analyze(100)
    expected = VolumeProfileEngine(config).merge_profiles(profiles[-2:])
    assert (bias.merged_val, bias.merged_vah) == (expected.val, expected.vah)
    assert (expected.val, expected.vah) != tuple(
        getattr(VolumeProfileEngine(VolumeProfileConfig()).merge_profiles(profiles[-2:]), key)
        for key in ("val", "vah"))
    merged_levels = [level for level in bias.qualified_levels
                     if level.level_type in (LevelType.MERGED_VAL, LevelType.MERGED_VAH)]
    assert len(merged_levels) == 2
    assert all(set(level.source_dates) == {"2023-01-02", "2023-01-03"} for level in merged_levels)
    instrument = get_btcusd_config()
    instrument.volume_profile = config
    assert InfoBarPipeline(instrument).profile_framing.config is config
