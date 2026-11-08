"""
Global settings and instrument-specific configuration.

The system runs on information-driven bars (see `data/models.py:InfoBar`).
All volume / delta thresholds that apply to a bar are therefore expressed as
scale-free ratios (delta/volume) or basis points of price, because absolute
bar volume depends on which of the six bar types is selected.

NOTE: defaults below are provisional and NOT yet calibrated against the
info-bar dataset (tracked as the next step in CLAUDE.md).
"""

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Optional

from orderflow_system.data.models import BarType


class Instrument(Enum):
    BTCUSD = "BTCUSDT"


class ProfileShape(Enum):
    P_SHAPE = "p_shape"          # Buyers in control, high volume at top, POC above 50%
    B_SHAPE = "b_shape"          # Sellers in control, high volume at bottom
    D_SHAPE = "d_shape"          # Balanced / normal distribution
    DOUBLE = "double_dist"       # Double distribution — transition day
    UNKNOWN = "unknown"


class BiasDirection(Enum):
    LONG = "long"                # Green — buyers in control
    SHORT = "short"              # Red — sellers in control
    NEUTRAL = "neutral"          # Blue — indecision / balanced
    WARNING = "warning"          # Orange — potential shift detected


@dataclass
class AbsorptionConfig:
    """Thresholds for absorption detection."""
    # ── Bar-level (works on info bars) ──
    min_delta_pct: float = 0.35               # |delta|/volume needed for "high effort"
    min_adverse_body_bps: float = 1.0         # Bar must close this far AGAINST the delta ("no reward")
    # ── Level-based (needs real trade-level footprint; inert on info bars) ──
    min_aggressive_volume: float = 20.0       # Min base-asset volume at a level
    max_price_displacement_ticks: float = 3.0 # Max ticks price can move (low result)
    rolling_window_seconds: float = 30.0      # Staleness window for level events
    min_attempts: int = 2                     # Min repeated absorption attempts
    big_trade_filter: float = 3.0             # Min volume at a level to count


@dataclass
class InitiativeConfig:
    """Thresholds for initiative auction detection."""
    min_delta_pct: float = 0.35               # |delta|/volume for "strong delta"
    volume_acceleration_min: float = 1.5      # Participation rate (vol/sec) vs rolling avg
    min_body_bps: float = 3.0                 # Minimum body (high result), basis points
    delta_price_alignment: bool = True        # Delta and price must agree


@dataclass
class SweepConfig:
    """Thresholds for book sweep detection (requires an L2 orderbook feed;
    not available from info-bar data, so the detector is not wired in)."""
    min_levels_swept: int = 3
    max_volume_per_level: float = 8.0
    max_time_ms: float = 2000.0
    thin_book_threshold: float = 5.0


@dataclass
class ExhaustionConfig:
    """Thresholds for exhaustion detection."""
    min_bars_declining: int = 5               # Bars of rising extremes with declining participation
    volume_decline_pct: float = 0.5           # Participation rate drops by 50%+ (rate is noisy; looser fires on >20% of bars)
    # Needs footprint price levels; when the data has none the requirement
    # cannot be evaluated and is skipped (never fabricated).
    requires_contrarian_imbalance: bool = False


@dataclass
class DivergenceConfig:
    """Thresholds for delta divergence detection."""
    lookback_bars: int = 10                   # Bars to look back for peaks
    min_price_new_extreme_bps: float = 1.0    # Price must make new high/low by this much
    delta_failure_pct: float = 0.8            # Cum-delta peak must fall short by (1-x) of window range


@dataclass
class VolumeProfileConfig:
    """Volume profile computation settings (one UTC-day session, BTC trades 24/7)."""
    value_area_pct: float = 0.68              # 68% of volume = value area
    lvn_stddev_factor: float = 1.5            # LVN = volume < mean - 1.5*stddev
    merge_max_days: int = 3                   # Max days to merge profiles
    tick_size: float = 10.0                   # Price bucket width (quote currency)


@dataclass
class RiskConfig:
    """Risk management / aggregation settings."""
    break_even_after_initiative: bool = True   # Move SL to BE after first initiative
    trail_on_initiative_prints: bool = True    # Trail stop on each new initiative bar
    min_rr_ratio: float = 2.0                 # Minimum reward:risk
    max_rr_ratio: float = 5.0                 # Maximum target R:R
    signal_cooldown_seconds: float = 30.0     # Min MARKET time between signals
    min_composite_score: float = 40.0         # Aggregator entry threshold
    default_stop_pct: float = 0.003           # Fallback stop distance (no usable value-area levels)
    default_target_pct: float = 0.006         # Fallback target distance (no usable value-area levels)


@dataclass
class TelegramConfig:
    """Telegram bot settings."""
    bot_token: str = ""
    chat_id: str = ""
    send_chart_snapshots: bool = True


@dataclass
class DashboardConfig:
    """Web dashboard settings."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "warning"        # uvicorn log level


@dataclass
class InfoBarConfig:
    """Information-driven bar dataset + replay settings."""
    root: str = "/Volumes/Market-Data/info-bars"  # <root>/<symbol>/<bar_type>/<UTC stamp>.parquet
    symbol: str = "BTCUSDT"
    bar_type: BarType = BarType.DOLLAR_IMBALANCE
    version: Optional[str] = None             # Dataset stamp (e.g. "20261026T135000Z"); None = newest
    start: Optional[str] = None               # Inclusive UTC date/datetime; None = from the beginning
    end: Optional[str] = None                 # Exclusive UTC date/datetime; None = to the end
    strict: bool = True                       # Raise on any malformed bar instead of skipping
    bars_per_second: float = 0.0              # Replay pacing for the live dashboard; 0 = as fast as possible
    history_bars: int = 5000                  # Bars retained in memory for charts / lookbacks
    persist: bool = False                     # Write bars / signals / profiles to SQLite while replaying


@dataclass
class InstrumentConfig:
    """Per-instrument configuration."""
    instrument: Instrument = Instrument.BTCUSD
    tick_size: float = 0.01                   # Exchange price increment (BTCUSDT)
    absorption: AbsorptionConfig = field(default_factory=AbsorptionConfig)
    initiative: InitiativeConfig = field(default_factory=InitiativeConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    exhaustion: ExhaustionConfig = field(default_factory=ExhaustionConfig)
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def get_btcusd_config() -> InstrumentConfig:
    """BTCUSDT — the instrument of the info-bar dataset."""
    return InstrumentConfig(instrument=Instrument.BTCUSD, tick_size=0.01)


def get_all_configs() -> list[InstrumentConfig]:
    """Return config for ALL instruments."""
    return [get_btcusd_config()]


# ── Info-bar data source ──
INFO_BARS = InfoBarConfig()

# Telegram config — user fills in their token/chat_id
TELEGRAM = TelegramConfig()

# ── Dashboard ──
# Web dashboard at http://localhost:8080
DASHBOARD = DashboardConfig()

# Database
DB_PATH = "orderflow_data.db"

# Logging
LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class TradeSignalConfig:
    """Untuned development defaults; every trade strategy input is opt-in."""
    absorption_levels: bool = False
    exhaustion_contrarian: str = "off"
    framing_profile_source: str = "bar_vwap"
    level_zone_frac: float = 0.25
    min_level_share: float = 0.05
    min_side_ratio: float = 0.65
    min_reject_bps: float = 2.0
    max_through_bps: float = 1.0
    contrarian_ratio: float = 2.5
    contrarian_zone_frac: float = 0.30
    contrarian_min_level_share: float = 0.02
    absorption_source: str = "bar"
    initiative_imbalance: str = "off"
    imbalance_ratio: float = 3.0
    imbalance_min_level_share: float = 0.01
    imbalance_bucket_mult: int = 1
    stacked_min_levels: int = 0

    def __post_init__(self):
        import math
        import numpy as np
        if not isinstance(self.absorption_levels, bool):
            raise ValueError('absorption_levels must be a bool')
        for name in ('exhaustion_contrarian', 'framing_profile_source', 'absorption_source', 'initiative_imbalance'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} must be a str')
        if self.exhaustion_contrarian not in ('off', 'bonus', 'required'):
            raise ValueError('invalid exhaustion_contrarian')
        if self.framing_profile_source not in ('bar_vwap', 'trades'):
            raise ValueError('invalid framing_profile_source')
        if self.absorption_source not in ('bar', 'trade') or (self.absorption_source == 'trade' and not self.absorption_levels):
            raise ValueError('trade absorption_source requires absorption_levels=True')
        if self.initiative_imbalance not in ('off', 'bonus', 'required'):
            raise ValueError('invalid initiative_imbalance')
        from numbers import Integral
        for name, allowed in (('imbalance_bucket_mult', (1, 2, 5, 10)), ('stacked_min_levels', range(5))):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in allowed:
                raise ValueError(f'invalid {name}')
            object.__setattr__(self, name, int(value))
        if self.initiative_imbalance == 'required' and not self.stacked_min_levels:
            raise ValueError('required initiative_imbalance needs stacked_min_levels >= 1')
        for name in ('level_zone_frac', 'min_level_share', 'min_side_ratio', 'min_reject_bps',
                     'max_through_bps', 'contrarian_ratio', 'contrarian_zone_frac', 'contrarian_min_level_share',
                     'imbalance_ratio', 'imbalance_min_level_share'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f'{name} must be a real number, not bool')
            value = float(value)
            object.__setattr__(self, name, value)
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('level_zone_frac', 'min_level_share', 'contrarian_zone_frac', 'contrarian_min_level_share',
                     'imbalance_min_level_share'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if not 0.5 < self.min_side_ratio <= 1:
            raise ValueError('min_side_ratio must be in (0.5, 1]')
        if self.min_reject_bps <= 0 or self.max_through_bps < 0 or self.contrarian_ratio <= 1:
            raise ValueError('invalid rejection, travel or contrarian threshold')
        if self.imbalance_ratio <= 1:
            raise ValueError('imbalance_ratio must be > 1')

    @property
    def any_enabled(self) -> bool:
        return (self.absorption_levels or self.exhaustion_contrarian != 'off'
                or self.framing_profile_source != 'bar_vwap' or self.absorption_source == 'trade'
                or self.initiative_imbalance != 'off')


def parse_trade_signals(value: str) -> TradeSignalConfig:
    """Shared CLI spelling, refusing unknown and conflicting modes."""
    names = value.split(',') if value else []
    allowed = {'absorption_levels', 'exhaustion_bonus', 'exhaustion_required', 'framing_trades'}
    if any(name not in allowed for name in names) or len(set(names)) != len(names):
        raise ValueError('invalid --trade-signals list')
    if 'exhaustion_bonus' in names and 'exhaustion_required' in names:
        raise ValueError('exhaustion_bonus and exhaustion_required are mutually exclusive')
    return TradeSignalConfig(
        absorption_levels='absorption_levels' in names,
        exhaustion_contrarian=('required' if 'exhaustion_required' in names else
                               'bonus' if 'exhaustion_bonus' in names else 'off'),
        framing_profile_source='trades' if 'framing_trades' in names else 'bar_vwap',
    )


def validate_trade_signal_range(signals, start, end, requested_range=None):
    """Phase 3 reads only the development period; end is exclusive."""
    if not signals.any_enabled:
        return
    from datetime import datetime, timezone
    def instant(value):
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).astimezone(timezone.utc)
    first = instant('2019-09-10')
    stop = instant('2024-01-01')
    if ((start is not None and not first <= instant(start) < stop) or
            (end is not None and not first < instant(end) <= stop)):
        raise ValueError('trade signals require the 2019-09-10 through 2023-12-31 development range')
    if requested_range is not None:
        lo, hi = requested_range
        if instant(lo) < first or instant(hi) >= stop:
            raise ValueError('trade signals require a development-only paired dataset')


@dataclass
class TradeFlowConfig:
    """Authenticated replay settings; legacy profile settings remain independent."""
    paired_root: str = "/Volumes/Market-Data/info-bars-paired"
    footprint_bucket: float = 1.0
    profile_bucket: float = 10.0
    duckdb_threads: int = 4
    duckdb_memory_limit: str = "1GB"
    duckdb_temp_dir: str = "output/duckdb_tmp"
    report_dir: str = "output/trade_replay"
    signals: TradeSignalConfig = field(default_factory=TradeSignalConfig)

    def __post_init__(self):
        import math
        from pathlib import Path
        for name, label in (('duckdb_temp_dir', 'DuckDB temp directory'), ('report_dir', 'trade report directory')):
            directory = Path(getattr(self, name))
            if not directory.is_absolute():
                directory = Path(__file__).resolve().parents[2]/directory
            directory = directory.resolve()
            if directory.is_relative_to('/Volumes'):
                raise ValueError(f'{label} must be local, outside /Volumes')
            setattr(self, name, str(directory))
        for width in (self.footprint_bucket, self.profile_bucket):
            scaled = width * 100
            if (not math.isfinite(scaled) or width <= 0 or round(scaled) < 1 or
                    abs(scaled-round(scaled)) >= 1e-9):
                raise ValueError('trade bucket must be positive and a multiple of $0.01')
            if round(scaled) > 2**63-1:
                raise ValueError('trade bucket exceeds int64 ticks')
