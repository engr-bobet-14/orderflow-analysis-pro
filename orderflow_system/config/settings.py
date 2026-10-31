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
