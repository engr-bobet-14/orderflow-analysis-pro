# Optuna search-space findings (2019-09-10 .. 2023-12-31, BTCUSDT)

Studies: `output/optimize/main/studies.db`. Costs: fee 5 bps + slippage 1 bps per side. Floors: 105 total trades, 11 per fold. Search segment only (first 80% of bars); the pseudo-holdout and 2024+ were never read.

## 1. Replay of the best trials — NO GROSS EDGE in either study

`scripts/diagnose_trades.py` replayed the three best feasible trials of each main study (stored replay digest matched on all six; fee 5.0, slippage 1.0, 5 splits). Cost-limited would require `gross_expectancy_r > 2 × se_r`; none qualifies.

| study | trial | gross_E_r | 2*se_r | verdict |
|---|---|---|---|---|
| tick | 197 | 0.0791 | 0.2419 | no gross edge |
| tick | 128 | 0.0706 | 0.2442 | no gross edge |
| tick | 191 | 0.0752 | 0.2616 | no gross edge |
| volume | 193 | 0.0902 | 0.1140 | no gross edge |
| volume | 196 | 0.0677 | 0.1059 | no gross edge |
| volume | 198 | 0.0679 | 0.1020 | no gross edge |

| study | trial | n_trades | gross_pnl | fee_total | slippage_total | net_pnl | gross_exp_r | net_exp_r | cost_r_mean | se_r |
|---|---|---|---|---|---|---|---|---|---|---|
| tick | 197 | 135 | 328.75 | 579.10 | 115.82 | -366.16 | 0.0791 | -0.2204 | 0.2995 | 0.1209 |
| tick | 128 | 136 | 216.00 | 574.41 | 114.88 | -473.29 | 0.0706 | -0.2107 | 0.2812 | 0.1221 |
| tick | 191 | 117 | 188.22 | 505.86 | 101.17 | -418.81 | 0.0752 | -0.1959 | 0.2711 | 0.1308 |
| volume | 193 | 468 | 852.63 | 1691.44 | 338.29 | -1177.10 | 0.0902 | -0.1059 | 0.1961 | 0.0570 |
| volume | 196 | 470 | 672.39 | 1691.73 | 338.35 | -1357.69 | 0.0677 | -0.1257 | 0.1933 | 0.0530 |
| volume | 198 | 487 | 690.81 | 1739.88 | 347.98 | -1397.04 | 0.0679 | -0.1242 | 0.1921 | 0.0510 |

Gross expectancy is +0.07..+0.09 R, 0.57-0.65 se (tick) and 1.3-1.6 se (volume): statistically indistinguishable from zero. Costs are 0.19-0.30 R per trade, 2-4x the gross, so net is -0.11 to -0.22 R. Even a cost-free run would not show a significant edge (`stress` zero-cost net equals gross).

**Widened bounds cannot create an edge.** The widening in `scripts/search_spaces/*.json` is justified by boundary evidence only (search hygiene), not by a cost-limited edge. Strategy weakness dominates.

Source: replay reports in `output/diagnose/` (git-ignored), produced by `scripts/diagnose_trades.py`.

## 2. Study inventory

| study | trials | feasible | best feasible | top-10 feasible trial numbers |
|---|---|---|---|---|
| `…fixed_tick_imbalance…__s4eb64ef3` | 200 | 157 | -0.223 | 106, 128, 164, 167, 181, 191, 192, 196, 197, 198 |
| `…fixed_volume_imbalance…__s4eb64ef3` | 200 | 184 | -0.940 | 143, 173, 183, 192, 193, 195, 196, 197, 198, 199 |

## 3. Boundary audit (`python scripts/search_space_audit.py --study-db … --study-name …`)

Rule (constants at module level in `scripts/search_space_audit.py`):
- Position `x` in [0, 1] is linear, or log-scale for `float_log` dimensions.
- **Top** = the `TOP_K=10` best feasible trials by value, counted when `x < 0.10` (`TOP_EDGE_LOW`) or `x > 0.90` (`TOP_EDGE_HIGH`).
- **Tail** = the last `TAIL=50` COMPLETE trials by number, as the fraction with `x < 0.05` or `x > 0.95`.
- `widen_*` is true when top >= `TOP_COUNT_MIN=5` **or** tail >= `TAIL_FRACTION_MIN=0.30`.

Tick:

| dimension | top_lo | top_hi | tail_lo | tail_hi | widen |
|---|---|---|---|---|---|
| price_proximity_pct | 8 | 0 | 0.28 | 0 | LOW |
| min_composite_score | 0 | 0 | 0 | 0.02 | - |
| signal_cooldown_seconds | 9 | 0 | 0.36 | 0 | LOW |
| volume_decline_pct | 0 | 0 | 0 | 0 | - |
| lookback_bars | 0 | 0 | 0 | 0 | - |
| delta_failure_pct | 7 | 0 | 0.30 | 0.02 | LOW |
| absorption_min_delta_pct | 0 | 0 | 0.02 | 0 | - |
| initiative_min_delta_pct | 0 | 2 | 0 | 0.10 | - |
| initiative_min_body_bps | 0 | 0 | 0 | 0 | - |
| initiative_volume_acceleration_min | 0 | 0 | 0 | 0 | - |
| default_stop_pct | 0 | 10 | 0 | 0.62 | HIGH |
| target_rr | 7 | 0 | 0.28 | 0 | LOW |

Volume:

| dimension | top_lo | top_hi | tail_lo | tail_hi | widen |
|---|---|---|---|---|---|
| price_proximity_pct | 8 | 0 | 0.28 | 0 | LOW |
| min_composite_score | 4 | 0 | 0.02 | 0 | - |
| signal_cooldown_seconds | 0 | 1 | 0 | 0.18 | - |
| volume_decline_pct | 1 | 0 | 0.02 | 0 | - |
| lookback_bars | 1 | 0 | 0 | 0 | - |
| delta_failure_pct | 0 | 0 | 0 | 0 | - |
| absorption_min_delta_pct | 0 | 9 | 0 | 0.36 | HIGH |
| initiative_min_delta_pct | 0 | 10 | 0 | 0.76 | HIGH |
| initiative_min_body_bps | 0 | 0 | 0 | 0 | - |
| initiative_volume_acceleration_min | 0 | 0 | 0 | 0 | - |
| default_stop_pct | 0 | 8 | 0 | 0 | HIGH |
| target_rr | 0 | 0 | 0 | 0 | - |

## 4. Conclusions

- No feasible trial in either study has a positive objective (best tick -0.223, best volume -0.940).
- The top trials are the last ~8 of 200: the search had not converged, and TPE collapsed onto boundary basins. 200 trials over 12 dimensions is thin.
- Tick value and trade count have Spearman -0.86: the optimizer escapes cost by trading less, with the best trials sitting near the 105-trade floor.
- Search-segment bars have `|delta|/volume` median 0.35 (tick) and 0.40 (volume); 27-33% of bars exceed 0.6 and 15-20% exceed 0.8, which explains the pressure on the delta-threshold dimensions.
- Strategy weakness dominates. Boundary pressure is a secondary, search-hygiene issue; insufficient trials is a third.

## 5. Decisions: widened bounds

Files: `fixed_tick_imbalance.json`, `fixed_volume_imbalance.json` (used via `--search-space`). Each is a superset of the defaults (widened side only, same kind); every other dimension keeps its default. `load_space` now also rejects `high > 1` for `absorption_min_delta_pct` and `initiative_min_delta_pct`.

| Dimension | Default | Tick file | Volume file |
|---|---|---|---|
| `default_stop_pct` (log) | 0.001-0.01 | 0.001-0.025 | 0.001-0.025 |
| `price_proximity_pct` (log) | 0.0005-0.005 | 0.0002-0.005 | 0.0002-0.005 |
| `signal_cooldown_seconds` | 5-120 | 1.0-120 | unchanged |
| `delta_failure_pct` | 0.5-0.95 | 0.2-0.95 | unchanged |
| `target_rr` | 1.0-4.0 | 0.6-4.0 | unchanged |
| `absorption_min_delta_pct` | 0.2-0.6 | unchanged | 0.2-0.9 |
| `initiative_min_delta_pct` | 0.2-0.6 | unchanged | 0.2-0.9 |

Notes:

- Stop 0.025 is about the 90th-percentile |move| over ~200 bars; trade counts will fall toward the 105 floor.
- Cooldown `low=1.0` because `load_space` requires `low > 0` and a cooldown below one bar (median 12-18 s) is already inert.
- `target_rr < 1` is allowed because `RiskConfig.min_rr_ratio` is not enforced anywhere.
- `|delta|/volume` is at most 1 by construction (27-33% of bars exceed 0.6, 15-20% exceed 0.8), so 0.9 is the practical ceiling for the delta thresholds.

Deliberately NOT widened despite mild pressure (fail the audit rule): tick `absorption_min_delta_pct` and `initiative_min_delta_pct`, `initiative_volume_acceleration_min`, volume `signal_cooldown_seconds`. All other dimensions are unchanged.

These bounds are justified by boundary evidence only; the replays in section 1 found no gross edge, so they cannot by themselves create one.
