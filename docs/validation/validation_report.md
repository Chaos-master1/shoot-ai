# Kawkab Validation Report

Generated: 2026-09-15T06:50:28.700552+00:00
Corpus: `data/statsbomb_corpus`

## model_cards — evaluated

| metric | value |
|---|---|
| registered | `17` |
| registered cards | 17 |

## xg_calibration — evaluated

| metric | value |
|---|---|
| n_shots | `128` |
| n_matches | `5` |
| brier | `0.0874` |
| roc_auc | `0.7717` |
| mae_vs_statsbomb_xg | `0.0489` |
| mean_predicted_xg | `0.1063` |
| mean_actual_goal_rate | `0.1172` |

## psxg_calibration — evaluated

| metric | value |
|---|---|
| n_shots | `114` |
| brier | `0.0768` |

## xt_grid — evaluated

| metric | value |
|---|---|
| shape | `[20, 32]` |
| min | `0.0134` |
| max | `0.8966` |
| all_nonnegative | `True` |
| max_in_attacking_third | `True` |

## pitch_control — evaluated

| metric | value |
|---|---|
| frames_evaluated | `80` |
| avg_home_control_pct | `42.2903` |
| avg_away_control_pct | `57.7097` |
| sums_to_100 | `100.0` |
| all_in_percent_range | `True` |
| home_third_control_pct | `85.9349` |
| away_third_control_pct | `3.0167` |

## ball_recovery — evaluated

| metric | value |
|---|---|
| recovery_events | `248` |
| home_recoveries | `135` |
| away_recoveries | `113` |
| home_zones_used | `25` |
| away_zones_used | `24` |
| recoveries_leading_to_shot | `13` |
| zones_spread | `True` |

## carry_xt — evaluated

| metric | value |
|---|---|
| carries | `265` |
| home_total_xt | `0.0` |
| away_total_xt | `0.5108` |
| home_progressive | `0` |
| away_progressive | `2` |
| xt_nonnegative | `True` |

## physical_load — evaluated

| metric | value |
|---|---|
| players | `22` |
| window_seconds | `79.8` |
| median_avg_speed_mps | `2.42` |
| within_sanity_band_1p8_2p6_mps | `True` |
