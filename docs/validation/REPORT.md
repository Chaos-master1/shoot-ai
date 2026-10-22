# Kawkab AI — Model Validation Report

*Generated 2026-09-16T01:01:41.988395+00:00 — fully reproducible:*
```
PYTHONPATH=src python scripts/validate_models.py
```

## xG (Expected Goals)

- Active model: **trained (trained_xg_coefficients.json)**
- Corpus: **297 StatsBomb open-data matches**, 7354 non-penalty shots
- Match-level train/val split: seed 13, 20% val (1449 shots)

| Model | Brier ↓ | AUC ↑ | LogLoss ↓ | ECE ↓ | mean xG |
|---|---|---|---|---|---|
| kawkab_active | 0.0707 | 0.7724 | 0.2547 | 0.0143 | 0.0999 |
| kawkab_heuristic | 0.0804 | 0.7715 | 0.3289 | 0.0682 | 0.0195 |
| statsbomb_xg_reference | 0.0667 | 0.7797 | 0.2454 | 0.0089 | 0.0941 |

- Brier 95% CI: Kawkab **0.0613–0.0805** vs StatsBomb 0.0569–0.0763
- Per-shot agreement with StatsBomb xG: MAE 0.0400, r = 0.801

*Penalties excluded (fixed 0.76 xG). Outcomes from StatsBomb open data (CC BY-NC-SA 4.0, non-commercial). Lower Brier = better; Brier 0.25 = coin-flip baseline.*

## PSxG (Post-Shot xG — goalkeeper metric)

- Trained on **2768 on-target shots** (2219 train / 549 held-out, match-level split)
- kawkab_psxg_trained: Brier **0.1673**, AUC **0.7533**, ECE 0.0537, mean PSxG 0.3057 (true goal rate 0.2750)

## xT (Expected Threat reference grid)

- League-wide grid trained from **297 matches / 534,927 actions** — shape 20x32
- Monotonicity check: max own-half zone 0.0245 < max final-third zone 0.8966 → **True**
