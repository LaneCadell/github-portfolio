# AVM Point Forecast vs. Naive Baseline

## Lift Metrics

| Metric | Q50 AVM | Rolling Median Baseline | Improvement |
|---|---:|---:|---:|
| MAPE | 20.67 | 36.11 | 15.43 |
| MdAPE | 16.58 | 30.01 | 13.43 |
| RMSE | 147663.36 | 203934.95 | 56271.59 |

## Notes

- Baseline predictions are produced by a rolling monthly median price model fitted only on the training partition.
- All error calculations are strictly evaluated on the held-out test set to prevent leakage.
- Improvement is computed as Baseline minus Q50 AVM so positive values indicate model lift.