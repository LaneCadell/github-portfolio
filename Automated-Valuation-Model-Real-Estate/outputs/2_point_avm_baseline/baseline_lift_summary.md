# AVM Point Forecast vs. Naive Baseline

## Lift Metrics

| Metric | Q50 AVM | Rolling Median Baseline | Improvement |
|---|---:|---:|---:|
| MAPE | 21.14 | 36.11 | 14.96 |
| MdAPE | 17.36 | 30.01 | 12.65 |
| RMSE | 152444.70 | 203934.95 | 51490.24 |

## Notes

- Baseline predictions are produced by a rolling monthly median price model fitted only on the training partition.
- All error calculations are strictly evaluated on the held-out test set to prevent leakage.
- Improvement is computed as Baseline minus Q50 AVM so positive values indicate model lift.