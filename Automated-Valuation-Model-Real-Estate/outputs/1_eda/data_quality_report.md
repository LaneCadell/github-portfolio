# Data Quality Report

## Outlier Pipeline Summary

- Original dataset size: **21,613** rows
- Final dataset size after deterministic outlier filtering: **17,495** rows

### Records removed by filter type

| Filter | Rows removed |
|---|---:|
| Original count | 21,613 |
| Price floor ceiling removed | 7 |
| Physical anomalies removed | 276 |
| Price per sqft extremes removed | 0 |
| Isolation forest removed | 214 |
| Price iqr removed | 958 |
| Sqft living iqr removed | 283 |
| Sqft lot iqr removed | 1,936 |
| Price per sqft iqr removed | 444 |

## Imputation Frequency by Feature

The imputation frequencies below are calculated using training-set statistics only, ensuring zero leakage into validation/test partitions.

| Feature | Train missing | Val missing | Test missing |
|---|---:|---:|---:|
| bedrooms | 0 | 0 | 0 |
| bathrooms | 0 | 0 | 0 |
| sqft_living | 0 | 0 | 0 |
| sqft_lot | 0 | 0 | 0 |
| sqft_above | 0 | 0 | 0 |
| sqft_basement | 0 | 0 | 0 |

## Notes

- Imputation uses group-wise median values computed from the training partition only.
- All missing architectural and geographic values are filled using the most conservative zipcode-level statistics available.
- The final dataset is ready for chronological train/validation/test splitting without leakage.