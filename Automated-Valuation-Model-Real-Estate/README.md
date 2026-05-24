# Risk-Aware Real Estate Pricing Engine

A production-grade pricing system that predicts not just a single home value, but a value range and a confidence score for each property. This project combines a quantile forecast model with a secondary error model, then uses uncertainty to shape risk-aware offers.

## What this project does

- Predicts a conservative lower bound, a central expected value, and an optimistic upper bound for each home.
- Measures how reliable those predictions are by learning from the primary model's past errors.
- Calibrates the predicted range so it hits the target coverage rate.
- Converts uncertainty into a risk-adjusted buy offer.
- Applies market-level feedback using recent resale outcomes.

## Pipeline overview

This pipeline is built around six clear stages:

1. Data loading and clean filtering of bad transactions
2. Feature engineering for structure, seasonality, and geography
3. Primary quantile model producing low / median / high forecasts3. Primary quantile modr 3. Primary quantile model producing low / median ratio3. Primary quantile model producing low / median / high forecasts3. Primary quantile modr 3. Primary quantile model producing low / median ratio3. Primary quantile model producing low / median / high forecasts3. Primary quantile modr 3. Primary quantile modetion](out3. Primary quantile modbuti3. Primary quantile model producing low / median / high forecasuare Foot](outputs/1_eda/price_per_sqft_distribution.png)

### Price versus living area

![Price vs Living Area](outputs/1_eda/price_vs_sqft.png)

### Property age distribution

![Age Distribution](outputs/1_eda/a![Age Distribution](outputs/1_eda/a![Age Distributio
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - estim- - - - - - - estimate, and an upper estimate.
- The gap between the lo- The gap between the lo- The gap between the lo- Tevel - The gap between the lo- The gap between the lo- The gap betwima- The gap between the lo- The gap between the lo- The gap between the lo- Tevel - The gap between the lo- The gap between the lo- The gap betwima- The gap between the lo- The gap between the lo- The gap between the lo- Tevel - The gap between the lo- The gap between the lo- The gap betwima- The gap between the lo- The gap between the lo- The gap between the lo- Tevel - The gap between the lo- The gap between the lo- The gap betwima- The gap between the lo- The gap between the lo- The gap between the lo- Tevel - The gap between the lo- The gap between ibution charts, and writes `outpu- The gap between the lo- The gap between the lo- The gne.- The gap between the lo-ing median baseline and compares the AVM's median forecast against it using MAPE, MdAPE, and RMSE.

### Generated phase 1 outputs

- `outputs/1_eda/data_quality_report.md`
- `outputs/1_eda/price_distribution.png`
- `outputs/1_eda/price_per_sqft_distribution.png`
- `outputs/1_eda/price_vs_sqft.png`
- `outputs/1_eda/age_distribution.png`
- `outputs/2_point_avm_baseline/model_lift_comparison.png` (after baseline run)
- `outputs/2_point_avm_baseline/baseline_lift_summary.md` (after baseline run)

## Installation

```bash
pip install -r requirements.txt
```

## Run the full production pipeline

```bash
python main.py --base-spread 0.05 --target-coverage 0.80
```

## Run phase 1 diagnostics

```bash
python 1_king_county_eda.py --data-path ./kc_house_data.csv
python 2_point_avm_baseline.py --data-path ./kc_house_data.csv
```

## Python API example

```python
from main import RiskAwareREPricingPipeline

pipeline = RiskAwareREPricingPipeline(data_path="./kc_house_data.csv")
results = pipeline.run_full_pipeline()

offer_report = results["offer_report"]
print(offer_report.head(10)[["property_id", "median_forecast", "buy_offer_price", "ees_score"]])
```

## Core concepts without equations

- **Q10:** a conservative lower price estimate.
- **Q50:** the central expected price estimate.
- **Q90:** an optimistic upper price estimate.
- **Uncertainty:** the width of the Q10–Q90 range.
- **EES:** a 0-100 confidence score for how reliable the forecast is.
- **Calibration:** a step that makes interval coverage match the chosen confidence level.
- **Offer generation:** starts from Q50 and increases the discount for higher uncertainty.

## Why this matters

This system is designed for acquisition decisions, not just value estimation. It gives buyers a price range and a risk signal, so offer decisions can be made with richer, more disciplined information.

## Recommended next step

Run `python 1_king_county_eda.py --data-path ./kc_house_data.csv` first, then run `python 2_point_avm_baseline.py --data-path ./kc_house_data.csv` to generate the model lift comparison report.
