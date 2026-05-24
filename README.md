# Data Science Portfolio

Welcome to my data science portfolio! This repository showcases my projects, skills, and experience in data analysis, machine learning, and business intelligence.

## About Me

I'm a data scientist passionate about turning data into actionable insights. With expertise in Python, SQL, and various ML frameworks, I help organizations make data-driven decisions.

## Projects

### AI Gambling Support Agent
A conversational AI agent built with Pydantic AI that helps users reach gambling addiction support resources by composing and sending support request emails.

- **Technologies**: Python, Pydantic AI, OpenAI/Claude APIs, SMTP
- **Features**: Multi-turn conversation, automated email sending, helpline resources

### Real-time Social Media Sentiment Dashboard
This Streamlit dashboard collects recent tweets about a specific topic, analyzes their sentiment using TextBlob's natural language processing, and visualizes the results with interactive charts and metrics.

- **Technologies**: Python, Plotly, Pandas, Streamlit, TextBlob
- **Features**: Real-time Sentiment Analysis, Interactive Visualizations, Engagement Tracking, Tweet Details, Summary Statistics

### Quantile Return Forecasting Pipeline
An end-to-end machine learning system for predicting 3-month forward stock returns across multiple quantiles using LightGBM quantile regression. Integrates market-based probabilities from APIs (Alpha Vantage, FRED, Kalshi) and reconstructs absolute price targets from return predictions.

- **Technologies**: Python, LightGBM, Scikit-learn, Pandas, Streamlit, APIs (Alpha Vantage, FRED, Kalshi)
- **Features**: 
  - Quantile regression (20th, 50th, 80th percentile forecasts)
  - Walk-forward backtesting with directional accuracy & information coefficient metrics
  - Technical indicators + macroeconomic + sentiment features
  - Universe filtering by market cap ($20B+)
  - CLI and Streamlit web interface with "fan chart" visualizations
- **Key Insight**: Predicting percentage returns (stationary) instead of prices enables more stable models and economically meaningful predictions
- **Metrics**: Directional Accuracy, Information Coefficient, MAE/MAPE for both returns and reconstructed prices

### Automated Valuation Model - Real Estate
A production-grade dual-model Automated Valuation Model (AVM) system that predicts property prices as probabilistic intervals, not point estimates. Combines LightGBM quantile regression (Stage 1) with a meta-error predictor (Stage 2) to isolate model confidence from prediction uncertainty, enabling dynamic risk-weighted offer generation and ensemble-ready architectures.

- **Technologies**: Python, LightGBM, Scikit-learn, Pandas, NumPy, Kings County Housing Dataset
- **Architecture**: 
  - **Stage 1**: Primary Quantile AVM (Q₁₀, Q₅₀, Q₉₀) with structural + macro + geo features
  - **Stage 2**: Meta-Error Predictor trained on out-of-fold errors → Expected Error Score (0-100 EES)
  - **Stage 3**: Calibration Engine (learns interval scaling for 80% empirical coverage)
  - **Stage 4**: Offer Engine (asymmetric bid spreads + ensemble weight generation)
  - **Stage 5**: Market Feedback Loop (post-hoc multiplier for macro regime detection)
- **Key Metrics**: 80.2% calibration coverage, 8.3% median MAPE, 71% directional accuracy
- **Production Features**: Chronological validation, safe target encoding, data leakage prevention, no-retrain multiplier adjustments
- **Ensembling Framework**: EES acts as universal gating signal for downstream multi-model integration (inverse-error variance weighting)

## Skills

- **Programming**: Python, SQL
- **Machine Learning**: Scikit-learn, TensorFlow, LightGBM, XGBoost, SHAP
- **Data Visualization**: Amazon Quick, Mode, Tableau, Looker, Matplotlib, Plotly
- **Tools**: Git, AWS
- **Other**: ETL pipelines, feature engineering, model validation

## Resume

For a detailed overview of my professional experience, education, and certifications, please view my resume:

[LaneCadell_Resume.pdf](./LaneCadell_Resume.pdf)

## Contact

- **Email**: [LaneDanielCadell@gmail.com](mailto:LaneDanielCadell@gmail.com)
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/lanecadell)
- **GitHub**: [GitHub](https://github.com/lanecadell)

Feel free to reach out if you'd like to discuss potential collaborations or opportunities!

---

*This portfolio is a work in progress. More projects and details coming soon!* 
