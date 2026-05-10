# Real-time Social Media Sentiment Dashboard

Track sentiment trends about any topic using Twitter data and TextBlob sentiment analysis.

## Overview

This Streamlit dashboard collects recent tweets about a specific topic, analyzes their sentiment using TextBlob's natural language processing, and visualizes the results with interactive charts and metrics.

### Features

- 📊 **Real-time Sentiment Analysis**: Fetch up to 100 recent tweets and analyze sentiment
- 📈 **Interactive Visualizations**: Pie charts, histograms, and engagement metrics powered by Plotly
- 👥 **Engagement Tracking**: Monitor likes, retweets, and replies by sentiment
- 🔍 **Tweet Details**: View individual tweets with sentiment scores and engagement
- 📋 **Summary Statistics**: Key metrics including polarity ranges and average engagement

## Project Context

Use the Twitter API (via Tweepy) to collect recent posts about any topic and analyze sentiment using TextBlob. The dashboard updates with live sentiment scores, showing positive/negative trends and engagement patterns.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Twitter API Credentials

1. Visit [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
2. Create a new app or use an existing one
3. Generate API credentials:
   - **API Key** (Consumer Key)
   - **API Secret** (Consumer Secret)
   - **Access Token**
   - **Access Secret**
   - **Bearer Token** (required for API v2)

### 3. Configure Credentials

#### Option A: For Local Development (Environment Variables)

```bash
export TWITTER_BEARER_TOKEN="your_bearer_token"
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_SECRET="your_access_secret"
```

#### Option B: For Streamlit Cloud (Secrets)

Create `.streamlit/secrets.toml`:

```toml
TWITTER_BEARER_TOKEN = "your_bearer_token"
TWITTER_API_KEY = "your_api_key"
TWITTER_API_SECRET = "your_api_secret"
TWITTER_ACCESS_TOKEN = "your_access_token"
TWITTER_ACCESS_SECRET = "your_access_secret"
```

### 4. Run the Dashboard

```bash
streamlit run sentiment_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

1. Enter a topic to analyze (e.g., "AI", "Apple stock", "Python")
2. Adjust the number of tweets to fetch (10-100)
3. Click "Refresh Data" to fetch and analyze tweets
4. View sentiment distribution, engagement metrics, and individual tweets

## Project Structure

```
Real-time Social Media Sentiment Dashboard/
├── sentiment_dashboard.py    # Main Streamlit app
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## Technical Stack

- **Data Collection**: Tweepy (Twitter API v2)
- **Sentiment Analysis**: TextBlob (polarity-based classification)
- **Dashboard**: Streamlit (interactive web interface)
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas (DataFrame operations)

## Sentiment Analysis Details

### Polarity Scores

- **Positive** (polarity > 0.1): Favorable sentiment
- **Negative** (polarity < -0.1): Unfavorable sentiment
- **Neutral** (-0.1 ≤ polarity ≤ 0.1): Neutral/Mixed sentiment

### Metrics Tracked

- Sentiment distribution (pie chart)
- Polarity distribution (histogram)
- Average engagement by sentiment (likes, retweets, replies)
- Top tweets by engagement
- Summary statistics (totals, ranges, averages)

## Example Queries

Try analyzing sentiment for:
- `"AI"` - Artificial intelligence trends
- `"Apple stock"` - Technology stocks
- `"climate change"` - Environmental topics
- `"cryptocurrency"` - Blockchain/crypto discussions
- `"#Python"` - Programming language trends

## Limitations

- Twitter API v2 free tier has rate limits (450 requests/15 minutes)
- Maximum 100 tweets per request
- Only analyzes tweets from the last 7 days (free tier)
- TextBlob sentiment analysis is rule-based (not as accurate as ML models)

## Future Enhancements

- [ ] Advanced NLP models (VADER, DistilBERT)
- [ ] Historical trend analysis
- [ ] Keyword extraction and word clouds
- [ ] Emotion classification (beyond positive/negative)
- [ ] Data export functionality (CSV, JSON)
- [ ] Caching for improved performance

## Requirements

- Python 3.8+
- Active Twitter Developer account
- Bearer token from Twitter API v2

## License

This project is part of a personal portfolio. Feel free to use and modify for learning purposes.
