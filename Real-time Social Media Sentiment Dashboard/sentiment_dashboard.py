"""
Real-time Social Media Sentiment Dashboard
Collects recent tweets about a topic and analyzes sentiment using TextBlob
"""

import os
import tweepy
import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from datetime import datetime, timedelta
import json
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load or initialize API credentials
def get_twitter_api():
    """Initialize Twitter API client with credentials from Streamlit secrets or environment variables"""
    try:
        # Try to get from Streamlit secrets
        api_key = st.secrets.get("TWITTER_API_KEY")
        api_secret = st.secrets.get("TWITTER_API_SECRET")
        access_token = st.secrets.get("TWITTER_ACCESS_TOKEN")
        access_secret = st.secrets.get("TWITTER_ACCESS_SECRET")
        bearer_token = st.secrets.get("TWITTER_BEARER_TOKEN")
    except:
        # Fall back to environment variables
        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_secret = os.getenv("TWITTER_ACCESS_SECRET")
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    
    if not bearer_token:
        return None
    
    client = tweepy.Client(bearer_token=bearer_token)
    return client


def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity


def fetch_tweets(client, query, count=100):
    """Fetch recent tweets about a topic"""
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(count, 100),  # API limit is 100 per request
            tweet_fields=['created_at', 'public_metrics'],
            expansions=['author_id'],
            user_fields=['username', 'public_metrics']
        )
        
        if not tweets.data:
            return []
        
        # Extract user info
        users = {user.id: user for user in tweets.includes['users']} if tweets.includes and 'users' in tweets.includes else {}
        
        tweet_data = []
        for tweet in tweets.data:
            user = users.get(tweet.author_id, None)
            tweet_data.append({
                'text': tweet.text,
                'created_at': tweet.created_at,
                'author': user.username if user else "Unknown",
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count'],
                'replies': tweet.public_metrics['reply_count']
            })
        
        return tweet_data
    
    except tweepy.TweepyException as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return []


def process_tweets(tweets):
    """Process tweets and add sentiment analysis"""
    processed = []
    for tweet in tweets:
        sentiment, polarity = analyze_sentiment(tweet['text'])
        processed.append({
            **tweet,
            'sentiment': sentiment,
            'polarity': polarity
        })
    return processed


# Main dashboard
st.title("📊 Real-time Social Media Sentiment Dashboard")
st.markdown("Track sentiment trends about any topic using Twitter data and TextBlob analysis")

# Sidebar for configuration
st.sidebar.header("Configuration")
topic = st.sidebar.text_input("Enter a topic to analyze", value="AI", help="e.g., 'Apple stock', 'AI', 'Python'")
num_tweets = st.sidebar.slider("Number of tweets to fetch", min_value=10, max_value=100, value=100, step=10)
refresh = st.sidebar.button("🔄 Refresh Data")

# Check for API credentials
client = get_twitter_api()
if not client:
    st.warning(
        "⚠️ Twitter API credentials not configured. "
        "\n\nPlease add your credentials:\n\n"
        "1. **For Streamlit Cloud**: Add to `.streamlit/secrets.toml`:\n"
        "```\n"
        "TWITTER_BEARER_TOKEN = \"your_bearer_token\"\n"
        "TWITTER_API_KEY = \"your_api_key\"\n"
        "TWITTER_API_SECRET = \"your_api_secret\"\n"
        "TWITTER_ACCESS_TOKEN = \"your_access_token\"\n"
        "TWITTER_ACCESS_SECRET = \"your_access_secret\"\n"
        "```\n\n"
        "2. **For local development**: Set environment variables:\n"
        "```bash\n"
        "export TWITTER_BEARER_TOKEN=\"your_bearer_token\"\n"
        "```\n\n"
        "Get your credentials from [Twitter Developer Portal](https://developer.twitter.com/)"
    )
    st.stop()

# Main content
if topic and (refresh or 'tweets_data' not in st.session_state):
    with st.spinner(f"Fetching tweets about '{topic}'..."):
        raw_tweets = fetch_tweets(client, topic, num_tweets)
        
        if raw_tweets:
            st.session_state.tweets_data = process_tweets(raw_tweets)
        else:
            st.error(f"No tweets found for '{topic}'. Try a different search term.")
            st.stop()

if 'tweets_data' in st.session_state:
    tweets_df = pd.DataFrame(st.session_state.tweets_data)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_counts = tweets_df['sentiment'].value_counts()
        positive_count = sentiment_counts.get('Positive', 0)
        st.metric("😊 Positive", positive_count, delta=f"{(positive_count/len(tweets_df)*100):.1f}%")
    
    with col2:
        negative_count = sentiment_counts.get('Negative', 0)
        st.metric("😞 Negative", negative_count, delta=f"{(negative_count/len(tweets_df)*100):.1f}%")
    
    with col3:
        neutral_count = sentiment_counts.get('Neutral', 0)
        st.metric("😐 Neutral", neutral_count, delta=f"{(neutral_count/len(tweets_df)*100):.1f}%")
    
    with col4:
        avg_polarity = tweets_df['polarity'].mean()
        st.metric("📈 Avg Sentiment Score", f"{avg_polarity:.3f}", delta=f"{avg_polarity*100:.1f}%")
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_chart = px.pie(
            tweets_df,
            names='sentiment',
            values=tweets_df['sentiment'].value_counts().values,
            color_discrete_map={'Positive': '#1f77b4', 'Negative': '#ff7f0e', 'Neutral': '#2ca02c'},
            title="Sentiment Breakdown"
        )
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with col2:
        polarity_hist = px.histogram(
            tweets_df,
            x='polarity',
            nbins=20,
            title="Polarity Distribution",
            labels={'polarity': 'Sentiment Polarity Score'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(polarity_hist, use_container_width=True)
    
    # Engagement metrics
    st.subheader("Engagement Metrics by Sentiment")
    engagement_data = tweets_df.groupby('sentiment')[['likes', 'retweets', 'replies']].mean().reset_index()
    engagement_chart = px.bar(
        engagement_data,
        x='sentiment',
        y=['likes', 'retweets', 'replies'],
        barmode='group',
        title="Average Engagement by Sentiment",
        labels={'value': 'Count', 'variable': 'Engagement Type'},
        color_discrete_map={'likes': '#1f77b4', 'retweets': '#ff7f0e', 'replies': '#2ca02c'}
    )
    st.plotly_chart(engagement_chart, use_container_width=True)
    
    # Top tweets
    st.subheader("Top Tweets by Engagement")
    top_tweets = tweets_df.nlargest(5, 'likes')[['author', 'text', 'sentiment', 'polarity', 'likes', 'retweets']]
    st.dataframe(
        top_tweets,
        column_config={
            'polarity': st.column_config.NumberColumn(format="%.3f"),
            'text': st.column_config.TextColumn(width="large")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Recent tweets with sentiment
    st.subheader("Recent Tweets")
    with st.expander("View all analyzed tweets"):
        display_cols = ['author', 'text', 'sentiment', 'polarity', 'created_at']
        st.dataframe(
            tweets_df[display_cols].sort_values('created_at', ascending=False),
            column_config={
                'polarity': st.column_config.NumberColumn(format="%.3f"),
                'text': st.column_config.TextColumn(width="large")
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Statistics panel
    st.subheader("Summary Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.write("**Total Tweets Analyzed**", len(tweets_df))
        st.write("**Date Range**", f"{tweets_df['created_at'].min().strftime('%Y-%m-%d')} to {tweets_df['created_at'].max().strftime('%Y-%m-%d')}")
    
    with stats_col2:
        st.write("**Total Engagement**", tweets_df[['likes', 'retweets', 'replies']].sum().sum().astype(int))
        st.write("**Avg Likes per Tweet**", f"{tweets_df['likes'].mean():.1f}")
    
    with stats_col3:
        st.write("**Most Common Sentiment**", tweets_df['sentiment'].value_counts().index[0])
        st.write("**Sentiment Polarity Range**", f"{tweets_df['polarity'].min():.3f} to {tweets_df['polarity'].max():.3f}")

else:
    st.info("Enter a topic and click 'Refresh Data' to begin analyzing sentiment")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    📊 Real-time Social Media Sentiment Dashboard | Powered by Tweepy & TextBlob | Data from Twitter API
    </div>
    """,
    unsafe_allow_html=True
)
