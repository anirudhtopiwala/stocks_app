import requests
import json
import os
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FILE_PATH = "portfolio.csv"

# Add these constants at the top of the file
CACHE_DIR = "cache"
NEWS_CACHE_FILE = os.path.join(CACHE_DIR, "news_cache.json")
CACHE_EXPIRY_HOURS = 4  # Cache news for 4 hours
RATE_LIMIT_DELAY = 1.2  # Delay between API calls in seconds

# Add this dictionary with industry mappings
TICKER_INDUSTRIES = {
    # Technology & Software
    "MSFT": "Technology",
    "AAPL": "Technology",
    "PLTR": "Technology",
    "SNOW": "Technology",
    "EPAM": "Technology Software",
    # Semiconductors
    "NVDA": "Semiconductors",
    "TSM": "Semiconductors",
    "ASML": "Semiconductors",
    # Social Media & Internet
    "META": "Internet & Social Media",
    "GOOG": "Internet & Social Media",
    "AMZN": "E-Commerce",
    # Financial Technology
    "COIN": "FinTech",
    "NU": "FinTech",
    "FOUR": "FinTech",
    # Space & Defense
    "RKLB": "Aerospace",
    "BWXT": "Defense & Nuclear",
    # Clean Energy & Future Tech
    "OKLO": "Nuclear Energy",
    "IONQ": "Quantum Computing",
    "REAX": "Clean Energy",
    # Consumer & Retail
    "CELH": "Consumer Goods",
    "WMT": "Retail",
    "COST": "Retail",
    # Healthcare
    "EXAS": "Healthcare",
    # International
    "BABA": "China E-Commerce",
    "SLVYY": "International Materials",
    # Consumer Staples
    "PM": "Consumer Staples",
    # ETFs
    "VGT": "Technology ETF",
    "SPY": "US Market ETF",
    "QQQM": "Tech-Heavy ETF",
    "IBIT": "Bitcoin ETF",
    # Precious Metals
    "GLDM": "Gold",
    # Electric Vehicles
    "TSLA": "Electric Vehicles",
}

# Group these into broader sectors for the pie chart
INDUSTRY_GROUPS = {
    "Technology": ["Technology", "Technology Software", "Semiconductors"],
    "Internet & Digital": ["Internet & Social Media", "E-Commerce", "China E-Commerce"],
    "Financial Services": ["FinTech"],
    "Innovation & Future Tech": [
        "Aerospace",
        "Defense & Nuclear",
        "Nuclear Energy",
        "Quantum Computing",
        "Clean Energy",
    ],
    "Consumer & Retail": ["Consumer Goods", "Retail", "Consumer Staples"],
    "Healthcare": ["Healthcare"],
    "ETFs": ["Technology ETF", "US Market ETF", "Tech-Heavy ETF", "Bitcoin ETF"],
    "Commodities": ["Gold"],
    "Automotive": ["Electric Vehicles"],
    "Other": ["International Materials"],
}


def get_broad_industry(specific_industry):
    """Convert specific industry to broader group"""
    for broad, specifics in INDUSTRY_GROUPS.items():
        if specific_industry in specifics:
            return broad
    return "Other"


def get_cached_news(ticker: str, allow_expired: bool = False) -> Optional[List[Dict]]:
    """Retrieve cached news, optionally allowing expired cache"""
    try:
        if not os.path.exists(NEWS_CACHE_FILE):
            return None

        with open(NEWS_CACHE_FILE, "r") as f:
            cache = json.load(f)

        if ticker not in cache:
            return None

        cached_data = cache[ticker]
        cached_time = datetime.fromisoformat(cached_data["timestamp"])

        # Check if cache is expired (unless we're allowing expired cache)
        if not allow_expired and datetime.now() - cached_time > timedelta(
            hours=CACHE_EXPIRY_HOURS
        ):
            return None

        return cached_data["articles"]
    except Exception as e:
        print(f"Error reading cache for {ticker}: {e}")
        return None


def save_to_cache(ticker, articles):
    """Save news articles to cache"""
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load existing cache or create new one
        cache = {}
        if os.path.exists(NEWS_CACHE_FILE):
            with open(NEWS_CACHE_FILE, "r") as f:
                cache = json.load(f)

        # Update cache with new data
        cache[ticker] = {"articles": articles, "timestamp": datetime.now().isoformat()}

        # Save updated cache
        with open(NEWS_CACHE_FILE, "w") as f:
            json.dump(cache, f)

    except Exception as e:
        print(f"Error saving to cache: {e}")


def get_current_price(ticker):
    """Fetch current price for a ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        current_data = stock.info
        return current_data.get("regularMarketPrice", None)
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None


def update_current_prices(df):
    """Update current prices for all tickers in parallel"""
    unique_tickers = df["ticker"].unique()
    prices = {}

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a dictionary of futures
        future_to_ticker = {
            executor.submit(get_current_price, ticker): ticker
            for ticker in unique_tickers
        }

        # Process completed futures
        for future in future_to_ticker:
            ticker = future_to_ticker[future]
            try:
                price = future.result()
                if price is not None:
                    prices[ticker] = price
                else:
                    print(f"Could not fetch price for {ticker}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    # Update the dataframe with new prices
    df["current_price"] = df["ticker"].map(prices)

    # Recalculate values based on new prices
    df["total_value"] = df["quantity"] * df["current_price"]
    df["profit_loss"] = df["total_value"] - df["total_cost"]
    df["return_percentage"] = (df["profit_loss"] / df["total_cost"]) * 100

    return df


def derive_ticker_type(ticker: str) -> str:
    """Derive the type of the ticker based on predefined rules."""
    etfs = ["QQQM", "VGT", "SPY"]  # Add more ETF tickers as needed
    if ticker in etfs:
        return "ETF"
    elif ticker in ["GLDM"]:  # Gold tickers
        return "GOLD"
    elif ticker in ["IBIT"]:  # Bitcoin tickers
        return "CRYPTO"
    else:
        return "STOCK"  # Default to STOCK for all other tickers


# Example usage in load_portfolio function
def load_portfolio(file_path: str) -> pd.DataFrame:
    try:
        # Read the simplified CSV
        df = pd.read_csv(file_path)
        print(f"Successfully read CSV. Shape: {df.shape}")
        print(f"Columns found: {df.columns.tolist()}")

        # Rename columns to internal names for consistency
        df = df.rename(
            columns={
                "TICKER": "ticker",
                "QUANTITY": "quantity",
                "COST/SHARE": "avg_cost",
            }
        )
        print(f"Columns after renaming: {df.columns.tolist()}")
        # Assign position based on row index (1-based)
        df["position"] = df.index + 1

        # Map industries using provided mappings
        df["specific_industry"] = df["ticker"].map(TICKER_INDUSTRIES).fillna("Other")
        df["industry"] = df["specific_industry"].apply(get_broad_industry)

        # Derive ticker type
        df["type"] = df["ticker"].apply(derive_ticker_type)

        # Fetch current prices
        df["total_cost"] = df["quantity"] * df["avg_cost"]
        print(f"Total cost: {df['total_cost'].sum()}")
        df = update_current_prices(df)  # Assumes this adds "current_price" column

        # Calculate financial columns
        df["total_value"] = df["quantity"] * df["current_price"]
        df["profit_loss"] = df["total_value"] - df["total_cost"]
        df["return_percentage"] = (df["profit_loss"] / df["total_cost"]) * 100
        df["return_percentage"] = df["return_percentage"].where(
            df["total_cost"] != 0, 0
        )

        # Calculate portfolio weight
        total_portfolio_value = df["total_value"].sum()
        df["portfolio_weight"] = (df["total_value"] / total_portfolio_value) * 100

        # Sort by portfolio_weight descending
        df = df.sort_values("portfolio_weight", ascending=False)

        # Define column order to match original output
        columns_order = [
            "position",
            "ticker",
            "type",  # Include the new type column
            "industry",
            "specific_industry",
            "quantity",
            "avg_cost",
            "total_cost",
            "current_price",
            "total_value",
            "profit_loss",
            "portfolio_weight",
            "return_percentage",
        ]
        df = df[columns_order]

        # Calculate portfolio summary
        portfolio_summary = {
            "Total Investment": df["total_cost"].sum(),
            "Current Value": df["total_value"].sum(),
            "Total Return": df["profit_loss"].sum(),
        }
        if portfolio_summary["Total Investment"] != 0:
            portfolio_summary["Return Percentage"] = (
                portfolio_summary["Total Return"]
                / portfolio_summary["Total Investment"]
            ) * 100
        else:
            portfolio_summary["Return Percentage"] = 0
        return df, portfolio_summary
    except Exception as e:
        print(f"Error in load_portfolio: {str(e)}")
        print(f"Full error details: ", e)
        raise


def get_news_for_ticker(ticker: str, api_key: str) -> List[Dict]:
    """Get news with caching, rate limiting and better error handling"""
    try:
        # Check cache first
        cached_news = get_cached_news(ticker)
        if cached_news is not None:
            print(f"Using cached news for {ticker}")
            return cached_news

        print(f"Fetching fresh news for {ticker}")
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": f"{ticker} stock",
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
            "pageSize": 5,
        }

        response = requests.get(url, params=params)
        print(f"Response status for {ticker}: {response.status_code}")

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            # Save to cache if we got articles
            if articles:
                print(f"Saving news to cache for {ticker}")
                save_to_cache(ticker, articles)
            return articles
        else:
            print(f"Error fetching news for {ticker}: {response.status_code}")
            print(f"Response: {response.text}")
            return []

    except Exception as e:
        print(f"Exception fetching news for {ticker}: {str(e)}")
        return []


def compile_news_briefing(portfolio_df, api_key):
    """Compile news with improved error handling and rate limiting"""
    news_briefing = {}
    cache_status = {}

    for ticker in portfolio_df["ticker"].unique():
        try:
            print(f"Getting news for {ticker}")
            cached_news = get_cached_news(ticker)

            if cached_news is not None:
                news_briefing[ticker] = cached_news[:5]
                cache_status[ticker] = "cached"
            else:
                articles = get_news_for_ticker(ticker, api_key)
                if articles:
                    news_briefing[ticker] = articles[:5]
                    cache_status[ticker] = "fresh"
                else:
                    news_briefing[ticker] = []
                    cache_status[ticker] = "error"

        except Exception as e:
            print(f"Error processing news for {ticker}: {e}")
            news_briefing[ticker] = []
            cache_status[ticker] = "error"

    return news_briefing, cache_status


def get_ticker_performance(ticker: str):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})
        prices = [
            float(day_data["4. close"]) for day_data in list(time_series.values())[:30]
        ]  # Last 30 days

        if not prices:
            return {
                "trend": 0.0,
                "volatility": 0.0,
                "ma7": 0.0,
                "ma30": 0.0,
                "trend_direction": "unknown",
            }

        # Calculate metrics
        start_price = prices[-1]
        end_price = prices[0]
        percent_change = ((end_price - start_price) / start_price) * 100

        # Moving averages
        ma7 = sum(prices[:7]) / 7 if len(prices) >= 7 else end_price
        ma30 = sum(prices) / len(prices)

        # Volatility calculation
        returns = [
            (prices[i] - prices[i + 1]) / prices[i + 1] for i in range(len(prices) - 1)
        ]
        volatility = (
            sum(r * r for r in returns) / len(returns)
        ) ** 0.5 * 100  # Annualized

        return {
            "trend": percent_change,
            "volatility": volatility,
            "ma7": ma7,
            "ma30": ma30,
            "trend_direction": "upward" if ma7 > ma30 else "downward",
        }
    return {
        "trend": 0.0,
        "volatility": 0.0,
        "ma7": 0.0,
        "ma30": 0.0,
        "trend_direction": "unknown",
    }


def generate_portfolio_suggestions(portfolio_df, market_data=None):
    suggestions = {}
    total_value = portfolio_df["total_value"].sum()

    for idx, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        weight = row["total_value"] / total_value * 100
        performance = get_ticker_performance(ticker)

        suggestion_points = []

        # Check position size
        if weight > 20:
            suggestion_points.append(
                "Consider reducing position size - currently over 20% of portfolio"
            )
        elif weight < 2:
            suggestion_points.append(
                "Consider increasing position or exiting - position size below 2%"
            )

        # Check performance trends
        if performance["trend_direction"] == "upward":
            if performance["trend"] > 10:
                suggestion_points.append(
                    f"Strong upward trend (+{performance['trend']:.1f}%) - consider taking partial profits"
                )
            else:
                suggestion_points.append("Upward trend - maintain position")
        elif performance["trend_direction"] == "downward":
            if performance["trend"] < -10:
                suggestion_points.append(
                    f"Strong downward trend ({performance['trend']:.1f}%) - consider averaging down or cutting losses"
                )
            else:
                suggestion_points.append("Downward trend - monitor closely")

        # Check volatility
        if performance["volatility"] > 30:
            suggestion_points.append(
                f"High volatility ({performance['volatility']:.1f}%) - consider hedging or reducing position"
            )

        # Combine suggestions
        suggestions[ticker] = {
            "weight": weight,
            "trend": performance["trend"],
            "volatility": performance["volatility"],
            "suggestions": suggestion_points,
        }

    return suggestions
