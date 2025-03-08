from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from portfolio import (
    load_portfolio,
    compile_news_briefing,
    generate_portfolio_suggestions,
    FILE_PATH,
    NEWS_API_KEY,
    OPENAI_API_KEY,
    get_news_for_ticker,
    get_cached_news,
    save_to_cache,
)
from openai import OpenAI

import os
import json
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
import asyncio
import yfinance as yf
import numpy as np

# Add your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CACHE_DIR = "cache"
AI_SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "ai_summaries.json")
AI_CACHE_EXPIRY_HOURS = 24  # AI summaries valid for 24 hours


def get_cached_ai_summary(ticker: str) -> dict:
    """Get cached AI summary if it exists and is not expired"""
    try:
        if not os.path.exists(AI_SUMMARY_CACHE_FILE):
            return None

        with open(AI_SUMMARY_CACHE_FILE, "r") as f:
            cache = json.load(f)

        if ticker in cache:
            cached_time = datetime.fromisoformat(cache[ticker]["timestamp"])
            if datetime.now() - cached_time < timedelta(hours=AI_CACHE_EXPIRY_HOURS):
                print(f"Using cached AI summary for {ticker}")
                return cache[ticker]["summary"]

        return None
    except Exception as e:
        print(f"Error reading AI summary cache: {e}")
        return None


def save_ai_summary_to_cache(ticker: str, summary: str):
    """Save AI summary to cache"""
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        cache = {}
        if os.path.exists(AI_SUMMARY_CACHE_FILE):
            with open(AI_SUMMARY_CACHE_FILE, "r") as f:
                cache = json.load(f)

        cache[ticker] = {"summary": summary, "timestamp": datetime.now().isoformat()}

        with open(AI_SUMMARY_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving AI summary to cache: {e}")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/portfolio")
async def get_portfolio():
    try:
        portfolio_df, portfolio_summary = load_portfolio(FILE_PATH)
        portfolio_data = portfolio_df.to_dict(orient="records")

        # Ensure numerical values are properly formatted
        for item in portfolio_data:
            item["portfolio_weight"] = float(item["portfolio_weight"])
            if "industry" not in item or not item["industry"]:
                item["industry"] = "Other"

        # Convert NumPy types to native Python types
        portfolio_summary = {
            key: float(value) if isinstance(value, (np.float64, np.int64)) else value
            for key, value in portfolio_summary.items()
        }
        return {"portfolio": portfolio_data, "summary": portfolio_summary}
    except Exception as e:
        return {"error": str(e)}


@app.get("/news")
async def get_news():
    portfolio_df, _ = load_portfolio(FILE_PATH)
    news_data, cache_status = compile_news_briefing(portfolio_df, NEWS_API_KEY)
    return {"news": news_data, "cache_status": cache_status}


@app.get("/suggestions")
async def get_suggestions():
    portfolio_df, _ = load_portfolio(FILE_PATH)
    suggestions = generate_portfolio_suggestions(portfolio_df, None)
    return {"suggestions": suggestions}


def is_valid_ticker(ticker: str) -> bool:
    """Check if a ticker symbol is valid using yfinance"""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return "regularMarketPrice" in info
    except:
        return False


@app.get("/validate-ticker/{ticker}")
async def validate_ticker(ticker: str):
    """Endpoint to validate ticker symbols"""
    is_valid = is_valid_ticker(ticker)
    return {"valid": is_valid}


@app.get("/news-summary/{ticker}")
async def get_news_summary(ticker: str):
    try:
        # First validate the ticker
        if not is_valid_ticker(ticker):
            return {"summary": f"Invalid ticker symbol: {ticker}", "status": "error"}

        print(f"Getting news for ticker: {ticker}")

        # Get news data
        news_data = get_news_for_ticker(ticker, NEWS_API_KEY)
        news_links = "\n".join([article["url"] for article in news_data[:5]])

        if not news_data:
            return {
                "summary": f"No recent news found for {ticker}.",
                "headlines": [],
                "status": "no_news",
            }

        # Format the prompt
        prompt = f"""Summarize the latest financial news for the stock ticker {ticker} as of today. 
        Prioritize the most recent and relevant financial updates, using the provided links 
        as a starting point and supplementing with other reputable sources from web searches. 
        Analyze key developments, such as earnings reports, analyst ratings, macroeconomic 
        factors, and industry trends.

        News sources to consider:
        {news_links}

        Provide a data-driven recommendation on whether investors should consider buying, 
        holding, or selling this stock. Justify the recommendation with quantitative 
        and qualitative insights."""

        try:
            # Generate summary using OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst providing detailed stock analysis and recommendations.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=1000,
            )
            summary = response.choices[0].message.content

            return {
                "summary": summary,
                "headlines": [article["title"] for article in news_data[:5]],
                "prompt": prompt,
                "status": "success",
            }
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            print(f"Prompt: {prompt}")
            return {
                "summary": "Unable to generate summary at this time.",
                "headlines": [],
                "status": "error",
            }

    except Exception as e:
        print(f"Error in get_news_summary: {str(e)}")
        return {
            "summary": "An error occurred while generating the summary.",
            "headlines": [],
            "status": "error",
        }


@app.get("/latest-news/{ticker}")
async def get_latest_news(ticker: str):
    try:
        # First validate the ticker
        if not is_valid_ticker(ticker):
            return {"summary": f"Invalid ticker symbol: {ticker}", "status": "error"}

        # Get news data (this function already handles caching)
        news_data = get_news_for_ticker(ticker, NEWS_API_KEY)

        if not news_data:
            return {
                "summary": f"No recent news found for {ticker}.",
                "articles": [],
                "status": "no_news",
            }

        return {"articles": news_data, "status": "success"}

    except Exception as e:
        print(f"Error in get_latest_news: {str(e)}")
        return {
            "summary": "An error occurred while fetching news.",
            "articles": [],
            "status": "error",
        }


@app.get("/ai-summary/{ticker}")
async def get_ai_summary(ticker: str):
    try:
        # Check cache first
        cached_summary = get_cached_ai_summary(ticker)
        if cached_summary:
            return {"summary": cached_summary, "status": "success", "source": "cached"}

        # Get news articles
        news_data = get_news_for_ticker(ticker, NEWS_API_KEY)

        if not news_data:
            return {
                "summary": f"No recent news found for {ticker}.",
                "status": "no_news",
            }

        # Prepare news content for summarization
        news_text = "\n\n".join(
            [
                f"Title: {article['title']}\nDescription: {article['description']}"
                for article in news_data[:5]
            ]
        )

        try:
            # Generate summary using OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news analyst. Provide a concise summary of the following news articles about a stock.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize these news articles about {ticker}:\n\n{news_text}",
                    },
                ],
                max_tokens=250,
            )
            summary = response.choices[0].message.content

            # Cache the summary
            save_ai_summary_to_cache(ticker, summary)

            return {"summary": summary, "status": "success", "source": "fresh"}
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return {
                "summary": "Unable to generate AI summary at this time.",
                "status": "error",
            }

    except Exception as e:
        print(f"Error in get_ai_summary: {str(e)}")
        return {
            "summary": "An error occurred while generating the summary.",
            "status": "error",
        }


async def generate_streaming_response(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst providing stock analysis.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        for chunk in response:
            if chunk and chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            await asyncio.sleep(0.1)  # Small delay to control flow

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
