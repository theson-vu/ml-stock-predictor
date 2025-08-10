# src/data_fetch.py
import os
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load .env file
load_dotenv()

def make_client():
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    return tradeapi.REST(api_key, api_secret, base_url)


def fetch_bars(symbol: str, start: str, end: str, timeframe: str = "1Day", client=None) -> pd.DataFrame:
    """
    Returns DataFrame with datetime index and columns including 'open','high','low','close','volume'.
    client can be injected for tests.
    """
    client = client or make_client()
    bars = client.get_bars(symbol, timeframe, start=start, end=end).df
    # normalize returned frame to simple column names if needed
    bars.index = pd.to_datetime(bars.index)
    return bars