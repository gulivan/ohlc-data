from datetime import datetime, timedelta

import numpy as np
import polars as pl


def generate_minute_data(
    seed: int | None = None,
    volatility: float = 0.001,
    initial_price: float = 100.0,
    trading_date: datetime | None = None,
) -> pl.DataFrame:
    """
    Generate 1-minute OHLC data for a single trading day (9:30 AM to 4:00 PM).
    Input:
        seed: Random seed.
        volatility: Price volatility parameter (standard deviation of returns)
        initial_price: Starting price for the day
        trading_date: Date for the trading session (defaults to today)
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate input parameters
    if volatility < 0:
        raise ValueError("Volatility must be non-negative")
    if initial_price <= 0:
        raise ValueError("Initial price must be positive")

    # Timestamp range for trading hours (9:30 AM to 4:00 PM = 390 minutes)
    if trading_date is None:
        trading_date = datetime.now().date()

    start_time = datetime.combine(
        trading_date, datetime.min.time().replace(hour=9, minute=30)
    )
    timestamps = [start_time + timedelta(minutes=i) for i in range(390)]

    dt = 1 / 390
    drift = 0.0001

    # Random shocks for each minute
    random_shocks = np.random.normal(0, volatility, 390)

    opens = np.zeros(390)
    highs = np.zeros(390)
    lows = np.zeros(390)
    closes = np.zeros(390)
    volumes = np.random.randint(
        1000, 10000, 390
    )  # Random volume between 1k and 10k (could be customized)

    current_price = initial_price

    for i in range(390):
        open_price = current_price
        opens[i] = open_price

        intra_minute_returns = np.random.normal(0, volatility * 0.5, 4)

        minute_return = drift * dt + random_shocks[i] * np.sqrt(dt)

        intra_prices = [open_price]
        temp_price = open_price

        for j in range(4):
            temp_price *= 1 + intra_minute_returns[j]
            intra_prices.append(temp_price)

        close_price = open_price * (1 + minute_return)
        intra_prices.append(close_price)

        all_prices = np.array(intra_prices)
        highs[i] = np.max(all_prices)
        lows[i] = np.min(all_prices)
        closes[i] = close_price
        current_price = close_price

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    # Ensure data integrity
    df = df.with_columns(
        [
            pl.when(pl.col("high") < pl.max_horizontal(["open", "close"]))
            .then(pl.max_horizontal(["open", "close"]))
            .otherwise(pl.col("high"))
            .alias("high"),
            pl.when(pl.col("low") > pl.min_horizontal(["open", "close"]))
            .then(pl.min_horizontal(["open", "close"]))
            .otherwise(pl.col("low"))
            .alias("low"),
        ]
    )

    return df


def validate_ohlc_data(df: pl.DataFrame) -> bool:
    """Data validation"""
    high_valid = (
        df.select(
            (pl.col("high") >= pl.col("open"))
            & (pl.col("high") >= pl.col("close"))
            & (pl.col("high") >= pl.col("low"))
        )
        .to_series()
        .all()
    )

    low_valid = (
        df.select(
            (pl.col("low") <= pl.col("open"))
            & (pl.col("low") <= pl.col("close"))
            & (pl.col("low") <= pl.col("high"))
        )
        .to_series()
        .all()
    )

    positive_prices = (
        df.select(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
        )
        .to_series()
        .all()
    )

    return high_valid and low_valid and positive_prices
