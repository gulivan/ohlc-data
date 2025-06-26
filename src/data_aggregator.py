import polars as pl


def aggregate_ohlc(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Aggregate OHLC data.

    Args:
        df: DataFrame with timestamp, open, high, low, close, volume columns
        timeframe: String specifying the timeframe ('5m', '30m', '1d')

    Returns:
        polars.DataFrame: Aggregated OHLC data
    """
    if df["timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    aggregated = df.group_by_dynamic(
        "timestamp",
        every=timeframe,
        closed="left",
        label="left",
    ).agg(
        [
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )

    aggregated = aggregated.filter(
        pl.col("open").is_not_null()
        & pl.col("high").is_not_null()
        & pl.col("low").is_not_null()
        & pl.col("close").is_not_null()
    )

    return aggregated


def aggregate_to_multiple_timeframes(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Folding 1-minute OHLC data to 5-minute, 30-minute, and daily timeframes.
    Args:
        df: DataFrame with 1-minute OHLC data
    Returns:
        tuple: (5min_df, 30min_df, daily_df)
    """
    df_5min = aggregate_ohlc(df, "5m")
    df_30min = aggregate_ohlc(df, "30m")
    df_daily = aggregate_ohlc(df, "1d")

    return df_5min, df_30min, df_daily


def validate_aggregated_data(
    original_df: pl.DataFrame,
    aggregated_df: pl.DataFrame,
    expected_bars: int,
    timeframe_name: str,
) -> bool:
    print(f"\nValidating {timeframe_name} aggregation...")

    actual_bars = len(aggregated_df)
    if actual_bars != expected_bars:
        print(f"✗ Expected {expected_bars} bars, got {actual_bars}")
        return False
    print(f"✓ Bar count correct: {actual_bars}")

    ohlc_valid = aggregated_df.select(
        [
            (pl.col("high") >= pl.col("open")).alias("high_ge_open"),
            (pl.col("high") >= pl.col("close")).alias("high_ge_close"),
            (pl.col("high") >= pl.col("low")).alias("high_ge_low"),
            (pl.col("low") <= pl.col("open")).alias("low_le_open"),
            (pl.col("low") <= pl.col("close")).alias("low_le_close"),
            (pl.col("open") > 0).alias("open_positive"),
            (pl.col("high") > 0).alias("high_positive"),
            (pl.col("low") > 0).alias("low_positive"),
            (pl.col("close") > 0).alias("close_positive"),
            (pl.col("volume") >= 0).alias("volume_non_negative"),
        ]
    )

    constraints = [
        "high_ge_open",
        "high_ge_close",
        "high_ge_low",
        "low_le_open",
        "low_le_close",
        "open_positive",
        "high_positive",
        "low_positive",
        "close_positive",
        "volume_non_negative",
    ]

    for constraint in constraints:
        if not ohlc_valid.select(pl.col(constraint)).to_series().all():
            print(f"✗ OHLC constraint violated: {constraint}")
            return False

    print("✓ All OHLC constraints satisfied")

    original_volume = original_df.select(pl.col("volume").sum()).item()
    aggregated_volume = aggregated_df.select(pl.col("volume").sum()).item()

    if abs(original_volume - aggregated_volume) > 1:  # Allow for small rounding
        print(f"✗ Volume not conserved: {original_volume} vs {aggregated_volume}")
        return False
    print(f"✓ Volume conserved: {aggregated_volume:,}")

    return True


# print stats
def get_aggregation_stats(df: pl.DataFrame, timeframe_name: str) -> None:
    print(f"\n{timeframe_name} Statistics:")
    print(f"Number of bars: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Total volume: {df['volume'].sum():,}")
    print(f"Average volume per bar: {df['volume'].mean():.0f}")
