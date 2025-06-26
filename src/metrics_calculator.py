import polars as pl


def calculate_moving_average(
    df: pl.DataFrame, window: int = 30, column: str = "close"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .rolling_mean(window_size=window, min_samples=1)
        .alias(f"ma_{window}")
    )


def calculate_moving_median(
    df: pl.DataFrame, window: int = 30, column: str = "close"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(column)
        .rolling_median(window_size=window, min_samples=1)
        .alias(f"median_{window}")
    )


#  Volume Weighted Average Price (VWAP)
def calculate_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)

    Typical Price = (High + Low + Close) / 3
    """
    return (
        df.with_columns(
            [
                # Calculate typical price
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias(
                    "typical_price"
                )
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("typical_price") * pl.col("volume")).cum_sum()
                    / pl.col("volume").cum_sum()
                ).alias("vwap")
            ]
        )
        .drop("typical_price")
    )  # Remove temporary column


# Bollinger Bands
def calculate_bollinger_bands(
    df: pl.DataFrame, window: int = 20, num_std: float = 2.0
) -> pl.DataFrame:
    return (
        df.with_columns(
            [
                # Moving average (middle band)
                pl.col("close")
                .rolling_mean(window_size=window, min_samples=1)
                .alias("bb_middle"),
                # Standard deviation with null handling
                pl.col("close")
                .rolling_std(window_size=window, min_samples=1)
                .fill_null(0.0)
                .alias("bb_std"),
            ]
        )
        .with_columns(
            [
                # Upper band - use middle band if std is null/zero
                pl.when(pl.col("bb_std").is_null() | (pl.col("bb_std") == 0))
                .then(pl.col("bb_middle"))
                .otherwise(pl.col("bb_middle") + (pl.col("bb_std") * num_std))
                .alias("bb_upper"),
                # Lower band - use middle band if std is null/zero
                pl.when(pl.col("bb_std").is_null() | (pl.col("bb_std") == 0))
                .then(pl.col("bb_middle"))
                .otherwise(pl.col("bb_middle") - (pl.col("bb_std") * num_std))
                .alias("bb_lower"),
            ]
        )
        .drop("bb_std")
    )  # Remove temporary std column


# Relative Strength Index (RSI)
def calculate_rsi(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    return (
        df.with_columns(
            [
                # Price changes
                (pl.col("close") - pl.col("close").shift(1)).alias("price_change")
            ]
        )
        .with_columns(
            [
                # Separate gains and losses
                pl.when(pl.col("price_change") > 0)
                .then(pl.col("price_change"))
                .otherwise(0.0)
                .alias("gain"),
                pl.when(pl.col("price_change") < 0)
                .then(-pl.col("price_change"))
                .otherwise(0.0)
                .alias("loss"),
            ]
        )
        .with_columns(
            [
                # Average gains and losses
                pl.col("gain")
                .rolling_mean(window_size=window, min_samples=1)
                .alias("avg_gain"),
                pl.col("loss")
                .rolling_mean(window_size=window, min_samples=1)
                .alias("avg_loss"),
            ]
        )
        .with_columns(
            [
                # RSI calculation
                pl.when(pl.col("avg_loss") == 0)
                .then(100.0)
                .otherwise(
                    100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss"))))
                )
                .alias("rsi")
            ]
        )
        .drop(["price_change", "gain", "loss", "avg_gain", "avg_loss"])
    )


def add_comprehensive_metrics(df: pl.DataFrame, timeframe_name: str) -> pl.DataFrame:
    # Determine appropriate window sizes based on timeframe
    if "1min" in timeframe_name or "minute" in timeframe_name:
        ma_window = 30  # 30 minutes
        median_window = 30  # 30 minutes
        bb_window = 20  # 20 minutes
        rsi_window = 14  # 14 minutes
    elif "5min" in timeframe_name:
        ma_window = 6  # 30 minutes (6 * 5min)
        median_window = 6  # 30 minutes
        bb_window = 4  # 20 minutes (4 * 5min)
        rsi_window = 3  # ~15 minutes (3 * 5min)
    elif "30min" in timeframe_name:
        ma_window = 1  # 30 minutes (1 * 30min)
        median_window = 1  # 30 minutes
        bb_window = 1  # 30 minutes
        rsi_window = 1  # 30 minutes
    else:  # daily or other
        ma_window = 1
        median_window = 1
        bb_window = 1
        rsi_window = 1

    result = (
        df.pipe(calculate_moving_average, window=ma_window)
        .pipe(calculate_moving_median, window=median_window)
        .pipe(calculate_vwap)
        .pipe(calculate_bollinger_bands, window=bb_window)
        .pipe(calculate_rsi, window=rsi_window)
    )

    return result


def add_metrics_to_dataframes(
    df_1min: pl.DataFrame, df_5min: pl.DataFrame, df_30min: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print("Calculating financial metrics...")

    # Add metrics to each timeframe
    df_1min_metrics = add_comprehensive_metrics(df_1min, "1min")
    df_5min_metrics = add_comprehensive_metrics(df_5min, "5min")
    df_30min_metrics = add_comprehensive_metrics(df_30min, "30min")

    print("✓ Financial metrics calculated for all timeframes")

    return df_1min_metrics, df_5min_metrics, df_30min_metrics


def validate_metrics(df: pl.DataFrame, timeframe_name: str) -> bool:
    print(f"\nValidating {timeframe_name} metrics...")

    # Check for null values in key metrics
    key_metrics = [
        "ma_30",
        "median_30",
        "vwap",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "rsi",
    ]

    for metric in key_metrics:
        if metric in df.columns:
            null_count = df.select(pl.col(metric).is_null().sum()).item()
            if null_count > 0:
                print(f"✗ {metric} has {null_count} null values")
                return False

    # Check that VWAP is within reasonable range relative to price
    price_min = df["low"].min()
    price_max = df["high"].max()
    vwap_min = df["vwap"].min()
    vwap_max = df["vwap"].max()

    if vwap_min < price_min * 0.9 or vwap_max > price_max * 1.1:
        print(f"✗ VWAP outside reasonable range: {vwap_min:.2f} - {vwap_max:.2f}")
        return False

    # Check RSI is between 0 and 100
    rsi_min = df["rsi"].min()
    rsi_max = df["rsi"].max()

    if rsi_min < 0 or rsi_max > 100:
        print(f"✗ RSI outside valid range [0,100]: {rsi_min:.2f} - {rsi_max:.2f}")
        return False

    # Check Bollinger Bands ordering
    bb_violations = df.select(
        (pl.col("bb_lower") > pl.col("bb_middle")).sum().alias("lower_gt_middle"),
        (pl.col("bb_middle") > pl.col("bb_upper")).sum().alias("middle_gt_upper"),
    )

    if (
        bb_violations["lower_gt_middle"].item() > 0
        or bb_violations["middle_gt_upper"].item() > 0
    ):
        print("✗ Bollinger Bands ordering violated")
        return False

    print(f"✓ All {timeframe_name} metrics validated successfully")
    return True


def get_metrics_summary(df: pl.DataFrame, timeframe_name: str) -> None:
    print(f"\n{timeframe_name} Metrics Summary:")

    if "ma_30" in df.columns:
        print(f"MA(30): ${df['ma_30'].mean():.2f} (avg)")
    if "vwap" in df.columns:
        print(f"VWAP: ${df['vwap'].mean():.2f} (avg)")
    if "rsi" in df.columns:
        print(
            f"RSI: {df['rsi'].mean():.1f} (avg), range {df['rsi'].min():.1f}-{df['rsi'].max():.1f}"
        )
    if "bb_middle" in df.columns:
        bb_width = (df["bb_upper"] - df["bb_lower"]).mean()
        print(f"Bollinger Band Width: ${bb_width:.3f} (avg)")
