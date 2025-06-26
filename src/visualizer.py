import os
from datetime import datetime, timedelta

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Rectangle

# headless mode
try:
    matplotlib.use("Agg")
except Exception:
    pass

# styling
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("default")
sns.set_palette("husl")


def plot_candlestick_chart(
    df: pl.DataFrame,
    title: str = "OHLC Candlestick Chart",
    figsize: tuple[int, int] = (15, 8),
    volume: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    df_pd = df.to_pandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"])

    if volume:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    colors = [
        "green" if close >= open else "red"
        for close, open in zip(df_pd["close"], df_pd["open"], strict=False)
    ]

    time_diff = df_pd["timestamp"].iloc[1] - df_pd["timestamp"].iloc[0]
    width_timedelta = timedelta(seconds=time_diff.total_seconds() * 0.4)

    for i, (_idx, row) in enumerate(df_pd.iterrows()):
        timestamp = row["timestamp"]
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        color = colors[i]

        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)

        rect = Rectangle(
            (
                timestamp - width_timedelta,
                body_bottom,
            ),
            width_timedelta * 2,
            body_height,
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
        )
        ax1.add_patch(rect)

        ax1.plot(
            [timestamp, timestamp],
            [low_price, high_price],
            color="black",
            linewidth=1,
            alpha=0.8,
        )

    ax1.set_title(title, fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if volume and ax2 is not None:
        colors_volume = [
            "green" if close >= open else "red"
            for close, open in zip(df_pd["close"], df_pd["open"], strict=False)
        ]
        ax2.bar(
            df_pd["timestamp"],
            df_pd["volume"],
            color=colors_volume,
            alpha=0.6,
            width=width_timedelta * 2,
        )
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.grid(True, alpha=0.3)

    if ax2 is not None:
        ax2.set_xlabel("Time", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax1.set_xlabel("Time", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig, (ax1, ax2) if volume else (fig, ax1)


def plot_price_with_indicators(
    df: pl.DataFrame,
    indicators: list[str] = None,
    title: str = "Price with Technical Indicators",
    figsize: tuple[int, int] = (15, 10),
    indicator_limit: int = 5,
) -> tuple[plt.Figure, plt.Axes]:
    if indicators is None:
        indicator_cols = [
            col
            for col in df.columns
            if col not in ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        indicators = indicator_cols[:indicator_limit]

    # Pandas for Matplotlib compatibility.
    df_pd = df.to_pandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
    )

    ax1.plot(
        df_pd["timestamp"],
        df_pd["close"],
        label="Close Price",
        color="black",
        linewidth=2,
        alpha=0.8,
    )

    # Plot indicators on price chart
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for indicator in indicators:
        if indicator in df_pd.columns and indicator != "rsi":
            if "bb_" in indicator:  # Bollinger Bands
                if indicator == "bb_middle":
                    ax1.plot(
                        df_pd["timestamp"],
                        df_pd[indicator],
                        label="BB Middle",
                        color=colors[color_idx],
                        alpha=0.7,
                    )
                    color_idx += 1
                elif indicator == "bb_upper":
                    ax1.plot(
                        df_pd["timestamp"],
                        df_pd[indicator],
                        label="BB Upper",
                        color="blue",
                        alpha=0.5,
                        linestyle="--",
                    )
                elif indicator == "bb_lower":
                    ax1.plot(
                        df_pd["timestamp"],
                        df_pd[indicator],
                        label="BB Lower",
                        color="blue",
                        alpha=0.5,
                        linestyle="--",
                    )
                    if "bb_upper" in df_pd.columns:
                        ax1.fill_between(
                            df_pd["timestamp"],
                            df_pd["bb_lower"],
                            df_pd["bb_upper"],
                            alpha=0.1,
                            color="blue",
                            label="BB Band",
                        )
            else:
                ax1.plot(
                    df_pd["timestamp"],
                    df_pd[indicator],
                    label=indicator.upper(),
                    color=colors[color_idx % len(colors)],
                    alpha=0.7,
                )
                color_idx += 1

    ax1.set_title(title, fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot RSI if available
    if "rsi" in df_pd.columns:
        ax2.plot(
            df_pd["timestamp"], df_pd["rsi"], label="RSI", color="purple", linewidth=2
        )
        ax2.axhline(
            y=70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)"
        )
        ax2.axhline(
            y=30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)"
        )
        ax2.axhline(y=50, color="gray", linestyle="-", alpha=0.5)
        ax2.fill_between(df_pd["timestamp"], 70, 100, alpha=0.1, color="red")
        ax2.fill_between(df_pd["timestamp"], 0, 30, alpha=0.1, color="green")
        ax2.set_ylabel("RSI", fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_correlation_heatmap(
    df: pl.DataFrame,
    title: str = "Price and Metrics Correlation",
    figsize: tuple[int, int] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in ["timestamp"] and df[col].dtype in [pl.Float64, pl.Int64]
    ]

    df_numeric = df.select(numeric_cols).to_pandas()
    correlation_matrix = df_numeric.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig, ax


def plot_returns_analysis(
    df: pl.DataFrame,
    title: str = "Returns Analysis",
    figsize: tuple[int, int] = (15, 10),
) -> tuple[plt.Figure, plt.Axes]:
    df_with_returns = df.with_columns(
        [
            (pl.col("close").pct_change()).alias("returns"),
            (pl.col("close").pct_change().rolling_std(window_size=20)).alias(
                "volatility"
            ),
        ]
    )

    df_pd = df_with_returns.to_pandas()
    df_pd["timestamp"] = pd.to_datetime(df_pd["timestamp"])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax1.plot(df_pd["timestamp"], df_pd["returns"], color="blue", alpha=0.7)
    ax1.set_title("Returns Over Time")
    ax1.set_ylabel("Returns")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    returns_clean = df_pd["returns"].dropna()
    ax2.hist(returns_clean, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
    ax2.axvline(
        returns_clean.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {returns_clean.mean():.4f}",
    )
    ax2.axvline(
        returns_clean.median(),
        color="green",
        linestyle="--",
        label=f"Median: {returns_clean.median():.4f}",
    )
    ax2.set_title("Returns Distribution")
    ax2.set_xlabel("Returns")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Rolling volatility
    ax3.plot(df_pd["timestamp"], df_pd["volatility"], color="red", alpha=0.7)
    ax3.set_title("Rolling Volatility (20-period)")
    ax3.set_ylabel("Volatility")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # Q-Q plot for normality check
    from scipy import stats

    stats.probplot(returns_clean, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot (Normality Check)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)


# image report
def create_comprehensive_report(
    df_1min: pl.DataFrame,
    df_5min: pl.DataFrame,
    df_30min: pl.DataFrame,
    df_daily: pl.DataFrame,
    output_dir: str = "./reports",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Generating comprehensive analysis report...")

    # 5-minute candlestick chart
    fig1, _ = plot_candlestick_chart(df_5min, "5-Minute OHLC with Volume", volume=True)
    fig1.savefig(f"{output_dir}/5min_candlestick.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 30-minute candlestick chart
    fig2, _ = plot_candlestick_chart(
        df_30min, "30-Minute OHLC with Volume", volume=True
    )
    fig2.savefig(f"{output_dir}/30min_candlestick.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Price with technical indicators (5-min)
    fig3, _ = plot_price_with_indicators(
        df_5min, title="5-Minute Price with Technical Indicators"
    )
    fig3.savefig(f"{output_dir}/5min_indicators.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # Price with technical indicators (30-min)
    fig4, _ = plot_price_with_indicators(
        df_30min, title="30-Minute Price with Technical Indicators"
    )
    fig4.savefig(f"{output_dir}/30min_indicators.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # Correlation heatmap (5-min data)
    fig5, _ = plot_correlation_heatmap(df_5min, "5-Minute Data Correlation Matrix")
    fig5.savefig(f"{output_dir}/5min_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # Returns analysis (1-min data)
    fig6, _ = plot_returns_analysis(df_1min, "1-Minute Returns Analysis")
    fig6.savefig(f"{output_dir}/returns_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig6)

    # Summary statistics report
    generate_statistics_report(df_1min, df_5min, df_30min, df_daily, output_dir)

    print(f"✓ Comprehensive report generated in {output_dir}/")
    print("  - Candlestick charts: 5min_candlestick.png, 30min_candlestick.png")
    print("  - Technical indicators: 5min_indicators.png, 30min_indicators.png")
    print("  - Correlation analysis: 5min_correlation.png")
    print("  - Returns analysis: returns_analysis.png")
    print("  - Statistics report: statistics_report.txt")


# text report
def generate_statistics_report(
    df_1min: pl.DataFrame,
    df_5min: pl.DataFrame,
    df_30min: pl.DataFrame,
    df_daily: pl.DataFrame,
    output_dir: str,
) -> None:
    report_path = f"{output_dir}/statistics_report.txt"

    with open(report_path, "w") as f:
        f.write("OHLC DATA PROCESSING - STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"1-minute bars: {len(df_1min):,}\n")
        f.write(f"5-minute bars: {len(df_5min):,}\n")
        f.write(f"30-minute bars: {len(df_30min):,}\n")
        f.write(f"Daily bars: {len(df_daily):,}\n\n")

        # Price statistics
        f.write("PRICE STATISTICS (1-MINUTE DATA)\n")
        f.write("-" * 35 + "\n")
        price_stats = df_1min.select(["open", "high", "low", "close"]).describe()
        f.write(str(price_stats) + "\n\n")

        # Volume statistics
        f.write("VOLUME STATISTICS\n")
        f.write("-" * 18 + "\n")
        volume_stats = df_1min.select(["volume"]).describe()
        f.write(str(volume_stats) + "\n\n")

        # Calculate and write returns statistics
        df_with_returns = df_1min.with_columns(
            (pl.col("close").pct_change()).alias("returns")
        )
        returns_stats = df_with_returns.select(["returns"]).describe()
        f.write("RETURNS STATISTICS\n")
        f.write("-" * 19 + "\n")
        f.write(str(returns_stats) + "\n\n")

        # Metrics summary (if available)
        if "vwap" in df_5min.columns:
            f.write("TECHNICAL INDICATORS SUMMARY (5-MINUTE DATA)\n")
            f.write("-" * 44 + "\n")

            metrics_cols = [
                col
                for col in df_5min.columns
                if col not in ["timestamp", "open", "high", "low", "close", "volume"]
            ]

            if metrics_cols:
                metrics_stats = df_5min.select(metrics_cols).describe()
                f.write(str(metrics_stats) + "\n\n")

        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")


def interactive_analysis_demo() -> None:
    """
    Demonstrate interactive analysis capabilities.
    """
    from src.data_aggregator import aggregate_to_multiple_timeframes
    from src.data_generator import generate_minute_data
    from src.metrics_calculator import add_metrics_to_dataframes

    print("Running interactive visualization demo...")

    # Generate sample data
    df_1min = generate_minute_data(seed=42)
    df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
    df_1min, df_5min, df_30min = add_metrics_to_dataframes(df_1min, df_5min, df_30min)

    # Create comprehensive report
    create_comprehensive_report(df_1min, df_5min, df_30min, df_daily)

    print("✓ Interactive analysis demo complete!")
    print("Check the ./reports directory for generated visualizations.")


if __name__ == "__main__":
    interactive_analysis_demo()
