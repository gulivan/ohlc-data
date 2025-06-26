"""
Comprehensive unit tests for the financial metrics calculator module.

These tests verify the correctness of financial metrics calculation, ensuring that:
1. Moving averages are calculated correctly
2. VWAP calculation is accurate
3. RSI values are within valid ranges
4. Bollinger Bands are properly computed
5. All metrics are added to dataframes correctly
"""

import os
import sys
import unittest

import numpy as np
import polars as pl

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_aggregator import aggregate_to_multiple_timeframes
from src.data_generator import generate_minute_data
from src.metrics_calculator import (
    add_metrics_to_dataframes,
    calculate_bollinger_bands,
    calculate_moving_average,
    calculate_moving_median,
    calculate_rsi,
    calculate_vwap,
    validate_metrics,
)


class TestMetricsCalculator(unittest.TestCase):
    """Test suite for the financial metrics calculator."""

    def setUp(self):
        """Set up test data."""
        self.df_1min = generate_minute_data(seed=42)
        self.df_5min, self.df_30min, self.df_daily = aggregate_to_multiple_timeframes(
            self.df_1min
        )

    def test_moving_average_calculation(self):
        """Test moving average calculation accuracy."""
        # Test with 5-minute data and 3-period window
        df_with_ma = calculate_moving_average(self.df_5min, window=3)

        # Manually calculate moving average for first few values
        closes = self.df_5min["close"].to_numpy()

        # First value should equal the close (window of 1)
        self.assertAlmostEqual(
            df_with_ma["ma_3"][0],
            closes[0],
            places=10,
            msg="First MA value should equal first close",
        )

        # Second value should be average of first two
        expected_ma_2 = (closes[0] + closes[1]) / 2
        self.assertAlmostEqual(
            df_with_ma["ma_3"][1],
            expected_ma_2,
            places=10,
            msg="Second MA value should be average of first two closes",
        )

        # Third value should be average of first three
        expected_ma_3 = (closes[0] + closes[1] + closes[2]) / 3
        self.assertAlmostEqual(
            df_with_ma["ma_3"][2],
            expected_ma_3,
            places=10,
            msg="Third MA value should be average of first three closes",
        )

        # Check that MA values are reasonable
        self.assertTrue(
            (df_with_ma["ma_3"] > 0).all(), "All MA values should be positive"
        )

    def test_moving_median_calculation(self):
        """Test moving median calculation accuracy."""
        # Test with 5-minute data and 3-period window
        df_with_median = calculate_moving_median(self.df_5min, window=3)

        # Manually calculate moving median for first few values
        closes = self.df_5min["close"].to_numpy()

        # First value should equal the close
        self.assertAlmostEqual(
            df_with_median["median_3"][0],
            closes[0],
            places=10,
            msg="First median value should equal first close",
        )

        # Second value should be median of first two (average since we have 2 values)
        expected_median_2 = np.median([closes[0], closes[1]])
        self.assertAlmostEqual(
            df_with_median["median_3"][1],
            expected_median_2,
            places=10,
            msg="Second median value should be median of first two closes",
        )

        # Third value should be median of first three
        expected_median_3 = np.median([closes[0], closes[1], closes[2]])
        self.assertAlmostEqual(
            df_with_median["median_3"][2],
            expected_median_3,
            places=10,
            msg="Third median value should be median of first three closes",
        )

    def test_vwap_calculation(self):
        """Test VWAP calculation accuracy."""
        df_with_vwap = calculate_vwap(self.df_5min)

        # Manually calculate VWAP for first few values
        highs = self.df_5min["high"].to_numpy()
        lows = self.df_5min["low"].to_numpy()
        closes = self.df_5min["close"].to_numpy()
        volumes = self.df_5min["volume"].to_numpy()

        # Calculate typical prices
        typical_prices = (highs + lows + closes) / 3

        # Calculate cumulative VWAP manually
        cum_tp_volume = np.cumsum(typical_prices * volumes)
        cum_volume = np.cumsum(volumes)
        manual_vwap = cum_tp_volume / cum_volume

        # Compare with calculated VWAP
        calculated_vwap = df_with_vwap["vwap"].to_numpy()

        np.testing.assert_allclose(
            calculated_vwap,
            manual_vwap,
            rtol=1e-10,
            err_msg="VWAP calculation should match manual calculation",
        )

        # VWAP should be within reasonable range of prices
        opens = self.df_5min["open"].to_numpy()
        min_price = min(lows.min(), opens.min(), closes.min())
        max_price = max(highs.max(), opens.max(), closes.max())

        self.assertTrue(
            (df_with_vwap["vwap"] >= min_price * 0.99).all(),
            "VWAP should be near the price range (lower bound)",
        )
        self.assertTrue(
            (df_with_vwap["vwap"] <= max_price * 1.01).all(),
            "VWAP should be near the price range (upper bound)",
        )

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        df_with_bb = calculate_bollinger_bands(self.df_5min, window=5, num_std=2.0)

        # Check that all Bollinger Band columns exist
        required_columns = ["bb_middle", "bb_upper", "bb_lower"]
        for col in required_columns:
            self.assertIn(col, df_with_bb.columns, f"Column {col} should exist")

        # Check Bollinger Band relationships
        self.assertTrue(
            (df_with_bb["bb_upper"] >= df_with_bb["bb_middle"]).all(),
            "Upper band should be >= middle band",
        )
        self.assertTrue(
            (df_with_bb["bb_middle"] >= df_with_bb["bb_lower"]).all(),
            "Middle band should be >= lower band",
        )

        # For the first value, bands should be equal (no standard deviation)
        first_row = df_with_bb.row(0, named=True)
        self.assertAlmostEqual(
            first_row["bb_upper"],
            first_row["bb_middle"],
            places=5,
            msg="First upper band should equal middle band",
        )
        self.assertAlmostEqual(
            first_row["bb_lower"],
            first_row["bb_middle"],
            places=5,
            msg="First lower band should equal middle band",
        )

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        df_with_rsi = calculate_rsi(self.df_5min, window=14)

        # Check that RSI column exists
        self.assertIn("rsi", df_with_rsi.columns, "RSI column should exist")

        # RSI should be between 0 and 100
        self.assertTrue((df_with_rsi["rsi"] >= 0).all(), "RSI should be >= 0")
        self.assertTrue((df_with_rsi["rsi"] <= 100).all(), "RSI should be <= 100")

        # First RSI value should be extreme (either 0 or 100) due to single price change
        first_rsi = df_with_rsi["rsi"][0]
        self.assertTrue(
            first_rsi == 0.0 or first_rsi == 100.0,
            f"First RSI should be 0 or 100, got {first_rsi}",
        )

    def test_add_metrics_to_dataframes(self):
        """Test that metrics are correctly added to all dataframes."""
        df_1min_with_metrics, df_5min_with_metrics, df_30min_with_metrics = (
            add_metrics_to_dataframes(self.df_1min, self.df_5min, self.df_30min)
        )

        # Check that essential metrics columns exist in all dataframes
        essential_metrics = ["vwap", "rsi"]

        for df, name in [
            (df_5min_with_metrics, "5-minute"),
            (df_30min_with_metrics, "30-minute"),
        ]:
            for metric in essential_metrics:
                self.assertIn(
                    metric, df.columns, f"{metric} should exist in {name} data"
                )

        # Check that moving averages exist
        ma_columns = [
            col for col in df_5min_with_metrics.columns if col.startswith("ma_")
        ]
        self.assertGreater(
            len(ma_columns), 0, "Should have at least one moving average column"
        )

        # Check that Bollinger Bands exist
        bb_columns = [
            col for col in df_5min_with_metrics.columns if col.startswith("bb_")
        ]
        self.assertGreaterEqual(
            len(bb_columns), 3, "Should have at least 3 Bollinger Band columns"
        )

    def test_metrics_validation(self):
        """Test the metrics validation function."""
        df_1min_with_metrics, df_5min_with_metrics, df_30min_with_metrics = (
            add_metrics_to_dataframes(self.df_1min, self.df_5min, self.df_30min)
        )

        # All metrics should pass validation
        self.assertTrue(
            validate_metrics(df_5min_with_metrics, "5-minute"),
            "5-minute metrics should pass validation",
        )
        self.assertTrue(
            validate_metrics(df_30min_with_metrics, "30-minute"),
            "30-minute metrics should pass validation",
        )

    def test_metrics_no_null_values(self):
        """Test that metrics don't contain null values."""
        df_1min_with_metrics, df_5min_with_metrics, df_30min_with_metrics = (
            add_metrics_to_dataframes(self.df_1min, self.df_5min, self.df_30min)
        )

        # Check for null values in key metrics
        key_metrics = ["vwap", "rsi"]

        for df, name in [
            (df_5min_with_metrics, "5-minute"),
            (df_30min_with_metrics, "30-minute"),
        ]:
            for metric in key_metrics:
                if metric in df.columns:
                    null_count = df.select(pl.col(metric).is_null().sum()).item()
                    self.assertEqual(
                        null_count,
                        0,
                        f"{metric} in {name} data should not have null values",
                    )

    def test_vwap_price_relationship(self):
        """Test that VWAP is within reasonable bounds relative to prices."""
        df_with_vwap = calculate_vwap(self.df_5min)

        # VWAP should generally be between the low and high of each period
        # (though it's a cumulative measure, so this isn't always strict)

        # At minimum, VWAP should be within the overall price range
        min_low = self.df_5min["low"].min()
        max_high = self.df_5min["high"].max()

        vwap_min = df_with_vwap["vwap"].min()
        vwap_max = df_with_vwap["vwap"].max()

        self.assertGreaterEqual(
            vwap_min, min_low * 0.99, "VWAP minimum should be near the overall low"
        )
        self.assertLessEqual(
            vwap_max, max_high * 1.01, "VWAP maximum should be near the overall high"
        )

    def test_moving_average_smoothness(self):
        """Test that moving averages are smoother than the underlying price."""
        df_with_ma = calculate_moving_average(self.df_5min, window=10)

        # Calculate volatility of close prices
        close_returns = self.df_5min.select(
            (pl.col("close").pct_change()).alias("returns")
        )["returns"].drop_nulls()
        close_volatility = close_returns.std()

        # Calculate volatility of moving average
        ma_returns = df_with_ma.select(
            (pl.col("ma_10").pct_change()).alias("ma_returns")
        )["ma_returns"].drop_nulls()
        ma_volatility = ma_returns.std()

        # Moving average should be less volatile than close prices
        self.assertLess(
            ma_volatility,
            close_volatility,
            "Moving average should be less volatile than close prices",
        )

    def test_rsi_extreme_values(self):
        """Test RSI behavior in extreme market conditions."""
        # Create artificial data with strong uptrend
        artificial_data = pl.DataFrame(
            {
                "timestamp": self.df_5min["timestamp"],
                "open": [100 + i for i in range(len(self.df_5min))],
                "high": [101 + i for i in range(len(self.df_5min))],
                "low": [99 + i for i in range(len(self.df_5min))],
                "close": [100.5 + i for i in range(len(self.df_5min))],
                "volume": [1000] * len(self.df_5min),
            }
        )

        df_with_rsi = calculate_rsi(artificial_data, window=14)

        # In a strong uptrend, RSI should tend towards higher values
        # (though exact values depend on the implementation)
        final_rsi = df_with_rsi["rsi"][-1]
        self.assertGreater(final_rsi, 50, "RSI should be > 50 in a consistent uptrend")

    def test_comprehensive_metrics_pipeline(self):
        """Test the complete metrics calculation pipeline."""
        # Start with original data
        original_1min = self.df_1min
        original_5min = self.df_5min
        original_30min = self.df_30min

        # Add metrics
        enhanced_1min, enhanced_5min, enhanced_30min = add_metrics_to_dataframes(
            original_1min, original_5min, original_30min
        )

        # Verify that original OHLCV data is preserved
        for orig_col in ["timestamp", "open", "high", "low", "close", "volume"]:
            self.assertTrue(
                enhanced_5min[orig_col].equals(original_5min[orig_col]),
                f"Original {orig_col} data should be preserved",
            )

        # Verify that new metrics columns are added
        original_columns = set(original_5min.columns)
        enhanced_columns = set(enhanced_5min.columns)
        new_columns = enhanced_columns - original_columns

        self.assertGreater(len(new_columns), 0, "Should have added new metrics columns")

        # Verify data integrity after metrics addition
        self.assertEqual(
            len(enhanced_5min), len(original_5min), "Row count should be preserved"
        )


if __name__ == "__main__":
    unittest.main(verbose=2)
