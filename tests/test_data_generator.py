"""
Comprehensive unit tests for the OHLC data generator module.

These tests verify the correctness of data generation, ensuring that:
1. Correct number of bars are generated
2. Timestamps follow proper sequence
3. OHLC price relationships are maintained
4. Reproducibility with seeds works correctly
"""

import os
import sys
import unittest
from datetime import datetime

import polars as pl

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_minute_data, validate_ohlc_data


class TestDataGenerator(unittest.TestCase):
    """Test suite for the OHLC data generator."""

    def test_data_length(self):
        """Test that the generator produces exactly 390 data points."""
        df = generate_minute_data(seed=42)
        self.assertEqual(
            len(df), 390, "Should generate 390 minute bars for a trading day"
        )

    def test_timestamp_sequence(self):
        """Test that timestamps are correctly sequenced from 9:30 AM to 4:00 PM."""
        df = generate_minute_data(seed=42)

        # Check that timestamps start at 9:30 AM
        first_time = df["timestamp"][0]
        self.assertEqual(first_time.hour, 9, "First timestamp should be at 9 AM hour")
        self.assertEqual(
            first_time.minute, 30, "First timestamp should be at 30 minutes"
        )

        # Check that timestamps end at 3:59 PM (last minute of trading)
        last_time = df["timestamp"][-1]
        self.assertEqual(
            last_time.hour, 15, "Last timestamp should be at 3 PM hour (15)"
        )
        self.assertEqual(last_time.minute, 59, "Last timestamp should be at 59 minutes")

        # Check that timestamps are 1 minute apart
        time_diffs = df.select(
            (pl.col("timestamp").diff().dt.total_seconds()).alias("diff")
        )["diff"].drop_nulls()

        for diff in time_diffs:
            self.assertEqual(
                diff, 60.0, f"Timestamps should be 60 seconds apart, got {diff}"
            )

    def test_price_integrity(self):
        """Test that OHLC prices maintain proper relationships."""
        df = generate_minute_data(seed=42)

        # High should be >= Open and Close
        high_ge_open = (df["high"] >= df["open"]).all()
        high_ge_close = (df["high"] >= df["close"]).all()
        self.assertTrue(high_ge_open, "High should be >= Open for all bars")
        self.assertTrue(high_ge_close, "High should be >= Close for all bars")

        # Low should be <= Open and Close
        low_le_open = (df["low"] <= df["open"]).all()
        low_le_close = (df["low"] <= df["close"]).all()
        self.assertTrue(low_le_open, "Low should be <= Open for all bars")
        self.assertTrue(low_le_close, "Low should be <= Close for all bars")

        # All prices should be positive
        self.assertTrue((df["open"] > 0).all(), "Open prices should be positive")
        self.assertTrue((df["high"] > 0).all(), "High prices should be positive")
        self.assertTrue((df["low"] > 0).all(), "Low prices should be positive")
        self.assertTrue((df["close"] > 0).all(), "Close prices should be positive")

        # Volume should be positive
        self.assertTrue((df["volume"] > 0).all(), "Volume should be positive")

    def test_reproducibility(self):
        """Test that setting the same seed produces the same data."""
        df1 = generate_minute_data(seed=42)
        df2 = generate_minute_data(seed=42)

        # Check that DataFrames are identical
        self.assertTrue(df1.equals(df2), "Same seed should produce identical data")

        # Test that different seeds produce different data
        df3 = generate_minute_data(seed=43)
        self.assertFalse(
            df1.equals(df3), "Different seeds should produce different data"
        )

    def test_price_continuity(self):
        """Test that prices show reasonable continuity between bars."""
        df = generate_minute_data(seed=42, volatility=0.001)

        # Calculate returns
        returns = df.select((pl.col("close").pct_change()).alias("returns"))[
            "returns"
        ].drop_nulls()

        # Returns should be reasonable (not too extreme)
        abs_returns = returns.abs()
        max_return = abs_returns.max()

        # With 0.1% volatility, returns should rarely exceed 1%
        self.assertLess(
            max_return, 0.01, "Returns should be reasonable given volatility"
        )

        # Mean return should be close to zero
        mean_return = returns.mean()
        self.assertLess(abs(mean_return), 0.001, "Mean return should be close to zero")

    def test_custom_parameters(self):
        """Test that custom parameters are respected."""
        initial_price = 150.0
        df = generate_minute_data(seed=42, initial_price=initial_price)

        # First open should be the initial price
        self.assertEqual(
            df["open"][0], initial_price, f"First open should be {initial_price}"
        )

        # Test with custom trading date
        custom_date = datetime(2023, 1, 15)
        df_custom = generate_minute_data(seed=42, trading_date=custom_date.date())

        # Check that the date is correct
        first_timestamp = df_custom["timestamp"][0]
        self.assertEqual(
            first_timestamp.date(),
            custom_date.date(),
            "Custom trading date should be respected",
        )

    def test_validation_function(self):
        """Test the OHLC validation function."""
        df = generate_minute_data(seed=42)

        # Valid data should pass validation
        self.assertTrue(
            validate_ohlc_data(df), "Valid OHLC data should pass validation"
        )

        # Create invalid data (high < low) and test
        df_invalid = df.clone()
        df_invalid = df_invalid.with_columns(
            [
                # Force low to be higher than high to create invalid data
                (pl.col("high") + 1.0).alias("low")
            ]
        )

        # This should fail validation
        self.assertFalse(
            validate_ohlc_data(df_invalid), "Invalid OHLC data should fail validation"
        )

    def test_volume_characteristics(self):
        """Test that volume data has expected characteristics."""
        df = generate_minute_data(seed=42)

        # Volume should be between 1000 and 10000 as specified
        min_volume = df["volume"].min()
        max_volume = df["volume"].max()

        self.assertGreaterEqual(min_volume, 1000, "Minimum volume should be >= 1000")
        self.assertLessEqual(max_volume, 10000, "Maximum volume should be <= 10000")

        # Volume should be integers
        volumes_float = df["volume"].cast(pl.Float64)
        volumes_int = df["volume"].cast(pl.Int64).cast(pl.Float64)

        self.assertTrue(
            (volumes_float == volumes_int).all(), "Volume should be integer values"
        )

    def test_data_types(self):
        """Test that all columns have the expected data types."""
        df = generate_minute_data(seed=42)

        # Check column presence
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column {col} should be present")

        # Check data types
        self.assertEqual(
            df["timestamp"].dtype, pl.Datetime, "Timestamp should be datetime"
        )
        self.assertEqual(df["open"].dtype, pl.Float64, "Open should be float64")
        self.assertEqual(df["high"].dtype, pl.Float64, "High should be float64")
        self.assertEqual(df["low"].dtype, pl.Float64, "Low should be float64")
        self.assertEqual(df["close"].dtype, pl.Float64, "Close should be float64")
        self.assertEqual(df["volume"].dtype, pl.Int64, "Volume should be int64")


if __name__ == "__main__":
    unittest.main(verbose=2)
