"""
Comprehensive unit tests for the OHLC data aggregator module.

These tests verify the correctness of data aggregation, ensuring that:
1. Aggregation produces correct number of bars
2. OHLC values are properly aggregated
3. Volume is correctly summed
4. Edge cases are handled properly
"""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_aggregator import (
    aggregate_ohlc,
    aggregate_to_multiple_timeframes,
    validate_aggregated_data,
)
from src.data_generator import generate_minute_data


class TestDataAggregator(unittest.TestCase):
    """Test suite for the OHLC data aggregator."""

    def setUp(self):
        """Set up test data."""
        self.df_1min = generate_minute_data(seed=42)

    def test_aggregation_lengths(self):
        """Test that aggregation produces the correct number of bars."""
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(self.df_1min)

        # 390 minutes / 5 = 78 bars
        self.assertEqual(len(df_5min), 78, "Should produce 78 5-minute bars (390/5)")

        # 390 minutes / 30 = 13 bars
        self.assertEqual(len(df_30min), 13, "Should produce 13 30-minute bars (390/30)")

        # 1 trading day = 1 daily bar
        self.assertEqual(len(df_daily), 1, "Should produce 1 daily bar")

    def test_5min_ohlc_integrity(self):
        """Test that 5-minute OHLC values are correctly aggregated."""
        df_5min = aggregate_ohlc(self.df_1min, "5m")

        # Test first 5-minute bar (first 5 1-minute bars)
        first_5min = df_5min.row(0, named=True)
        first_5_1min = self.df_1min.slice(0, 5)

        self.assertEqual(
            first_5min["open"],
            first_5_1min["open"][0],
            "5-min Open should match first 1-min Open",
        )
        self.assertEqual(
            first_5min["close"],
            first_5_1min["close"][-1],
            "5-min Close should match last 1-min Close",
        )
        self.assertEqual(
            first_5min["high"],
            first_5_1min["high"].max(),
            "5-min High should be max of 1-min Highs",
        )
        self.assertEqual(
            first_5min["low"],
            first_5_1min["low"].min(),
            "5-min Low should be min of 1-min Lows",
        )
        self.assertEqual(
            first_5min["volume"],
            first_5_1min["volume"].sum(),
            "5-min Volume should be sum of 1-min Volumes",
        )

    def test_30min_ohlc_integrity(self):
        """Test that 30-minute OHLC values are correctly aggregated."""
        df_30min = aggregate_ohlc(self.df_1min, "30m")

        # Test first 30-minute bar (first 30 1-minute bars)
        first_30min = df_30min.row(0, named=True)
        first_30_1min = self.df_1min.slice(0, 30)

        self.assertEqual(
            first_30min["open"],
            first_30_1min["open"][0],
            "30-min Open should match first 1-min Open",
        )
        self.assertEqual(
            first_30min["close"],
            first_30_1min["close"][-1],
            "30-min Close should match last 1-min Close",
        )
        self.assertEqual(
            first_30min["high"],
            first_30_1min["high"].max(),
            "30-min High should be max of 1-min Highs",
        )
        self.assertEqual(
            first_30min["low"],
            first_30_1min["low"].min(),
            "30-min Low should be min of 1-min Lows",
        )
        self.assertEqual(
            first_30min["volume"],
            first_30_1min["volume"].sum(),
            "30-min Volume should be sum of 1-min Volumes",
        )

    def test_daily_aggregation(self):
        """Test that daily aggregation works correctly."""
        df_daily = aggregate_ohlc(self.df_1min, "1d")

        # Should have exactly one bar
        self.assertEqual(len(df_daily), 1, "Should have exactly one daily bar")

        daily_bar = df_daily.row(0, named=True)

        # Daily values should match entire dataset
        self.assertEqual(
            daily_bar["open"],
            self.df_1min["open"][0],
            "Daily Open should match first 1-min Open",
        )
        self.assertEqual(
            daily_bar["close"],
            self.df_1min["close"][-1],
            "Daily Close should match last 1-min Close",
        )
        self.assertEqual(
            daily_bar["high"],
            self.df_1min["high"].max(),
            "Daily High should be max of all 1-min Highs",
        )
        self.assertEqual(
            daily_bar["low"],
            self.df_1min["low"].min(),
            "Daily Low should be min of all 1-min Lows",
        )
        self.assertEqual(
            daily_bar["volume"],
            self.df_1min["volume"].sum(),
            "Daily Volume should be sum of all 1-min Volumes",
        )

    def test_aggregation_edge_cases(self):
        """Test edge cases such as the first and last bars of the day."""
        df_30min = aggregate_ohlc(self.df_1min, "30m")

        # Test first 30-minute bar (9:30-10:00)
        first_30min = df_30min.row(0, named=True)
        self.df_1min.slice(0, 30)

        # Verify timestamp alignment
        expected_first_timestamp = self.df_1min["timestamp"][0]
        self.assertEqual(
            first_30min["timestamp"].replace(second=0, microsecond=0),
            expected_first_timestamp.replace(second=0, microsecond=0),
            "First 30-min timestamp should align with first 1-min timestamp",
        )

        # Test last 30-minute bar (15:30-16:00)
        last_30min = df_30min.row(-1, named=True)
        last_30_1min = self.df_1min.slice(-30, 30)

        self.assertEqual(
            last_30min["open"],
            last_30_1min["open"][0],
            "Last 30-min Open should match first 1-min Open in that period",
        )
        self.assertEqual(
            last_30min["close"],
            last_30_1min["close"][-1],
            "Last 30-min Close should match last 1-min Close",
        )

    def test_aggregated_data_validation(self):
        """Test the aggregated data validation function."""
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(self.df_1min)

        # All validations should pass
        self.assertTrue(
            validate_aggregated_data(self.df_1min, df_5min, 78, "5-minute"),
            "5-minute aggregation validation should pass",
        )
        self.assertTrue(
            validate_aggregated_data(self.df_1min, df_30min, 13, "30-minute"),
            "30-minute aggregation validation should pass",
        )
        self.assertTrue(
            validate_aggregated_data(self.df_1min, df_daily, 1, "Daily"),
            "Daily aggregation validation should pass",
        )

    def test_volume_conservation(self):
        """Test that volume is conserved during aggregation."""
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(self.df_1min)

        original_volume = self.df_1min["volume"].sum()

        # Volume should be conserved across all timeframes
        self.assertEqual(
            df_5min["volume"].sum(),
            original_volume,
            "5-minute aggregation should conserve volume",
        )
        self.assertEqual(
            df_30min["volume"].sum(),
            original_volume,
            "30-minute aggregation should conserve volume",
        )
        self.assertEqual(
            df_daily["volume"].sum(),
            original_volume,
            "Daily aggregation should conserve volume",
        )

    def test_price_relationships_after_aggregation(self):
        """Test that OHLC relationships are maintained after aggregation."""
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(self.df_1min)

        for df, name in [
            (df_5min, "5-minute"),
            (df_30min, "30-minute"),
            (df_daily, "daily"),
        ]:
            # High should be >= Open and Close
            self.assertTrue(
                (df["high"] >= df["open"]).all(), f"{name}: High should be >= Open"
            )
            self.assertTrue(
                (df["high"] >= df["close"]).all(), f"{name}: High should be >= Close"
            )

            # Low should be <= Open and Close
            self.assertTrue(
                (df["low"] <= df["open"]).all(), f"{name}: Low should be <= Open"
            )
            self.assertTrue(
                (df["low"] <= df["close"]).all(), f"{name}: Low should be <= Close"
            )

            # All prices should be positive
            self.assertTrue((df["open"] > 0).all(), f"{name}: Open should be positive")
            self.assertTrue((df["high"] > 0).all(), f"{name}: High should be positive")
            self.assertTrue((df["low"] > 0).all(), f"{name}: Low should be positive")
            self.assertTrue(
                (df["close"] > 0).all(), f"{name}: Close should be positive"
            )
            self.assertTrue(
                (df["volume"] > 0).all(), f"{name}: Volume should be positive"
            )

    def test_timestamp_alignment(self):
        """Test that timestamps are correctly aligned for different timeframes."""
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(self.df_1min)

        # 5-minute bars should start every 5 minutes
        for i in range(1, len(df_5min)):
            time_diff = (
                df_5min["timestamp"][i] - df_5min["timestamp"][i - 1]
            ).total_seconds()
            self.assertEqual(
                time_diff,
                300,
                f"5-minute bars should be 300 seconds apart, got {time_diff}",
            )

        # 30-minute bars should start every 30 minutes
        for i in range(1, len(df_30min)):
            time_diff = (
                df_30min["timestamp"][i] - df_30min["timestamp"][i - 1]
            ).total_seconds()
            self.assertEqual(
                time_diff,
                1800,
                f"30-minute bars should be 1800 seconds apart, got {time_diff}",
            )

    def test_empty_periods_handling(self):
        """Test handling of empty periods (though shouldn't occur with our data)."""
        # Create test data with a gap
        df_with_gap = self.df_1min.slice(0, 100)  # Only first 100 minutes

        df_5min_gap = aggregate_ohlc(df_with_gap, "5m")

        # Should have 20 bars (100 / 5)
        self.assertEqual(len(df_5min_gap), 20, "Should handle partial data correctly")

        # All aggregated values should still be valid
        self.assertTrue(
            (df_5min_gap["open"].is_not_null()).all(), "All opens should be non-null"
        )
        self.assertTrue(
            (df_5min_gap["volume"] > 0).all(), "All volumes should be positive"
        )

    def test_aggregation_consistency(self):
        """Test that aggregation is consistent across multiple runs."""
        df_5min_1, _, _ = aggregate_to_multiple_timeframes(self.df_1min)
        df_5min_2, _, _ = aggregate_to_multiple_timeframes(self.df_1min)

        # Results should be identical
        self.assertTrue(
            df_5min_1.equals(df_5min_2),
            "Aggregation should be deterministic and consistent",
        )


if __name__ == "__main__":
    unittest.main(verbose=2)
