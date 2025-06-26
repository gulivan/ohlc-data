import os
import sys
import time
import unittest

import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_aggregator import aggregate_to_multiple_timeframes
from src.data_generator import generate_minute_data
from src.metrics_calculator import add_metrics_to_dataframes
from src.simple_visualizer import create_summary_dashboard


class TestIntegration(unittest.TestCase):
    def test_complete_pipeline(self):
        # Step 1: Generate data
        start_time = time.time()
        df_1min = generate_minute_data(seed=42)
        generation_time = time.time() - start_time

        self.assertEqual(len(df_1min), 390, "Should generate 390 1-minute bars")
        self.assertLess(
            generation_time, 1.0, "Data generation should be fast (< 1 second)"
        )

        start_time = time.time()
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        aggregation_time = time.time() - start_time

        self.assertEqual(len(df_5min), 78, "Should have 78 5-minute bars")
        self.assertEqual(len(df_30min), 13, "Should have 13 30-minute bars")
        self.assertEqual(len(df_daily), 1, "Should have 1 daily bar")
        self.assertLess(
            aggregation_time, 1.0, "Aggregation should be fast (< 1 second)"
        )

        # Step 3: Calculate metrics
        start_time = time.time()
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )
        metrics_time = time.time() - start_time

        self.assertGreater(
            len(df_5min_enhanced.columns),
            len(df_5min.columns),
            "Should have added metrics columns",
        )
        self.assertLess(
            metrics_time, 2.0, "Metrics calculation should be fast (< 2 seconds)"
        )

        self._verify_pipeline_integrity(
            df_1min, df_5min_enhanced, df_30min_enhanced, df_daily
        )

        print(
            f"Pipeline performance: Generation={generation_time:.3f}s, "
            f"Aggregation={aggregation_time:.3f}s, Metrics={metrics_time:.3f}s"
        )

    def _verify_pipeline_integrity(self, df_1min, df_5min, df_30min, df_daily):
        # Volume conservation
        original_volume = df_1min["volume"].sum()
        self.assertEqual(
            df_5min["volume"].sum(),
            original_volume,
            "Volume should be conserved in 5-minute aggregation",
        )
        self.assertEqual(
            df_30min["volume"].sum(),
            original_volume,
            "Volume should be conserved in 30-minute aggregation",
        )
        self.assertEqual(
            df_daily["volume"].sum(),
            original_volume,
            "Volume should be conserved in daily aggregation",
        )

        # Price range consistecy
        min_price_1min = df_1min["low"].min()
        max_price_1min = df_1min["high"].max()

        self.assertAlmostEqual(
            df_5min["low"].min(),
            min_price_1min,
            places=10,
            msg="Minimum price should be consistent",
        )
        self.assertAlmostEqual(
            df_5min["high"].max(),
            max_price_1min,
            places=10,
            msg="Maximum price should be consistent",
        )

        for df, name in [
            (df_5min, "5-minute"),
            (df_30min, "30-minute"),
            (df_daily, "daily"),
        ]:
            self.assertTrue((df["high"] >= df["open"]).all(), f"{name}: High >= Open")
            self.assertTrue((df["high"] >= df["close"]).all(), f"{name}: High >= Close")
            self.assertTrue((df["low"] <= df["open"]).all(), f"{name}: Low <= Open")
            self.assertTrue((df["low"] <= df["close"]).all(), f"{name}: Low <= Close")

    def test_pipeline_with_different_seeds(self):
        """Test different random seeds."""
        seeds = [42, 123, 321, 777]
        results = []

        for seed in seeds:
            df_1min = generate_minute_data(seed=seed)
            df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
            df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
                add_metrics_to_dataframes(df_1min, df_5min, df_30min)
            )

            results.append(
                {
                    "seed": seed,
                    "total_volume": df_1min["volume"].sum(),
                    "price_range": df_1min["high"].max() - df_1min["low"].min(),
                    "final_price": df_1min["close"][-1],
                    "avg_vwap": df_5min_enhanced["vwap"].mean(),
                }
            )

        # Results should be different for different seeds
        unique_volumes = {r["total_volume"] for r in results}
        unique_ranges = {r["price_range"] for r in results}

        self.assertGreater(
            len(unique_volumes), 1, "Different seeds should  return different volumes"
        )
        self.assertGreater(
            len(unique_ranges),
            1,
            "Different seeds should produce different price ranges",
        )

        for result in results:
            self.assertGreater(result["total_volume"], 0, "Volume should be positive")
            self.assertGreater(
                result["price_range"], 0, "Price range should be positive"
            )
            self.assertGreater(
                result["final_price"], 0, "Final price should be positive"
            )
            self.assertGreater(result["avg_vwap"], 0, "Average VWAP should be positive")

    def test_pipeline_performance_scaling(self):
        """Test pipeline performance with different data sizes."""
        # Test with single day (normal case)
        start_time = time.time()
        df_1min = generate_minute_data(seed=42)
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )
        single_day_time = time.time() - start_time

        self.assertLess(
            single_day_time, 5.0, "Single day pipeline should complete in < 5 seconds"
        )

        processing_rate = len(df_1min) / single_day_time
        self.assertGreater(processing_rate, 100, "Should process > 100 bars per second")

        print(f"Processing rate: {processing_rate:.0f} bars/second")

    def test_error_handling(self):
        """Test error handling in the pipeline."""
        with self.assertRaises((ValueError, TypeError)):
            generate_minute_data(volatility=-0.1)  # Negative volatility

        with self.assertRaises((ValueError, TypeError)):
            generate_minute_data(initial_price=0)  # Zero initial price

    def test_data_quality_metrics(self):
        """Test data quality throughout the pipeline."""
        df_1min = generate_minute_data(seed=42)
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )

        # Check for missing data
        for df, name in [
            (df_1min, "1-minute"),
            (df_5min_enhanced, "5-minute"),
            (df_30min_enhanced, "30-minute"),
        ]:
            for col in ["open", "high", "low", "close", "volume"]:
                null_count = df.select(pl.col(col).is_null().sum()).item()
                self.assertEqual(null_count, 0, f"{name} {col} should have no nulls")

        # Check for unrealistic values
        for df, name in [
            (df_5min_enhanced, "5-minute"),
            (df_30min_enhanced, "30-minute"),
        ]:
            if "rsi" in df.columns:
                rsi_out_of_bounds = ((df["rsi"] < 0) | (df["rsi"] > 100)).sum()
                self.assertEqual(
                    rsi_out_of_bounds, 0, f"{name} RSI should be in [0,100] range"
                )

    def test_memory_efficiency(self):
        """Test memory efficiency of the pipeline."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run pipeline
        df_1min = generate_minute_data(seed=42)
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable (< 50MB for single day)
        self.assertLess(
            memory_increase,
            50,
            f"Memory usage should be reasonable, used {memory_increase:.1f}MB",
        )

        print(f"Memory usage: {memory_increase:.1f}MB")

    def test_reproducibility_across_runs(self):
        """Test that the pipeline produces identical results across multiple runs."""
        # Run pipeline twice with same seed
        results1 = self._run_complete_pipeline(seed=42)
        results2 = self._run_complete_pipeline(seed=42)

        # Results should be identical
        self.assertTrue(
            results1["df_1min"].equals(results2["df_1min"]),
            "1-minute data should be identical across runs",
        )
        self.assertTrue(
            results1["df_5min"].equals(results2["df_5min"]),
            "5-minute data should be identical across runs",
        )
        self.assertTrue(
            results1["df_30min"].equals(results2["df_30min"]),
            "30-minute data should be identical across runs",
        )

    def _run_complete_pipeline(self, seed):
        """Helper method to run the complete pipeline."""
        df_1min = generate_minute_data(seed=seed)
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )

        return {
            "df_1min": df_1min_enhanced,
            "df_5min": df_5min_enhanced,
            "df_30min": df_30min_enhanced,
            "df_daily": df_daily,
        }

    def test_end_to_end_with_visualization(self):
        """Test the complete pipeline including visualization generation."""
        # Run complete pipeline
        df_1min = generate_minute_data(seed=42)
        df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
        df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
            add_metrics_to_dataframes(df_1min, df_5min, df_30min)
        )

        # Generate analysis reports (this also tests the visualizer)
        try:
            create_summary_dashboard(
                df_1min_enhanced,
                df_5min_enhanced,
                df_30min_enhanced,
                df_daily,
                "./test_reports",
            )
            visualization_success = True
        except Exception as e:
            visualization_success = False
            print(f"Visualization failed: {e}")

        self.assertTrue(
            visualization_success, "Visualization should complete without errors"
        )

        # Clean up test reports
        import shutil

        if os.path.exists("./test_reports"):
            shutil.rmtree("./test_reports")

    def test_pipeline_with_custom_parameters(self):
        """Test pipeline with various custom parameters."""
        custom_configs = [
            {"volatility": 0.002, "initial_price": 150.0},
            {"volatility": 0.0005, "initial_price": 50.0},
            {"volatility": 0.001, "initial_price": 200.0},
        ]

        for config in custom_configs:
            with self.subTest(config=config):
                # Run pipeline with custom configuration
                df_1min = generate_minute_data(seed=42, **config)
                df_5min, df_30min, df_daily = aggregate_to_multiple_timeframes(df_1min)
                df_1min_enhanced, df_5min_enhanced, df_30min_enhanced = (
                    add_metrics_to_dataframes(df_1min, df_5min, df_30min)
                )

                # Verify basic constraints
                self.assertEqual(len(df_1min), 390, "Should have 390 bars")
                self.assertEqual(
                    df_1min["open"][0],
                    config["initial_price"],
                    f"Initial price should be {config['initial_price']}",
                )

                # Verify pipeline completed successfully
                self.assertIn(
                    "vwap", df_5min_enhanced.columns, "Should have VWAP metric"
                )
                self.assertIn("rsi", df_5min_enhanced.columns, "Should have RSI metric")


if __name__ == "__main__":
    unittest.main(verbose=2)
