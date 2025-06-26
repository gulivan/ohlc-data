import os
import sys
import time
import unittest
from io import StringIO

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import test modules
from tests.test_data_aggregator import TestDataAggregator
from tests.test_data_generator import TestDataGenerator
from tests.test_integration import TestIntegration
from tests.test_metrics_calculator import TestMetricsCalculator


class TestResult:
    """Custom test result class to capture test statistics."""

    def __init__(self):
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def success_rate(self):
        if self.tests_run == 0:
            return 0
        return (self.success_count / self.tests_run) * 100


def run_test_suite(test_class, test_name):
    """Run a specific test suite and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name} Tests")
    print(f"{'=' * 60}")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

    # Capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)

    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print results
    output = stream.getvalue()
    print(output)

    # Calculate statistics
    test_result = TestResult()
    test_result.tests_run = result.testsRun
    test_result.failures = len(result.failures)
    test_result.errors = len(result.errors)
    test_result.skipped = len(getattr(result, "skipped", []))
    test_result.success_count = (
        test_result.tests_run - test_result.failures - test_result.errors
    )
    test_result.start_time = start_time
    test_result.end_time = end_time

    # Print summary
    print(f"\n{test_name} Summary:")
    print(f"  Tests run: {test_result.tests_run}")
    print(f"  Successes: {test_result.success_count}")
    print(f"  Failures: {test_result.failures}")
    print(f"  Errors: {test_result.errors}")
    print(f"  Skipped: {test_result.skipped}")
    print(f"  Success rate: {test_result.success_rate:.1f}%")
    print(f"  Duration: {test_result.duration:.3f} seconds")

    return test_result, result.wasSuccessful()


def run_all_tests():
    """Run all test suites and provide comprehensive reporting."""
    print("OHLC Data Processing - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Test suites to run
    test_suites = [
        (TestDataGenerator, "Data Generator"),
        (TestDataAggregator, "Data Aggregator"),
        (TestMetricsCalculator, "Metrics Calculator"),
        (TestIntegration, "Integration"),
    ]

    # Run all test suites
    all_results = []
    all_successful = True
    total_start_time = time.time()

    for test_class, test_name in test_suites:
        try:
            test_result, success = run_test_suite(test_class, test_name)
            all_results.append((test_name, test_result))
            if not success:
                all_successful = False
        except Exception as e:
            print(f"\nERROR running {test_name} tests: {e}")
            all_successful = False

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Print overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL TEST SUMMARY")
    print(f"{'=' * 70}")

    total_tests = sum(result.tests_run for _, result in all_results)
    total_successes = sum(result.success_count for _, result in all_results)
    total_failures = sum(result.failures for _, result in all_results)
    total_errors = sum(result.errors for _, result in all_results)

    print(f"Total tests run: {total_tests}")
    print(f"Total successes: {total_successes}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(
        f"Overall success rate: {(total_successes / total_tests * 100) if total_tests > 0 else 0:.1f}%"
    )
    print(f"Total duration: {total_duration:.3f} seconds")

    # Detailed breakdown
    print("\nDetailed Breakdown:")
    print(
        f"{'Test Suite':<20} {'Tests':<6} {'Success':<8} {'Failures':<9} {'Errors':<7} {'Time':<8}"
    )
    print(f"{'-' * 20} {'-' * 6} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 8}")

    for test_name, result in all_results:
        print(
            f"{test_name:<20} {result.tests_run:<6} {result.success_count:<8} "
            f"{result.failures:<9} {result.errors:<7} {result.duration:<8.3f}"
        )

    # Performance metrics
    if total_tests > 0:
        avg_test_time = total_duration / total_tests
        print("\nPerformance Metrics:")
        print(f"  Average test time: {avg_test_time:.3f} seconds")
        print(f"  Tests per second: {total_tests / total_duration:.1f}")

    # Final status
    print(f"\n{'=' * 70}")
    if all_successful:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The OHLC data processing pipeline is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the test output above for details.")
    print(f"{'=' * 70}")

    return all_successful


def run_quick_tests():
    """Run a subset of critical tests for quick validation."""
    print("Running Quick Test Suite (Critical Tests Only)")
    print("=" * 50)

    # Define critical tests
    critical_tests = [
        ("test_data_length", TestDataGenerator),
        ("test_price_integrity", TestDataGenerator),
        ("test_aggregation_lengths", TestDataAggregator),
        ("test_5min_ohlc_integrity", TestDataAggregator),
        ("test_vwap_calculation", TestMetricsCalculator),
        ("test_complete_pipeline", TestIntegration),
    ]

    suite = unittest.TestSuite()
    for test_method, test_class in critical_tests:
        suite.addTest(test_class(test_method))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\nQuick Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OHLC data processing tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only critical tests for quick validation",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.quick:
        success = run_quick_tests()
    else:
        success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
