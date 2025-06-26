Hometask for OHLC data.

Implementation relies on *polars* as its core which is notably faster than *pandas* due to various optimizations and vector-based computation.



## Structure
```
  src/
  ├── utils/benchmark.py            # Compare initial vs optimized approaches
  ├── data_aggregator.py            # Data aggregation
  ├── data_generator.py             # 1-minute OHLC data generation
  ├── metrics_calculator.py         # Financial metrics and technical indicators
  ├── performance_optimizations.py  # Optimized implementation for extra speed and memory effeciency
  ├── simple_visualizer.py          # Text visualization/reports
  └── visualizer.py                 # Visuzalization (saves to charts.png)
```

### Installation and usage

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Set env and run application
```bash
uv venv -p 3.11
source .venv/bin/activate
uv sync

# Run the main pipeline (first run may take longer)
uv run python main.py --save-data --seed 123 --volatility 0.002 --initial-price 150.0 --visualize --optimize

# tests
uv run pytest
```

### Using Docker

```bash
docker build -t ohlc-processor .
docker run ohlc-processor
```


## Optimization
When `--optimize` flag is enabled then optimized algorythms are utilized:
- vectorized operations for random arrays
- np.cumprod for price calculations
- polars lazy frames support
- array pre-allocation
