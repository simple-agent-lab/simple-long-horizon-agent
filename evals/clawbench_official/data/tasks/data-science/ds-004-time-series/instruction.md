# Time Series Decomposition and Forecasting

## Objective

You are given a time series dataset `workspace/timeseries.csv` containing 120 data points with two columns: `date` (in YYYY-MM-DD format) and `value` (a float). Your task is to:

1. Read the time series data from `workspace/timeseries.csv`.
2. Perform time series decomposition into three components:
   - Trend
   - Seasonal
   - Residual

   Use an additive decomposition method suitable for daily or monthly data.

3. Compute the autocorrelation of the original time series values for lags 1 through 12.

4. Forecast the next 12 periods using a weighted moving average method. The weights should decrease linearly from the most recent observation to the oldest in the window.

5. Write the decomposition results to `workspace/decomposition.csv` with columns:
   - `date` (same dates as input)
   - `trend` (trend component values)
   - `seasonal` (seasonal component values)
   - `residual` (residual component values)

6. Write the forecast results to `workspace/forecast.json` as a JSON object with two keys:
   - `dates`: list of the next 12 dates (continuing from the last date in the input)
   - `forecast`: list of the corresponding forecasted values


## Details

- The input CSV `timeseries.csv` contains 120 rows, each with a date and a value.
- Use an appropriate frequency for decomposition (e.g., monthly or weekly) based on the data.
- Autocorrelation should be computed on the original `value` series, for lags 1 to 12.
- The weighted moving average forecast should use the most recent 12 observations, with weights linearly decreasing from 12 to 1.
- Output files must be saved exactly as specified in the workspace directory.


## Submission

Your solution should be a script or program that reads `workspace/timeseries.csv`, performs the computations, and writes the two output files:

- `workspace/decomposition.csv`
- `workspace/forecast.json`

Ensure your output files are correctly formatted and contain all required data.


## Example snippet for forecast weights

If the last 12 observations are `[v1, v2, ..., v12]` (v12 is most recent), weights are `[1, 2, ..., 12]` applied correspondingly, normalized to sum to 1.


Good luck!