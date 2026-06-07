# Sales Trend Analysis and Quarterly Forecast

## Description

You are provided with a CSV file `monthly_sales.csv` located in the workspace directory. This file contains 36 months of sales data with two columns:

- `month`: in the format `YYYY-MM` (e.g., 2021-01)
- `sales`: integer sales figures for that month

Your task is to:

1. Read the `monthly_sales.csv` file.
2. Compute the 3-month and 6-month moving averages of sales.
3. Detect seasonality by calculating seasonal indices for each month of the year.
4. Forecast the sales for the next 3 months beyond the last month in the data.
5. Determine the overall trend (e.g., "increasing", "decreasing", or "stable") based on the moving averages.
6. Write the results to `trend_analysis.json` in the workspace directory.


## Output Format

Write a JSON file named `trend_analysis.json` with the following structure:

```json
{
  "moving_averages": {
    "3_month": [float, ...],  // list of 3-month moving averages aligned with months (starting from month 3)
    "6_month": [float, ...]   // list of 6-month moving averages aligned with months (starting from month 6)
  },
  "seasonal_indices": {
    "01": float,  // January index
    "02": float,  // February index
    ...
    "12": float   // December index
  },
  "forecast": [
    {"month": "YYYY-MM", "forecasted_sales": float},
    ... (3 entries for next 3 months)
  ],
  "overall_trend": "increasing" | "decreasing" | "stable"
}
```


## Notes

- Use the 3-month and 6-month moving averages to help identify the trend.
- Seasonal indices should be computed as the average ratio of actual sales to moving average for each calendar month.
- Forecasting can be done using a simple method such as extending the trend and adjusting for seasonality.
- The overall trend is "increasing" if the 6-month moving average shows a clear upward slope, "decreasing" if downward, and "stable" otherwise.


## Files

- Input: `workspace/monthly_sales.csv`
- Output: `workspace/trend_analysis.json`
