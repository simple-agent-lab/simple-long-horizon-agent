You are given `workspace/assets.csv` with fixed assets (asset_name, cost, salvage_value, useful_life_years).

**Your task:**
1. For each asset, calculate depreciation using 3 methods: straight-line, double-declining-balance, sum-of-years-digits
2. Generate year-by-year schedules for each asset and method
3. Write `workspace/depreciation_schedules.csv` (asset, method, year, depreciation, accumulated, book_value)
4. Write `workspace/depreciation_summary.json` with total depreciation per method and per asset.
