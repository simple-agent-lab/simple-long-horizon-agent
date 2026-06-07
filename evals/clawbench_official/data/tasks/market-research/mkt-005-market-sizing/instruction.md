# Calculate TAM SAM SOM Market Sizing

## Overview

You are provided with two data files in the workspace:

- `market_data.csv`: Contains market segments with the following columns:
  - `segment`: Name of the market segment
  - `population`: Total population size of the segment
  - `adoption_rate`: Percentage (0-1) of the population likely to adopt the product/service
  - `avg_spend`: Average spend per adopting customer in USD

- `company_data.json`: Contains company-specific data:
  - `market_share`: Company's market share as a decimal (e.g., 0.1 for 10%)
  - `serviceable_regions`: List of segments that the company can serve
  - `target_segments`: List of segments the company targets

## Task

Calculate the following market sizing metrics:

- **TAM (Total Addressable Market):** The total revenue opportunity if 100% of the population in all segments adopted the product at the average spend.

- **SAM (Serviceable Available Market):** The revenue opportunity within the serviceable regions, considering adoption rates.

- **SOM (Serviceable Obtainable Market):** The portion of SAM that the company can realistically capture, calculated as SAM multiplied by the company's market share.


## Steps

1. Read `market_data.csv` and `company_data.json` from the workspace.
2. Calculate TAM as the sum over all segments of `population * avg_spend` (assuming 100% adoption).
3. Calculate SAM as the sum over serviceable regions of `population * adoption_rate * avg_spend`.
4. Calculate SOM as `SAM * market_share`.
5. Write the results as JSON to `market_sizing.json` in the workspace with the structure:

```json
{
  "TAM": <float>,
  "SAM": <float>,
  "SOM": <float>
}
```

## Notes

- Use floating point numbers rounded to two decimal places in the output.
- Assume all segments in `serviceable_regions` and `target_segments` exist in `market_data.csv`.
- The `target_segments` field is provided for context but not required for calculations.

## Example

If `market_data.csv` contains:

| segment | population | adoption_rate | avg_spend |
|---------|------------|---------------|-----------|
| A       | 1000       | 0.5           | 100       |
| B       | 2000       | 0.3           | 150       |

and `company_data.json` contains:

```json
{
  "market_share": 0.1,
  "serviceable_regions": ["A"],
  "target_segments": ["A"]
}
```

Then:

- TAM = (1000 * 100) + (2000 * 150) = 100,000 + 300,000 = 400,000
- SAM = (1000 * 0.5 * 100) = 50,000
- SOM = 50,000 * 0.1 = 5,000

Output `market_sizing.json`:

```json
{
  "TAM": 400000.00,
  "SAM": 50000.00,
  "SOM": 5000.00
}
```

---

Complete the task as described above.