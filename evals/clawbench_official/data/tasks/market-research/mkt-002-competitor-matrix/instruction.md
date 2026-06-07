# Build Competitive Analysis Scoring Matrix

## Overview

You are provided with a CSV file named `products.csv` located in the workspace directory. This file contains market data for various companies and their products with the following columns:

- `company`: Name of the company
- `product`: Name of the product
- `price`: Price of the product (float)
- `features_count`: Number of features the product offers (integer)
- `rating`: Customer rating (float, scale 1-5)
- `market_share_pct`: Market share percentage (float, 0-100)

Your task is to process this data to build a competitive analysis scoring matrix. You need to:

1. **Score each company** on the following criteria:
   - **price_competitiveness**: Lower prices get higher scores. Normalize scores between 1 (least competitive) and 10 (most competitive).
   - **feature_richness**: Companies with more features get higher scores, normalized 1-10.
   - **customer_satisfaction**: Based on average rating, normalized 1-10.
   - **market_position**: Based on market share percentage, normalized 1-10.

2. **Aggregate scores per company** by averaging the scores of all their products.

3. **Output two files**:
   - `competitor_matrix.csv`: A CSV file with columns:
     - `company`
     - `price_competitiveness`
     - `feature_richness`
     - `customer_satisfaction`
     - `market_position`
   - `competitive_summary.json`: A JSON file summarizing the rankings of companies for each criterion. For each criterion, list companies ordered from highest to lowest score.

## Requirements

- Read the input file `products.csv` from the workspace.
- Perform normalization of scores for each criterion on the scale 1 to 10.
- Average scores across all products of the same company.
- Write the output files `competitor_matrix.csv` and `competitive_summary.json` to the workspace.

## Notes

- Use min-max normalization for each criterion across all products before averaging by company.
- If all values for a criterion are the same, assign a score of 10 for all.
- The JSON summary should be a dictionary with keys `price_competitiveness`, `feature_richness`, `customer_satisfaction`, and `market_position`. Each key maps to a list of company names ordered by descending score.

Example snippet of `competitor_matrix.csv`:

```
company,price_competitiveness,feature_richness,customer_satisfaction,market_position
Acme Corp,7.5,8.2,9.1,6.4
Beta Inc,9.8,6.5,7.3,5.0
```

Example snippet of `competitive_summary.json`:

```json
{
  "price_competitiveness": ["Beta Inc", "Acme Corp", "Gamma LLC"],
  "feature_richness": ["Acme Corp", "Gamma LLC", "Beta Inc"],
  "customer_satisfaction": ["Acme Corp", "Beta Inc", "Gamma LLC"],
  "market_position": ["Gamma LLC", "Acme Corp", "Beta Inc"]
}
```

## Deliverables

- `competitor_matrix.csv`
- `competitive_summary.json`

Ensure your solution is robust and handles the provided data correctly.