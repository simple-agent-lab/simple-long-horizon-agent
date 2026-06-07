# Price Elasticity and Revenue Optimization

## Description

You are provided with a CSV file `workspace/pricing_history.csv` containing historical pricing data for a product over multiple months. The CSV has the following columns:

- `month`: The month identifier (e.g., `2023-01`)
- `price`: The price of the product in that month (float)
- `units_sold`: Number of units sold in that month (integer)
- `revenue`: Total revenue in that month (float)

Your task is to:

1. Calculate the **price elasticity of demand** between each pair of consecutive months. The price elasticity between month i and i+1 is computed as:

   \[
   E_i = \frac{\% \text{ change in units sold}}{\% \text{ change in price}} = \frac{(Q_{i+1} - Q_i) / Q_i}{(P_{i+1} - P_i) / P_i}
   \]

   where \(Q_i\) and \(P_i\) are units sold and price in month i.

2. Compute the average price elasticity over all consecutive month pairs.

3. Estimate the revenue-maximizing price using the average elasticity and the last observed month data. Assume a linear demand curve and that revenue \(R = P \times Q\). Use the formula:

   \[
   P^* = \frac{E}{E + 1} \times P_{last}
   \]

   where \(E\) is the average elasticity (use its absolute value) and \(P_{last}\) is the last observed price.

4. Estimate the maximum revenue at this optimal price by assuming demand changes according to elasticity:

   \[
   Q^* = Q_{last} \times \left(\frac{P^*}{P_{last}}\right)^E
   \]

   \[
   R^* = P^* \times Q^*
   \]

5. Write the results to `workspace/pricing_analysis.json` with the following structure:

```json
{
  "elasticities": [E_1, E_2, ..., E_{n-1}],
  "avg_elasticity": avg_E,
  "optimal_price": P^*,
  "max_revenue_estimate": R^*
}
```

## Requirements

- Read `pricing_history.csv` from the workspace.
- Perform the calculations as described.
- Write the JSON output to `pricing_analysis.json` in the workspace.
- Use floating point numbers with reasonable precision.

## Notes

- The price elasticity values can be positive or negative depending on the data; use the absolute value of average elasticity for the optimal price formula.
- The dataset contains at least 20 months of data.
- Ensure your solution handles division by zero or zero changes gracefully.

## Example

If the CSV contains:

| month  | price | units_sold | revenue |
|--------|-------|------------|---------|
| 2023-01| 10.0  | 100        | 1000.0  |
| 2023-02| 9.5   | 110        | 1045.0  |

Then the elasticity between these two months is:

\[
E = \frac{(110 - 100)/100}{(9.5 - 10)/10} = \frac{0.1}{-0.05} = -2.0
\]

The output JSON would include this elasticity and subsequent calculations.

---

Complete the task by implementing the described steps.