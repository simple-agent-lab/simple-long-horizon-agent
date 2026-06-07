# A/B Test Statistical Analysis

## Description

You are given a CSV file `ab_test.csv` located in the workspace directory. This file contains data from an A/B test with the following columns:

- `user_id`: Unique identifier for each user
- `group`: The group the user was assigned to, either `control` or `treatment`
- `converted`: Binary indicator (0 or 1) whether the user converted

Your task is to:

1. Read the `ab_test.csv` file.
2. Compute the conversion rate for each group (`control` and `treatment`).
3. Perform a chi-squared test to determine if the difference in conversion rates is statistically significant.
4. Calculate the 95% confidence interval for the difference in conversion rates.
5. Determine if the result is significant at the 0.05 level.
6. Write the results to a JSON file `ab_results.json` in the workspace directory with the following keys:

- `control_rate`: Conversion rate of the control group (float)
- `treatment_rate`: Conversion rate of the treatment group (float)
- `lift`: Relative lift in conversion rate ((treatment_rate - control_rate) / control_rate)
- `chi2`: Chi-squared test statistic (float)
- `p_value`: P-value from the chi-squared test (float)
- `significant`: Boolean indicating if p_value < 0.05
- `recommendation`: String, "Implement treatment" if significant and lift > 0, otherwise "Do not implement treatment"

## Requirements

- Use Python or any tool of your choice to perform the analysis.
- Ensure the JSON output is valid and contains all required fields.
- The confidence interval calculation is for your internal verification and does not need to be output.

## Example

If the control group conversion rate is 0.10 and the treatment group conversion rate is 0.12, with a significant p-value, the lift is 0.2 (20%), and the recommendation should be "Implement treatment".

## Files

- Input: `workspace/ab_test.csv`
- Output: `workspace/ab_results.json`


---

Good luck!