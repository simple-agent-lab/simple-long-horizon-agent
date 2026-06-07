# Experimental Data Hypothesis Testing

## Description

You are given a CSV file `workspace/experiment_data.csv` containing experimental measurements grouped by a categorical variable `group`. Each row has two columns:

- `group`: the group label (string)
- `measurement`: a numeric measurement (float)

Your task is to perform statistical hypothesis testing to determine if there are significant differences between the groups.

- If there are exactly two groups, perform an independent two-sample t-test.
- If there are three or more groups, perform a one-way ANOVA test.

Additionally, compute the effect size using Cohen's d for two groups or eta squared for ANOVA.

## Outputs

Write two output files in the workspace:

1. `workspace/stats_results.csv` with columns:
   - `test`: the name of the test performed (`t-test` or `ANOVA`)
   - `statistic`: the test statistic value (float)
   - `p_value`: the p-value of the test (float)
   - `effect_size`: the effect size (float)
   - `significant`: boolean indicating if the result is significant at alpha=0.05

2. `workspace/stats_summary.json` containing a JSON object with keys:
   - `test`: test name
   - `statistic`: test statistic
   - `p_value`: p-value
   - `effect_size`: effect size
   - `significant`: boolean

## Requirements

- Read the input CSV file from `workspace/experiment_data.csv`.
- Perform the appropriate statistical test based on the number of groups.
- Compute effect size:
  - For two groups, use Cohen's d.
  - For three or more groups, use eta squared.
- Use significance level alpha=0.05.
- Write the results to the specified output files.

## Notes

- You may use Python libraries such as `scipy` and `numpy`.
- Ensure numeric values in output files have reasonable precision (e.g., 4 decimal places).
- The `significant` field is `true` if p_value < 0.05, else `false`.


## Example

If the input has two groups "A" and "B", perform a t-test and compute Cohen's d.

If the input has groups "A", "B", and "C", perform ANOVA and compute eta squared.


Good luck!