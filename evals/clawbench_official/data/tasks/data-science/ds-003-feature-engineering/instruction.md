# Feature Engineering and Selection Pipeline

## Task Description

You are given a dataset `raw_data.csv` located in the workspace directory. This dataset contains a mix of numeric and categorical features along with a numeric target variable.

Your task is to perform a feature engineering and selection pipeline with the following steps:

1. **Read the input CSV file:** `workspace/raw_data.csv`.

2. **Feature Engineering:**
   - **Binning:** For at least one numeric feature, create a binned categorical feature by dividing the values into equal-width bins.
   - **One-Hot Encoding:** Convert all categorical features (including the binned feature) into one-hot encoded features.
   - **Interaction Terms:** Create new features by multiplying pairs of numeric features (including the original numeric features and the log-transformed features).
   - **Log Transform:** Apply a natural logarithm transform (using `log1p` to handle zeros) to all numeric features.

3. **Correlation Computation:**
   - Compute the Pearson correlation coefficient between each engineered feature and the target variable.

4. **Feature Selection:**
   - Select the top 10 features with the highest absolute correlation to the target.

5. **Output:**
   - Write the selected features and the target variable to `workspace/engineered_features.csv`.
   - Write a JSON report `workspace/feature_report.json` containing:
     - The list of all engineered features with their correlation values.
     - The names of the selected top 10 features.


## Input File Format

- `raw_data.csv` is a CSV file with a header row.
- It contains multiple columns:
  - Mixed numeric and categorical features.
  - One column named `target` which is numeric.


## Output Files

- `engineered_features.csv`: CSV file containing the top 10 selected features and the target column.
- `feature_report.json`: JSON file with correlation values for all engineered features and the list of selected features.


## Requirements

- Use Python or Bash tools to implement the pipeline.
- Ensure the output files are created in the workspace directory.
- Use `log1p` for log transforms to handle zero or negative values safely.
- The correlation should be Pearson correlation.


## Evaluation

Your solution will be evaluated by:
- Correctness of feature engineering steps.
- Correctness of correlation computation.
- Correctness of feature selection.
- Correct output files with correct formats.


Good luck!