# Differential Gene Expression Analysis

## Task Description

You are provided with a gene expression matrix file located at `workspace/expression_matrix.csv`. This CSV file contains rows representing genes and columns representing samples. The first column is the gene identifier, and the subsequent columns are expression values for samples. The sample columns are labeled with condition information in the header, e.g., `Sample1_Control`, `Sample2_Control`, `Sample3_Treated`, `Sample4_Treated`.

Your task is to:

1. **Read** the expression matrix CSV file.
2. **Normalize** the expression values for each sample using log2 transformation after adding a pseudocount of 1.
3. **Compute** the fold-change for each gene between the two conditions (Treated vs Control). Fold-change is defined as the mean expression in Treated samples divided by the mean expression in Control samples (on the original scale, before log transformation).
4. **Perform** an independent two-sample t-test (unequal variance) for each gene comparing the expression values between the two conditions.
5. **Determine** significance for each gene using a p-value threshold of 0.05.
6. **Write** the results to `workspace/deg_results.csv` with columns:
   - `gene`
   - `fold_change` (float, rounded to 4 decimals)
   - `p_value` (float, rounded to 6 decimals)
   - `significant` (boolean, `True` if p-value < 0.05 else `False`)
7. **Write** a summary JSON file to `workspace/expression_summary.json` containing:
   - total number of genes
   - number of significant genes
   - mean fold-change of significant genes

## Input File Format

`expression_matrix.csv` (CSV):
- First column: `gene` (string)
- Subsequent columns: sample expression values (float)
- Column headers include condition labels, e.g. `Sample1_Control`, `Sample2_Treated`.

Example header:
```
gene,Sample1_Control,Sample2_Control,Sample3_Treated,Sample4_Treated
```

## Output Files

- `deg_results.csv` (CSV): gene-wise differential expression results.
- `expression_summary.json` (JSON): summary statistics.

## Notes
- Use log2(x + 1) normalization for expression values.
- Fold-change is computed on original scale (not log-transformed).
- Use two-sided t-test with unequal variance.
- Round fold-change to 4 decimals and p-values to 6 decimals in output.
- Use p-value < 0.05 to determine significance.

## Example

If the gene `GeneA` has mean expression 10 in Treated and 5 in Control, fold-change is 2.0.
If the t-test p-value is 0.03, then `significant` is `True`.

---

Complete the task by implementing the above steps and writing the output files accordingly.