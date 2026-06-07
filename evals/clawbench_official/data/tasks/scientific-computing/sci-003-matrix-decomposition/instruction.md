# SVD Matrix Decomposition and Reconstruction

## Description

You are given a data matrix stored in `workspace/data_matrix.csv`. This matrix has 50 rows and 20 columns.

Your tasks are:

1. Read the matrix from `workspace/data_matrix.csv`.
2. Perform Singular Value Decomposition (SVD) on this matrix.
3. Reconstruct the matrix using the top k singular values/vectors for k in `[1, 2, 3, 5, 10, 15, 20]`.
4. For each k, compute the reconstruction error defined as the Frobenius norm of the difference between the original matrix and the reconstructed matrix.
5. Determine the optimal k that yields the smallest reconstruction error.
6. Write a JSON file `workspace/svd_analysis.json` containing:
   - `singular_values`: the list of singular values from the decomposition,
   - `reconstruction_errors`: a dictionary mapping each k to its reconstruction error,
   - `optimal_k`: the k with the smallest reconstruction error.

## Input

- `workspace/data_matrix.csv`: CSV file with 50 rows and 20 columns of floating-point numbers.

## Output

- `workspace/svd_analysis.json`: JSON file with the following structure:

```json
{
  "singular_values": [float, float, ...],
  "reconstruction_errors": {
    "1": float,
    "2": float,
    "3": float,
    "5": float,
    "10": float,
    "15": float,
    "20": float
  },
  "optimal_k": int
}
```

## Notes

- Use the Frobenius norm (L2 norm of the matrix difference) to compute reconstruction error.
- Singular values should be listed in descending order.
- The optimal k is the one with the minimum reconstruction error.
- You may use any standard scientific computing libraries.

## Evaluation

Your output JSON will be verified for correctness of singular values, reconstruction errors, and the optimal k.

Good luck!