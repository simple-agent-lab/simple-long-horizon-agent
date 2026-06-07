# Monte Carlo Pi Estimation with Convergence Analysis

## Description

Estimate the value of \( \pi \) using the Monte Carlo method by generating random points inside the unit square \([0,1] \times [0,1]\) and counting how many fall inside the unit circle of radius 1 centered at the origin.

You will perform this estimation for the following sample sizes:

- 100
- 1,000
- 10,000
- 100,000
- 1,000,000

For each sample size \( n \):

1. Generate \( n \) random points uniformly in the unit square.
2. Count the number of points that fall inside the quarter circle of radius 1 (i.e., points \( (x,y) \) where \( x^2 + y^2 \leq 1 \)).
3. Estimate \( \pi \) as \( 4 \times \frac{\text{points inside circle}}{n} \).
4. Compute the absolute error of the estimate compared to the true value of \( \pi \).

Additionally, analyze the convergence rate of the estimates by comparing the errors between successive sample sizes. The convergence rate between two sample sizes \( n_i \) and \( n_{i+1} \) is defined as:

\[
\text{convergence_rate} = \frac{\log(\text{error}_{i} / \text{error}_{i+1})}{\log(n_{i+1} / n_i)}
\]

For the smallest sample size, the convergence rate should be recorded as `null`.

## Output Files

Your program must write the following files in the workspace directory:

1. `mc_results.csv` with columns:

   - `n`: sample size
   - `estimate`: estimated value of \( \pi \)
   - `error`: absolute error of the estimate
   - `convergence_rate`: convergence rate (null for the smallest sample size)

   The file should have one row per sample size.

2. `mc_summary.json` containing a JSON object with the following keys:

   - `true_pi`: the true value of \( \pi \) (use `math.pi`)
   - `estimates`: a list of the estimates for each sample size
   - `errors`: a list of the errors for each sample size
   - `convergence_rates`: a list of convergence rates (null for the first entry)

## Requirements

- Use a fixed random seed for reproducibility.
- Use the sample sizes exactly as specified.
- Write the output files in the workspace directory.
- Ensure numeric values in CSV are written with at least 6 decimal places.

## Example (partial)

```
n,estimate,error,convergence_rate
100,3.12,0.021592,null
1000,3.14,0.001592,0.95
...
```

## Evaluation

Your solution will be tested for correctness of the Monte Carlo estimation, error calculation, convergence rate computation, and correct output file formatting.

Good luck!