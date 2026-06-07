# Constrained Optimization with Gradient Descent

## Objective

You are given a constrained optimization problem defined by a quadratic objective function and linear inequality constraints. Your task is to implement a projected gradient descent algorithm to find the minimum of the objective function subject to the constraints.

## Input

Read the file `workspace/optimization_problem.json`. It contains a JSON object with the following fields:

- `Q`: a 2D list of floats representing a positive semidefinite matrix (the quadratic term in the objective).
- `c`: a list of floats representing the linear term in the objective.
- `constraints`: a list of constraint objects, each with:
  - `A`: a list of floats representing coefficients of a linear inequality constraint.
  - `b`: a float representing the right-hand side of the inequality.

The optimization problem is:

Minimize:

$$ f(x) = \frac{1}{2} x^T Q x + c^T x $$

Subject to:

$$ A_i x \leq b_i, \quad \text{for all constraints } i $$

where $x$ is a vector of variables.

## Output

Write the results to `workspace/optimization_result.json` as a JSON object with the following fields:

- `solution`: list of floats representing the optimal vector $x$ found.
- `objective_value`: float, the value of the objective function at the solution.
- `iterations`: integer, number of iterations performed.
- `convergence_history`: list of floats, the objective value at each iteration.

## Requirements

- Implement projected gradient descent:
  - Use a fixed step size.
  - After each gradient step, project the iterate onto the feasible region defined by the linear inequalities.
- Stop when either:
  - The change in objective value between iterations is less than 1e-6, or
  - The number of iterations reaches 1000.
- Use the initial point $x=0$.

## Notes

- The projection onto the feasible region defined by linear inequalities can be done by solving a quadratic program or using any suitable method.
- You may use standard Python libraries such as `numpy` and `scipy`.

## Example

If the problem is minimizing $\frac{1}{2}x^T x$ subject to $x \geq 0$, the solution is $x=0$.

## Evaluation

Your solution will be tested on multiple problem instances with realistic quadratic objectives and constraints.

---

Good luck!