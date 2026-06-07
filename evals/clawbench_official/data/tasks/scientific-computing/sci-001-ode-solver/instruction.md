# Runge-Kutta ODE Solver for Predator-Prey Model

## Description

You are tasked with implementing a numerical solver for the Lotka-Volterra predator-prey model using the classical 4th-order Runge-Kutta method.

The system of differential equations is:

\[
\frac{dx}{dt} = a x - b x y
\]
\[
\frac{dy}{dt} = -c y + d x y
\]

where:

- \(x(t)\) is the prey population
- \(y(t)\) is the predator population
- Parameters:
  - \(a = 1.1\)
  - \(b = 0.4\)
  - \(c = 0.4\)
  - \(d = 0.1\)

Initial conditions:

- \(x(0) = 10\)
- \(y(0) = 10\)

Time interval:

- \(t \in [0, 50]\)
- Time step \(\Delta t = 0.01\)

## Requirements

1. Implement the classical 4th-order Runge-Kutta method to solve the system numerically.
2. Simulate the system from \(t=0\) to \(t=50\) with the given parameters and initial conditions.
3. Write the results to a CSV file named `simulation.csv` in the workspace directory.
   - The CSV must have three columns with headers: `t`, `prey`, `predator`.
   - Each row corresponds to a time step.
4. Compute the equilibrium point \((x^*, y^*)\) of the system analytically:

\[
x^* = \frac{c}{d}, \quad y^* = \frac{a}{b}
\]

5. Write the equilibrium point to a JSON file named `equilibrium.json` in the workspace directory with keys `prey` and `predator`.

## Output files

- `simulation.csv`:

```
t,prey,predator
0.00,10.0,10.0
0.01, ... , ...
...
50.00, ... , ...
```

- `equilibrium.json`:

```json
{
  "prey": 4.0,
  "predator": 2.75
}
```

## Notes

- Use double precision floating-point arithmetic.
- Ensure numerical stability and accuracy.
- The solution should be reproducible.
- The CSV should have at least 5001 rows (including header).

Good luck!