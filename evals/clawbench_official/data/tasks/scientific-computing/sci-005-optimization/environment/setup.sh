#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE"

python3 - <<'EOF'
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Problem dimension
n = 5
m = 7  # number of constraints

# Generate a random positive semidefinite matrix Q
A_rand = np.random.randn(n, n)
Q = np.dot(A_rand.T, A_rand)  # Q is symmetric positive semidefinite

# Generate linear term c
c = np.random.uniform(-5, 5, size=n).tolist()

# Generate constraints: A x <= b
# For each constraint, generate a random vector A_i and scalar b_i
constraints = []
for _ in range(m):
    A_i = np.random.uniform(-3, 3, size=n)
    # To ensure feasibility, pick a random point x_feas and set b_i accordingly
    x_feas = np.random.uniform(-1, 1, size=n)
    b_i = float(np.dot(A_i, x_feas) + abs(np.random.uniform(0.5, 2.0)))  # b_i > A_i x_feas
    constraints.append({
        "A": A_i.tolist(),
        "b": b_i
    })

problem = {
    "Q": Q.tolist(),
    "c": c,
    "constraints": constraints
}

with open(f"{"$WORKSPACE"}/optimization_problem.json", "w") as f:
    json.dump(problem, f, indent=2)
EOF
