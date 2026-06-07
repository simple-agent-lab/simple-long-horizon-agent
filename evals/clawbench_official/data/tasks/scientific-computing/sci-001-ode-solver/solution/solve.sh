#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE"

# Parameters
A=1.1
B=0.4
C=0.4
D=0.1
X0=10.0
Y0=10.0
T0=0.0
TEND=50.0
DT=0.01

# Number of steps
NSTEPS=$(python3 -c "import math; print(int(round(($TEND - $T0)/$DT)))")

# Function to compute derivatives
# Arguments: t x y
# Outputs dx dt, dy dt
function f() {
  local x=$1
  local y=$2
  # dx/dt = a*x - b*x*y
  local dx=$(python3 -c "print($A * $x - $B * $x * $y)")
  # dy/dt = -c*y + d*x*y
  local dy=$(python3 -c "print(-$C * $y + $D * $x * $y)")
  echo "$dx $dy"
}

# Runge-Kutta 4th order step
# Arguments: x y dt
# Outputs: x_next y_next
function rk4_step() {
  local x=$1
  local y=$2
  local dt=$3

  read k1x k1y <<< $(f $x $y)
  read k2x k2y <<< $(f $(python3 -c "print($x + $dt/2 * $k1x)") $(python3 -c "print($y + $dt/2 * $k1y)"))
  read k3x k3y <<< $(f $(python3 -c "print($x + $dt/2 * $k2x)") $(python3 -c "print($y + $dt/2 * $k2y)"))
  read k4x k4y <<< $(f $(python3 -c "print($x + $dt * $k3x)") $(python3 -c "print($y + $dt * $k3y)"))

  local x_next=$(python3 -c "print($x + $dt/6 * ($k1x + 2*$k2x + 2*$k3x + $k4x))")
  local y_next=$(python3 -c "print($y + $dt/6 * ($k1y + 2*$k2y + 2*$k3y + $k4y))")

  echo "$x_next $y_next"
}

# Write header
echo "t,prey,predator" > "$WORKSPACE/simulation.csv"

# Initial conditions
t=$T0
x=$X0
y=$Y0

# Write initial row
printf "%.2f,%.10f,%.10f\n" "$t" "$x" "$y" >> "$WORKSPACE/simulation.csv"

for ((i=1; i<=NSTEPS; i++)); do
  read x y <<< $(rk4_step $x $y $DT)
  t=$(python3 -c "print($T0 + $i * $DT)")
  printf "%.2f,%.10f,%.10f\n" "$t" "$x" "$y" >> "$WORKSPACE/simulation.csv"
  # Safety check: populations should not be negative
  if (( $(echo "$x < 0" | bc -l) )); then
    echo "Warning: prey population negative at t=$t" >&2
  fi
  if (( $(echo "$y < 0" | bc -l) )); then
    echo "Warning: predator population negative at t=$t" >&2
  fi
done

# Compute equilibrium
prey_eq=$(python3 -c "print($C / $D)")
pred_eq=$(python3 -c "print($A / $B)")

cat > "$WORKSPACE/equilibrium.json" <<EOF
{
  "prey": $prey_eq,
  "predator": $pred_eq
}
EOF
