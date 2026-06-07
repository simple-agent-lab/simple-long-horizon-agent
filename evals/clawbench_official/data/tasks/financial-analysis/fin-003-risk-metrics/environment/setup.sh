#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random, math
random.seed(42)
with open('$WORKSPACE/portfolio_returns.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['date','return'])
    for i in range(500):
        d = f'2023-{(i//21)+1:02d}-{(i%21)+1:02d}'
        r = random.gauss(0.0005, 0.015)
        w.writerow([d, f'{r:.6f}'])
"
