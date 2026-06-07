#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random
random.seed(42)
# current_price = 95.0
with open('$WORKSPACE/financials.csv','w',newline='') as f:
    f.write('# current_price=95.0\n')
    w = csv.writer(f)
    w.writerow(['year','revenue','ebitda','capex','depreciation','working_capital_change','debt','cash','shares_outstanding'])
    rev = 10000
    for y in range(2019, 2024):
        growth = random.uniform(0.05, 0.15)
        rev = round(rev * (1 + growth))
        ebitda = round(rev * random.uniform(0.18, 0.25))
        capex = round(rev * random.uniform(0.04, 0.08))
        dep = round(capex * 0.7)
        wc = round(rev * random.uniform(-0.02, 0.03))
        debt = round(rev * random.uniform(0.2, 0.35))
        cash = round(rev * random.uniform(0.05, 0.12))
        shares = 500
        w.writerow([y, rev, ebitda, capex, dep, wc, debt, cash, shares])
"
