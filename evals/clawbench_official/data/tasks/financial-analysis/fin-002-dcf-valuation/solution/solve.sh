#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
rows = []
with open('$WORKSPACE/financials.csv') as f:
    price_line = f.readline()
    current_price = float(price_line.split('=')[1])
    reader = csv.DictReader(f)
    rows = list(reader)
for r in rows:
    for k in r:
        if k != 'year':
            r[k] = float(r[k])
# FCF = EBITDA - Capex - WC change
fcfs = [r['ebitda'] - r['capex'] - r['working_capital_change'] for r in rows]
# Growth rate from last 3 years
growths = [(fcfs[i]-fcfs[i-1])/abs(fcfs[i-1]) for i in range(-2,0)]
avg_growth = sum(growths)/len(growths)
avg_growth = max(0.02, min(avg_growth, 0.25))
last_fcf = fcfs[-1]
projected = [round(last_fcf * (1+avg_growth)**(i+1), 2) for i in range(5)]
# WACC
latest = rows[-1]
debt_ratio = latest['debt'] / (latest['debt'] + current_price * latest['shares_outstanding'])
equity_ratio = 1 - debt_ratio
wacc = round(equity_ratio * 0.10 + debt_ratio * 0.05 * (1-0.25), 4)
# Terminal value
tv = round(projected[-1] * (1+0.02) / (wacc - 0.02), 2)
# Enterprise value
ev = sum(p / (1+wacc)**(i+1) for i, p in enumerate(projected))
ev += tv / (1+wacc)**5
ev = round(ev, 2)
equity_val = round(ev - latest['debt'] + latest['cash'], 2)
ivps = round(equity_val / latest['shares_outstanding'], 2)
if ivps > current_price * 1.1:
    summary = 'undervalued'
elif ivps < current_price * 0.9:
    summary = 'overvalued'
else:
    summary = 'fairly_valued'
json.dump({'projected_fcf': projected, 'wacc': wacc, 'terminal_value': tv,
           'enterprise_value': ev, 'equity_value': equity_val,
           'intrinsic_value_per_share': ivps, 'valuation_summary': summary},
          open('$WORKSPACE/dcf_valuation.json','w'), indent=2)
"
