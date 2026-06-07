#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
with open('$WORKSPACE/exchange_rates.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    currencies = header[1:]
    rates = {}
    for row in reader:
        c1 = row[0]
        for i, c2 in enumerate(currencies):
            rates[(c1, c2)] = float(row[i+1])
opps = []
for a in currencies:
    for b in currencies:
        if b == a: continue
        for c in currencies:
            if c == a or c == b: continue
            product = rates[(a,b)] * rates[(b,c)] * rates[(c,a)]
            if product > 1.0:
                profit = round((product - 1) * 100, 4)
                opps.append({'path': [a,b,c,a], 'product': round(product,6), 'profit_pct': profit})
opps.sort(key=lambda x: -x['profit_pct'])
best = opps[0] if opps else None
json.dump({'opportunities': opps, 'total_opportunities': len(opps),
           'best_opportunity': best}, open('$WORKSPACE/arbitrage_results.json','w'), indent=2)
"
