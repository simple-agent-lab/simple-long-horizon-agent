#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random
random.seed(42)
currencies = ['USD','EUR','GBP','JPY','CHF','AUD','CAD','NZD']
base_rates = {'USD':1.0,'EUR':1.08,'GBP':1.27,'JPY':0.0067,'CHF':1.13,'AUD':0.65,'CAD':0.74,'NZD':0.61}
with open('$WORKSPACE/exchange_rates.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['from/to'] + currencies)
    for c1 in currencies:
        row = [c1]
        for c2 in currencies:
            if c1 == c2:
                row.append('1.0')
            else:
                true_rate = base_rates[c2] / base_rates[c1]
                noise = random.uniform(-0.003, 0.003)
                row.append(f'{true_rate + noise:.6f}')
        w.writerow(row)
"
