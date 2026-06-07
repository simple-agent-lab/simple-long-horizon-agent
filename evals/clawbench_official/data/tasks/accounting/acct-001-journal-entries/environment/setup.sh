#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random
random.seed(42)
types = ['sale','purchase','salary','rent','loan']
with open('$WORKSPACE/transactions.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['date','description','amount','type'])
    for i in range(20):
        t = random.choice(types)
        amt = round(random.uniform(500, 15000), 2)
        d = f'2024-01-{i+1:02d}'
        desc = f'{t.title()} transaction #{i+1}'
        w.writerow([d, desc, amt, t])
"
