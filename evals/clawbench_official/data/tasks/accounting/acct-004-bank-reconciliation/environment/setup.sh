#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random
random.seed(42)
# Generate bank statement
bank = []
ledger = []
for i in range(30):
    d = f'2024-01-{i+1:02d}'
    amt = round(random.uniform(-5000, 8000), 2)
    bank.append((d, f'TXN-{i+1}', amt))
    if random.random() > 0.15:  # 85% match
        offset = random.randint(0, 2)
        ld = f'2024-01-{min(i+1+offset,30):02d}'
        ledger.append((ld, f'CHK-{i+1}', amt))
# Add some ledger-only items (outstanding)
for i in range(3):
    ledger.append((f'2024-01-{28+i:02d}', f'CHK-OUT-{i}', round(random.uniform(-3000, -500), 2)))
# Add bank-only items (bank charges)
bank.append(('2024-01-15', 'BANK-FEE', -25.00))
bank.append(('2024-01-30', 'BANK-INT', 12.50))
with open('$WORKSPACE/bank_statement.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['date','reference','amount'])
    for r in bank: w.writerow(r)
with open('$WORKSPACE/company_ledger.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['date','reference','amount'])
    for r in ledger: w.writerow(r)
"
