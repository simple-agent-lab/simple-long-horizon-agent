#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
from datetime import datetime, timedelta
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))
bank = read_csv('$WORKSPACE/bank_statement.csv')
ledger = read_csv('$WORKSPACE/company_ledger.csv')
matched = []
bank_used = set()
ledger_used = set()
for i, b in enumerate(bank):
    for j, l in enumerate(ledger):
        if j in ledger_used: continue
        if abs(float(b['amount']) - float(l['amount'])) < 0.01:
            bd = datetime.strptime(b['date'], '%Y-%m-%d')
            ld = datetime.strptime(l['date'], '%Y-%m-%d')
            if abs((bd-ld).days) <= 2:
                matched.append((i,j))
                bank_used.add(i)
                ledger_used.add(j)
                break
outstanding = [ledger[j] for j in range(len(ledger)) if j not in ledger_used and float(ledger[j]['amount']) < 0]
deposits = [ledger[j] for j in range(len(ledger)) if j not in ledger_used and float(ledger[j]['amount']) > 0]
bank_only = [bank[i] for i in range(len(bank)) if i not in bank_used]
bank_bal = sum(float(b['amount']) for b in bank)
book_bal = sum(float(l['amount']) for l in ledger)
adj_bank = bank_bal - sum(float(o['amount']) for o in outstanding) + sum(float(d['amount']) for d in deposits)
adj_book = book_bal + sum(float(b['amount']) for b in bank_only)
json.dump({'matched_count': len(matched),
           'outstanding_checks': [{'ref': o['reference'], 'amount': float(o['amount'])} for o in outstanding],
           'deposits_in_transit': [{'ref': d['reference'], 'amount': float(d['amount'])} for d in deposits],
           'bank_charges': [{'ref': b['reference'], 'amount': float(b['amount'])} for b in bank_only],
           'adjusted_bank_balance': round(adj_bank, 2),
           'adjusted_book_balance': round(adj_book, 2),
           'is_reconciled': abs(adj_bank - adj_book) < 1.0},
          open('$WORKSPACE/reconciliation.json','w'), indent=2)
"
