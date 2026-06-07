#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
from collections import defaultdict
accounts = defaultdict(lambda: {'debit': 0, 'credit': 0})
with open('$WORKSPACE/ledger.csv') as f:
    for r in csv.DictReader(f):
        accounts[r['account']]['debit'] += float(r['debit'] or 0)
        accounts[r['account']]['credit'] += float(r['credit'] or 0)
with open('$WORKSPACE/trial_balance.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['account','total_debit','total_credit','balance'])
    for acct in sorted(accounts):
        d, c = accounts[acct]['debit'], accounts[acct]['credit']
        w.writerow([acct, round(d,2), round(c,2), round(d-c,2)])
td = sum(v['debit'] for v in accounts.values())
tc = sum(v['credit'] for v in accounts.values())
balanced = abs(td - tc) < 0.01
json.dump({'total_accounts': len(accounts), 'total_debits': round(td,2),
           'total_credits': round(tc,2), 'is_balanced': balanced,
           'imbalance_amount': round(abs(td-tc),2), 'accounts_with_issues': []},
          open('$WORKSPACE/validation_report.json','w'), indent=2)
"
