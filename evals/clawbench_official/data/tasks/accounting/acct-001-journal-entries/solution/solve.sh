#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
account_map = {
    'sale': ('Cash', 'Revenue'),
    'purchase': ('Inventory', 'Accounts Payable'),
    'salary': ('Salary Expense', 'Cash'),
    'rent': ('Rent Expense', 'Cash'),
    'loan': ('Cash', 'Loan Payable')
}
with open('$WORKSPACE/transactions.csv') as f:
    txns = list(csv.DictReader(f))
entries = []
for i, t in enumerate(txns):
    debit_acct, credit_acct = account_map[t['type']]
    amt = float(t['amount'])
    eid = f'JE-{i+1:03d}'
    entries.append({'date': t['date'], 'entry_id': eid, 'account': debit_acct, 'debit': amt, 'credit': 0})
    entries.append({'date': t['date'], 'entry_id': eid, 'account': credit_acct, 'debit': 0, 'credit': amt})
with open('$WORKSPACE/journal.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['date','entry_id','account','debit','credit'])
    w.writeheader()
    w.writerows(entries)
td = sum(e['debit'] for e in entries)
tc = sum(e['credit'] for e in entries)
json.dump({'total_debits': round(td,2), 'total_credits': round(tc,2), 'entry_count': len(entries)},
          open('$WORKSPACE/journal_summary.json','w'), indent=2)
"
