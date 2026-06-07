#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, random
random.seed(42)
accounts = ['Cash','Accounts Receivable','Inventory','Equipment','Accounts Payable',
            'Revenue','COGS','Salary Expense','Rent Expense','Utilities Expense']
with open('$WORKSPACE/ledger.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['date','account','debit','credit'])
    for i in range(80):
        d = f'2024-{(i//28)+1:02d}-{(i%28)+1:02d}'
        acct = random.choice(accounts)
        amt = round(random.uniform(100, 5000), 2)
        if acct in ['Cash','Accounts Receivable','Inventory','Equipment','COGS','Salary Expense','Rent Expense','Utilities Expense']:
            w.writerow([d, acct, amt, 0])
            cr_acct = random.choice(['Revenue','Accounts Payable','Cash'])
            w.writerow([d, cr_acct, 0, amt])
        else:
            w.writerow([d, acct, 0, amt])
            dr_acct = random.choice(['Cash','Inventory'])
            w.writerow([d, dr_acct, amt, 0])
"
