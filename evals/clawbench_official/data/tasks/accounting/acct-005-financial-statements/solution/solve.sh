#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
with open('$WORKSPACE/trial_balance.csv') as f:
    rows = list(csv.DictReader(f))
revenue = [(r['account'], float(r['credit_balance'])) for r in rows if r['type']=='revenue']
expenses = [(r['account'], float(r['debit_balance'])) for r in rows if r['type']=='expense']
total_rev = sum(v for _,v in revenue)
total_exp = sum(v for _,v in expenses)
ni = total_rev - total_exp
with open('$WORKSPACE/income_statement.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['category','account','amount'])
    for a,v in revenue: w.writerow(['Revenue',a,v])
    w.writerow(['Total Revenue','',total_rev])
    for a,v in expenses: w.writerow(['Expense',a,v])
    w.writerow(['Total Expenses','',total_exp])
    w.writerow(['Net Income','',ni])
assets = [(r['account'], float(r['debit_balance'])) for r in rows if r['type']=='asset']
liabs = [(r['account'], float(r['credit_balance'])) for r in rows if r['type']=='liability']
equity = [(r['account'], float(r['credit_balance'])) for r in rows if r['type']=='equity']
ta = sum(v for _,v in assets)
tl = sum(v for _,v in liabs)
te = sum(v for _,v in equity)
with open('$WORKSPACE/balance_sheet.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['category','account','amount'])
    for a,v in assets: w.writerow(['Asset',a,v])
    w.writerow(['Total Assets','',ta])
    for a,v in liabs: w.writerow(['Liability',a,v])
    w.writerow(['Total Liabilities','',tl])
    for a,v in equity: w.writerow(['Equity',a,v])
    w.writerow(['Retained Earnings (NI)','',ni])
    w.writerow(['Total Equity','',te+ni])
balanced = abs(ta - (tl + te + ni)) < 1.0
json.dump({'total_revenue':total_rev,'total_expenses':total_exp,'net_income':ni,
           'total_assets':ta,'total_liabilities':tl,'total_equity':te,
           'equation_balanced':balanced}, open('$WORKSPACE/financial_summary.json','w'), indent=2)
"
