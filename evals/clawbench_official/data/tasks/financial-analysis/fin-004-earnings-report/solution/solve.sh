#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json, re

def parse_report(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                val = val.strip().replace('%', '').replace(',', '')
                try:
                    data[key] = float(val)
                except:
                    pass
    return data

q1 = parse_report('$WORKSPACE/earnings_q1.txt')
q2 = parse_report('$WORKSPACE/earnings_q2.txt')
metrics = ['revenue','cost_of_revenue','gross_profit','operating_income','net_income','eps','gross_margin','operating_margin','net_margin']
with open('$WORKSPACE/earnings_comparison.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric','q1_value','q2_value','change','change_pct'])
    for m in metrics:
        v1 = q1.get(m, 0)
        v2 = q2.get(m, 0)
        chg = round(v2 - v1, 2)
        pct = round(chg/v1*100, 2) if v1 != 0 else 0
        w.writerow([m, v1, v2, chg, pct])
trend = 'mixed'
if q2.get('revenue',0) > q1.get('revenue',0) and q2.get('net_income',0) > q1.get('net_income',0):
    trend = 'improving'
elif q2.get('revenue',0) < q1.get('revenue',0) and q2.get('net_income',0) < q1.get('net_income',0):
    trend = 'declining'
json.dump({'q1': {k: q1.get(k,0) for k in ['revenue','net_income','eps']},
           'q2': {k: q2.get(k,0) for k in ['revenue','net_income','eps']},
           'trend': trend}, open('$WORKSPACE/earnings_summary.json','w'), indent=2)
"
