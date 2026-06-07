#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json, math, statistics
with open('$WORKSPACE/portfolio_returns.csv') as f:
    reader = csv.DictReader(f)
    returns = [float(r['return']) for r in reader]
mean_r = statistics.mean(returns)
std_r = statistics.stdev(returns)
ann_ret = round(mean_r * 252, 6)
ann_vol = round(std_r * math.sqrt(252), 6)
sorted_r = sorted(returns)
var95 = round(sorted_r[int(len(sorted_r)*0.05)], 6)
var99 = round(sorted_r[int(len(sorted_r)*0.01)], 6)
sharpe = round((ann_ret - 0.04) / ann_vol, 4)
cum = 0
peak = 0
max_dd = 0
for r in returns:
    cum += r
    if cum > peak: peak = cum
    dd = cum - peak
    if dd < max_dd: max_dd = dd
if sharpe < 0.5: rating = 'poor'
elif sharpe < 1.0: rating = 'moderate'
elif sharpe < 2.0: rating = 'good'
else: rating = 'excellent'
json.dump({'mean_daily_return': round(mean_r,6), 'daily_std': round(std_r,6),
           'annualized_return': ann_ret, 'annualized_volatility': ann_vol,
           'var_95': var95, 'var_99': var99, 'sharpe_ratio': sharpe,
           'max_drawdown': round(max_dd,6), 'risk_rating': rating},
          open('$WORKSPACE/risk_report.json','w'), indent=2)
"
