#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
with open('$WORKSPACE/assets.csv') as f:
    assets = list(csv.DictReader(f))
rows = []
summary = {}
for a in assets:
    name = a['asset_name']
    cost = float(a['cost'])
    salvage = float(a['salvage_value'])
    life = int(a['useful_life_years'])
    dep_base = cost - salvage
    summary[name] = {}
    # Straight-line
    annual = dep_base / life
    acc = 0
    for y in range(1, life+1):
        acc += annual
        rows.append([name, 'straight-line', y, round(annual,2), round(acc,2), round(cost-acc,2)])
    summary[name]['straight-line'] = round(dep_base, 2)
    # Double declining
    rate = 2.0 / life
    bv = cost
    acc = 0
    for y in range(1, life+1):
        dep = min(round(bv * rate, 2), bv - salvage)
        dep = max(dep, 0)
        acc += dep
        bv -= dep
        rows.append([name, 'double-declining', y, round(dep,2), round(acc,2), round(bv,2)])
    summary[name]['double-declining'] = round(acc, 2)
    # Sum of years
    soy = life * (life+1) / 2
    acc = 0
    for y in range(1, life+1):
        frac = (life - y + 1) / soy
        dep = round(dep_base * frac, 2)
        acc += dep
        rows.append([name, 'sum-of-years', y, dep, round(acc,2), round(cost-acc,2)])
    summary[name]['sum-of-years'] = round(acc, 2)
with open('$WORKSPACE/depreciation_schedules.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['asset','method','year','depreciation','accumulated','book_value'])
    w.writerows(rows)
json.dump(summary, open('$WORKSPACE/depreciation_summary.json','w'), indent=2)
"
