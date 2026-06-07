#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv
assets = [('Office Equipment', 50000, 5000, 5), ('Vehicle', 35000, 7000, 7),
          ('Machinery', 120000, 10000, 10), ('Furniture', 15000, 1500, 8)]
with open('$WORKSPACE/assets.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['asset_name','cost','salvage_value','useful_life_years'])
    for a in assets:
        w.writerow(a)
"
