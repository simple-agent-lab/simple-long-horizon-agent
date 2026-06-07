#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, json
objectives = [
    ('OBJ-01','Understand basic algebraic operations','Understand','algebra'),
    ('OBJ-02','Apply linear equation solving techniques','Apply','algebra'),
    ('OBJ-03','Analyze statistical distributions','Analyze','statistics'),
    ('OBJ-04','Remember key historical dates and events','Remember','history'),
    ('OBJ-05','Evaluate scientific experiment designs','Evaluate','science'),
    ('OBJ-06','Create data visualizations from datasets','Create','data_science'),
    ('OBJ-07','Understand cell biology fundamentals','Understand','biology'),
    ('OBJ-08','Apply Newton laws to physical problems','Apply','physics'),
    ('OBJ-09','Analyze literary narrative structures','Analyze','literature'),
    ('OBJ-10','Evaluate economic policy impacts','Evaluate','economics'),
]
content = {
    'algebra': ['variables','equations','inequalities','polynomials','factoring'],
    'statistics': ['mean','median','mode','standard deviation','normal distribution'],
    'history': ['world wars','industrial revolution','civil rights','cold war'],
    'science': ['scientific method','hypothesis','variables','control groups'],
    'data_science': ['charts','graphs','scatter plots','bar charts','dashboards'],
    'biology': ['cells','mitosis','DNA','proteins','organelles'],
    'physics': ['force','mass','acceleration','gravity','momentum'],
    'literature': ['plot','character','theme','symbolism','narrative'],
    'economics': ['supply','demand','GDP','inflation','fiscal policy'],
}
with open('$WORKSPACE/learning_objectives.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['objective_id','description','bloom_level','topic'])
    for o in objectives: w.writerow(o)
with open('$WORKSPACE/content_outline.json','w') as f:
    json.dump(content, f, indent=2)
"
