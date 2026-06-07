#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv, json, random
random.seed(42)
topics = ['arithmetic','algebra','geometry','trigonometry','calculus','statistics','probability','linear_algebra']
prereqs = {
    'arithmetic': [],
    'algebra': ['arithmetic'],
    'geometry': ['arithmetic'],
    'trigonometry': ['geometry','algebra'],
    'calculus': ['algebra','trigonometry'],
    'statistics': ['algebra'],
    'probability': ['algebra','statistics'],
    'linear_algebra': ['algebra']
}
with open('$WORKSPACE/prerequisites.json','w') as f:
    json.dump(prereqs, f, indent=2)
with open('$WORKSPACE/student_performance.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['student_id','topic','score','max_score'])
    for i in range(15):
        sid = f'S{i+1:03d}'
        for t in topics:
            max_s = 100
            if t in ['arithmetic','geometry']:
                score = random.randint(60, 100)
            elif t in ['calculus','linear_algebra','probability']:
                score = random.randint(20, 80)
            else:
                score = random.randint(35, 90)
            w.writerow([sid, t, score, max_s])
"
