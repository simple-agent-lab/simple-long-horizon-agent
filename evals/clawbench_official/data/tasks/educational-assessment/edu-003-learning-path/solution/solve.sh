#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json
from collections import Counter, defaultdict
with open('$WORKSPACE/prerequisites.json') as f:
    prereqs = json.load(f)
students = defaultdict(dict)
with open('$WORKSPACE/student_performance.csv') as f:
    for r in csv.DictReader(f):
        students[r['student_id']][r['topic']] = (int(r['score']), int(r['max_score']))
result = {'students': {}, 'summary': {}}
gap_counter = Counter()
for sid in sorted(students):
    gaps = []
    mastered = []
    for topic, (score, max_s) in students[sid].items():
        if score < max_s * 0.6:
            gaps.append(topic)
            gap_counter[topic] += 1
        else:
            mastered.append(topic)
    # Topological sort of gaps based on prerequisites
    path = []
    visited = set()
    def visit(t):
        if t in visited: return
        visited.add(t)
        for p in prereqs.get(t, []):
            if p in gaps:
                visit(p)
        path.append(t)
    for g in gaps:
        visit(g)
    result['students'][sid] = {'gaps': gaps, 'mastered': mastered, 'learning_path': path, 'gap_count': len(gaps)}
n = len(students)
avg_gaps = round(sum(d['gap_count'] for d in result['students'].values()) / n, 2)
most_common = gap_counter.most_common(1)[0][0] if gap_counter else ''
result['summary'] = {'total_students': n, 'avg_gap_count': avg_gaps, 'most_common_gap': most_common}
json.dump(result, open('$WORKSPACE/learning_plans.json','w'), indent=2)
"
