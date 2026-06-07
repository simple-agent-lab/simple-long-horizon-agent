#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
python3 -c "
import csv, json, re
with open('$WORKSPACE/course_objectives.csv') as f:
    objs = list(csv.DictReader(f))
with open('$WORKSPACE/standards.csv') as f:
    stds = list(csv.DictReader(f))
def words(text):
    return set(re.findall(r'\w+', text.lower())) - {'and','the','in','of','to','a','for','with','from','students'}
alignments = []
summary = []
mapped_objs = set()
covered_stds = set()
for o in objs:
    ow = words(o['description'])
    best_std = None
    best_score = 0
    for s in stds:
        sw = words(s['description'])
        if len(ow | sw) == 0: continue
        jaccard = len(ow & sw) / len(ow | sw)
        if jaccard > best_score:
            best_score = jaccard
            best_std = s['standard_id']
        if jaccard >= 0.15:
            alignments.append({'objective_id': o['objective_id'], 'standard_id': s['standard_id'], 'similarity_score': round(jaccard, 4)})
    if best_score >= 0.15:
        mapped_objs.add(o['objective_id'])
        covered_stds.add(best_std)
    summary.append({'objective_id': o['objective_id'], 'best_match': best_std or 'none', 'score': round(best_score, 4)})
with open('$WORKSPACE/alignment_matrix.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['objective_id','standard_id','similarity_score'])
    w.writeheader()
    w.writerows(alignments)
unmapped = [o['objective_id'] for o in objs if o['objective_id'] not in mapped_objs]
uncovered = [s['standard_id'] for s in stds if s['standard_id'] not in covered_stds]
json.dump({'total_objectives': len(objs), 'total_standards': len(stds),
           'mapped_objectives': len(mapped_objs), 'unmapped_objectives': unmapped,
           'uncovered_standards': uncovered, 'coverage_rate': round(len(mapped_objs)/len(objs), 4),
           'alignment_summary': summary}, open('$WORKSPACE/gap_analysis.json','w'), indent=2)
"
