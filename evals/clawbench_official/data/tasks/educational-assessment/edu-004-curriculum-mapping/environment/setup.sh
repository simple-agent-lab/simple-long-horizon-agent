#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"
mkdir -p "$WORKSPACE"
python3 -c "
import csv
objectives = [
    ('OBJ-01','Solve linear equations and inequalities','Apply'),
    ('OBJ-02','Analyze statistical data distributions','Analyze'),
    ('OBJ-03','Write persuasive essays with evidence','Create'),
    ('OBJ-04','Evaluate historical primary sources','Evaluate'),
    ('OBJ-05','Design controlled scientific experiments','Create'),
    ('OBJ-06','Calculate derivatives and integrals','Apply'),
    ('OBJ-07','Compare economic systems and policies','Analyze'),
    ('OBJ-08','Interpret literary themes and symbolism','Analyze'),
    ('OBJ-09','Model physical forces and motion','Apply'),
    ('OBJ-10','Construct geometric proofs','Create'),
    ('OBJ-11','Analyze chemical reaction equations','Analyze'),
    ('OBJ-12','Evaluate environmental impact data','Evaluate'),
    ('OBJ-13','Program basic algorithms in Python','Apply'),
    ('OBJ-14','Synthesize research from multiple sources','Create'),
    ('OBJ-15','Apply probability to real-world scenarios','Apply'),
    ('OBJ-16','Critique artistic movements and styles','Evaluate'),
    ('OBJ-17','Design database schemas','Create'),
    ('OBJ-18','Analyze biological cell processes','Analyze'),
    ('OBJ-19','Compose original musical arrangements','Create'),
    ('OBJ-20','Choreograph dance performances','Create'),
]
standards = [
    ('STD-01','Students solve equations and inequalities in algebra','Mathematics'),
    ('STD-02','Students analyze and interpret statistical data','Mathematics'),
    ('STD-03','Students write argumentative and persuasive texts','English'),
    ('STD-04','Students evaluate historical documents and sources','Social Studies'),
    ('STD-05','Students design and conduct scientific investigations','Science'),
    ('STD-06','Students apply calculus concepts including derivatives','Mathematics'),
    ('STD-07','Students compare political and economic systems','Social Studies'),
    ('STD-08','Students analyze literary works for themes','English'),
    ('STD-09','Students model forces motion and energy in physics','Science'),
    ('STD-10','Students construct and verify geometric proofs','Mathematics'),
    ('STD-11','Students analyze chemical reactions and equations','Science'),
    ('STD-12','Students evaluate environmental science data','Science'),
    ('STD-13','Students apply programming and algorithmic thinking','Technology'),
    ('STD-14','Students demonstrate physical fitness and wellness','Physical Education'),
    ('STD-15','Students apply probability and statistics concepts','Mathematics'),
]
with open('$WORKSPACE/course_objectives.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['objective_id','description','bloom_level'])
    for o in objectives: w.writerow(o)
with open('$WORKSPACE/standards.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['standard_id','description','domain'])
    for s in standards: w.writerow(s)
"
