You are given `workspace/student_performance.csv` (student_id, topic, score, max_score) and `workspace/prerequisites.json` (topic -> list of prerequisite topics).

**Your task:**
1. Read both files
2. For each student, identify knowledge gaps: topics where score < 60% of max_score
3. Using the prerequisite graph, build a learning path: if a student has a gap in topic X, and X requires Y, and Y also has a gap, Y should come before X
4. Write `workspace/learning_plans.json`:
```json
{
  "students": {
    "S001": {
      "gaps": ["algebra", "calculus"],
      "mastered": ["arithmetic", "geometry"],
      "learning_path": ["algebra", "calculus"],
      "gap_count": 2
    },
    ...
  },
  "summary": {
    "total_students": 10,
    "avg_gap_count": 2.5,
    "most_common_gap": "algebra"
  }
}
```
