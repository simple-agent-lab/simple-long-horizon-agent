You are given `workspace/course_objectives.csv` (objective_id, description, bloom_level) and `workspace/standards.csv` (standard_id, description, domain).

**Your task:**
1. Read both files
2. Map each objective to the most relevant standard(s) by keyword overlap (Jaccard similarity on word sets)
3. Write `workspace/alignment_matrix.csv` (objective_id, standard_id, similarity_score)
4. Write `workspace/gap_analysis.json`:
```json
{
  "total_objectives": 20,
  "total_standards": 15,
  "mapped_objectives": 18,
  "unmapped_objectives": ["OBJ-19", "OBJ-20"],
  "uncovered_standards": ["STD-14"],
  "coverage_rate": 0.93,
  "alignment_summary": [{"objective_id": "OBJ-01", "best_match": "STD-03", "score": 0.65}, ...]
}
```
Use a threshold of 0.15 for a valid mapping.
