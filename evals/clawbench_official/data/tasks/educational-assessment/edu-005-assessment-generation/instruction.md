You are given `workspace/learning_objectives.csv` (objective_id, description, bloom_level, topic) and `workspace/content_outline.json` (topic -> key_concepts list).

**Your task:**
1. Read both files
2. For each learning objective, generate 3 questions at different difficulty levels:
   - Easy: factual recall (Remember/Understand bloom levels)
   - Medium: application (Apply/Analyze bloom levels)
   - Hard: evaluation/creation (Evaluate/Create bloom levels)
3. Each question should reference key concepts from the content outline
4. Write `workspace/question_bank.json`:
```json
{
  "questions": [
    {"id": "Q001", "objective_id": "OBJ-01", "difficulty": "easy", "question": "...", "answer": "...", "bloom_level": "Remember", "topic": "algebra"},
    ...
  ],
  "statistics": {
    "total_questions": 30,
    "by_difficulty": {"easy": 10, "medium": 10, "hard": 10},
    "by_topic": {"algebra": 6, ...},
    "objectives_covered": 10
  }
}
```
