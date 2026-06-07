# Student Essay Rubric-Based Grading

## Overview

You are given a directory `workspace/essays/` containing multiple student essay text files. Each file is named with the student's identifier (e.g., `student1.txt`, `student2.txt`, etc.).

You are also given a CSV file `workspace/rubric.csv` which defines the grading rubric. This CSV contains the following columns:

- `criterion`: The name of the grading criterion (e.g., "Clarity", "Argument Strength", "Grammar").
- `max_points`: The maximum points achievable for this criterion (integer).
- `keywords`: A semicolon-separated list of keywords relevant to this criterion.

## Task

Your task is to:

1. Read all student essays from the `workspace/essays/` directory.
2. Read the rubric from `workspace/rubric.csv`.
3. For each student essay and each rubric criterion:
   - Score the essay based on:
     - **Keyword presence:** Check how many of the criterion's keywords appear in the essay text (case-insensitive). Award points proportionally to the fraction of keywords found.
     - **Text quality metrics:** Compute the average sentence length (in words) and the spelling error count (you may use a simple dictionary-based approach or a standard English word list). Use these metrics to adjust the score:
       - Longer average sentence length (between 10 and 20 words) is better.
       - Fewer spelling errors increase the score.

4. Combine keyword presence and text quality metrics to compute a final score per criterion, capped at `max_points`.

5. Write the results to a JSON file `workspace/grades.json` with the following structure:

```json
{
  "student1": {
    "Clarity": 8.5,
    "Argument Strength": 7.0,
    "Grammar": 9.0
  },
  "student2": {
    "Clarity": 6.0,
    "Argument Strength": 8.5,
    "Grammar": 7.5
  },
  ...
}
```

## Details and Requirements

- Keyword matching should be case-insensitive and match whole words only.
- Average sentence length is the total number of words divided by the number of sentences.
- For spelling errors, consider any word not in a standard English word list as an error.
- The final score per criterion should be a float rounded to one decimal place.
- You may design your own reasonable formula to combine keyword presence and text quality metrics.
- The output JSON must include all students and all rubric criteria.

## Deliverables

- A JSON file `workspace/grades.json` as described.


---

Good luck!