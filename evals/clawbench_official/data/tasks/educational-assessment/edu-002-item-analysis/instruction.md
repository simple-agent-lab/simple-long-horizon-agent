# Exam Item Difficulty and Discrimination Analysis

You are provided with a CSV file named `exam_responses.csv` located in the workspace directory. This file contains exam response data for multiple students. The columns are:

- `student_id`: Unique identifier for each student
- `item_1` to `item_30`: Binary responses to 30 exam items (1 = correct, 0 = incorrect)
- `total_score`: Total score of the student across all items

Your task is to perform an item analysis by computing the following statistics for each item:

1. **Difficulty (p-value):** The proportion of students who answered the item correctly.
2. **Discrimination Index:** The difference between the proportion of correct responses in the top 27% of scorers and the bottom 27% of scorers.
3. **Point-Biserial Correlation:** The correlation between the item score (0/1) and the total test score.

You must write two output files:

- `item_analysis.csv`: A CSV file with columns:
  - `item`: Item name (e.g., `item_1`)
  - `difficulty`: Difficulty (p-value) of the item
  - `discrimination_index`: Discrimination index of the item
  - `point_biserial`: Point-biserial correlation of the item

- `exam_summary.json`: A JSON file summarizing the exam with the following keys:
  - `num_students`: Number of students
  - `num_items`: Number of items (30)
  - `mean_total_score`: Mean total score across all students
  - `std_total_score`: Standard deviation of total scores

**Requirements:**

- Read the input file `workspace/exam_responses.csv`.
- Perform the computations as described.
- Write the output files `workspace/item_analysis.csv` and `workspace/exam_summary.json`.

**Notes:**

- Use the top 27% and bottom 27% of students based on `total_score` for the discrimination index calculation.
- The point-biserial correlation is the Pearson correlation between the binary item response and the total score.
- Ensure output files are valid and correctly formatted.

Good luck!