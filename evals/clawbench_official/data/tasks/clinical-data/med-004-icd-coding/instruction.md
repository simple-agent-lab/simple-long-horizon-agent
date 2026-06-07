# Clinical Notes ICD-10 Coding

## Task Description

You are provided with a workspace containing two types of data:

- A directory `workspace/clinical_notes/` containing multiple text files. Each file contains clinical notes for a patient encounter.
- A CSV file `workspace/icd_mapping.csv` which contains mappings from medical keywords to ICD-10 codes and their descriptions.

Your task is to:

1. Read all text files in the `workspace/clinical_notes/` directory.
2. Read the `icd_mapping.csv` file which has three columns: `keyword`, `icd_code`, and `description`.
3. For each clinical note file, identify all keywords from the mapping that appear in the text (case-insensitive).
4. Assign the corresponding ICD-10 codes to that clinical note.
5. Produce a JSON file `workspace/coded_records.json` that maps each clinical note filename to a list of assigned ICD-10 codes.

## Input

- `workspace/clinical_notes/` directory with multiple `.txt` files containing clinical notes.
- `workspace/icd_mapping.csv` file with columns:
  - `keyword` (string)
  - `icd_code` (string)
  - `description` (string)

## Output

- A JSON file `workspace/coded_records.json` with the structure:

```json
{
  "note1.txt": ["I10", "E11.9"],
  "note2.txt": ["J45.909"],
  ...
}
```

Each key is a clinical note filename, and the value is a list of ICD-10 codes matched from the keywords found in that note.

## Requirements

- Keyword matching should be case-insensitive.
- If a keyword appears multiple times in a note, its ICD-10 code should only appear once in the output list.
- Notes with no matching keywords should have an empty list as their value.
- The output JSON should be pretty-printed with indentation for readability.

## Evaluation

Your solution will be tested on synthetic clinical notes and ICD mappings. Correctness will be judged on accurate keyword matching and correct ICD-10 code assignment per note.

---

Good luck!
