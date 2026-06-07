# Research Paper Metadata Extraction

## Task Description

You are given a workspace containing a directory `papers/` with multiple text files. Each file simulates an academic research paper with structured sections including:

- Title
- Authors
- Year
- Keywords
- Abstract
- References

Your task is to:

1. Read all text files in the `papers/` directory.
2. Extract the following metadata from each paper:
   - Title
   - Authors (as a list)
   - Year (integer)
   - Keywords (as a list)
   - Abstract (string)
   - References (as a list of strings)
3. Write two output files in the workspace root:
   - `bibliography.csv`: a CSV file with columns `title`, `authors` (semicolon-separated), `year`, `keywords` (semicolon-separated)
   - `metadata.json`: a JSON file containing a list of paper metadata objects, each with keys `title`, `authors`, `year`, `keywords`, `abstract`, and `references`.

## Input

- Directory: `workspace/papers/`
- Each file is a UTF-8 encoded text file with the following format:

```
Title: <paper title>
Authors: <author1>; <author2>; ...
Year: <year>
Keywords: <keyword1>; <keyword2>; ...

Abstract:
<abstract text possibly spanning multiple lines>

References:
<reference1>
<reference2>
...
```

- There is a blank line before `Abstract:` and before `References:` sections.
- The abstract and references sections may span multiple lines.

## Output

- `bibliography.csv` with columns:
  - title
  - authors (semicolon-separated)
  - year
  - keywords (semicolon-separated)

- `metadata.json` containing a JSON array of paper metadata objects with keys:
  - title (string)
  - authors (list of strings)
  - year (integer)
  - keywords (list of strings)
  - abstract (string)
  - references (list of strings)

## Requirements

- Your solution must correctly parse all papers in the `papers/` directory.
- Preserve the order of papers as sorted by filename.
- Trim whitespace around extracted fields.
- Handle multi-line abstract and references sections.
- Output files must be valid UTF-8.

## Evaluation

Your output files will be verified for correctness of extracted metadata and formatting.

Good luck!