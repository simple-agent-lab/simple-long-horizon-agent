# Translation Quality Assessment

## Description

You are given two directories inside the workspace:

- `source/`: Contains original source text files.
- `translation/`: Contains translated text files corresponding to the source files.

Each file in `translation/` corresponds to a file with the same name in `source/`.

Your task is to:

1. Read all paired files from `source/` and `translation/`.
2. Compute a BLEU-like score for each pair, based on n-gram precision (up to 4-grams) of the translated text against the source text.
3. Identify and flag any missing segments (lines present in source but missing in translation) or added segments (lines present in translation but missing in source).
4. Write a JSON file `translation_qa.json` in the workspace root with the following structure:

```json
{
  "files": {
    "filename1.txt": {
      "bleu_score": 0.75,
      "missing_segments": [2, 5],
      "added_segments": [8]
    },
    "filename2.txt": {
      "bleu_score": 0.60,
      "missing_segments": [],
      "added_segments": [3]
    }
  },
  "overall_score": 0.68,
  "flagged_issues": {
    "missing_segments_files": ["filename1.txt"],
    "added_segments_files": ["filename1.txt", "filename2.txt"]
  }
}
```

## Details

- The BLEU-like score is computed as the geometric mean of n-gram precisions (1 to 4-grams) with equal weights, without brevity penalty.
- N-gram precision is the fraction of n-grams in the translation that appear in the source.
- Missing segments are line numbers (1-indexed) present in source but not in translation.
- Added segments are line numbers (1-indexed) present in translation but not in source.

## Requirements

- Read all files in `source/` and `translation/` directories.
- Compute BLEU-like scores per file.
- Detect missing and added segments per file.
- Write the JSON report to `translation_qa.json` in the workspace root.

## Constraints

- Assume all files are UTF-8 encoded text files.
- The number of files is at least 20.
- Each file has multiple lines (segments).

## Evaluation

Your solution will be tested on synthetic data with known scores and flagged issues.

Good luck!