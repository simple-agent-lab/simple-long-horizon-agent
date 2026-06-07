# Document Readability Metrics Audit

## Description

You are given a set of text documents located in the `workspace/documents/` directory. Your task is to analyze each document and compute several readability metrics:

- **Flesch-Kincaid Grade Level (FK Grade)**
- **Gunning Fog Index (Fog Index)**
- **Coleman-Liau Index (CLI)**

Additionally, compute the average sentence length (in words) and average word length (in characters) for each document.

You must write the results into a CSV file named `workspace/readability_report.csv` with the following columns:

| document | fk_grade | fog_index | cli | avg_sentence_length | avg_word_length |
|----------|----------|-----------|-----|---------------------|-----------------|

- `document`: The filename of the text document (e.g., `doc1.txt`).
- `fk_grade`: The Flesch-Kincaid Grade Level score (float rounded to 2 decimals).
- `fog_index`: The Gunning Fog Index score (float rounded to 2 decimals).
- `cli`: The Coleman-Liau Index score (float rounded to 2 decimals).
- `avg_sentence_length`: Average number of words per sentence (float rounded to 2 decimals).
- `avg_word_length`: Average number of characters per word (float rounded to 2 decimals).

## Requirements

- Read all `.txt` files in `workspace/documents/`.
- Compute the readability metrics as defined below.
- Write the results to `workspace/readability_report.csv`.
- Use standard English language rules for sentence and word tokenization.

## Readability Metrics Definitions

- **Flesch-Kincaid Grade Level:**

  \[ FK = 0.39 \times (\frac{total\ words}{total\ sentences}) + 11.8 \times (\frac{total\ syllables}{total\ words}) - 15.59 \]

- **Gunning Fog Index:**

  \[ Fog = 0.4 \times \left( \frac{total\ words}{total\ sentences} + 100 \times \frac{complex\ words}{total\ words} \right) \]

  where complex words are words with 3 or more syllables, excluding proper nouns, familiar jargon, and compound words.

- **Coleman-Liau Index:**

  \[ CLI = 0.0588L - 0.296S - 15.8 \]

  where:
  - \(L\) = average number of letters per 100 words
  - \(S\) = average number of sentences per 100 words

## Notes

- For syllable counting, use a reasonable heuristic (e.g., count vowel groups).
- Sentences can be split on `.`, `!`, `?` followed by whitespace.
- Words are sequences of alphabetic characters.
- Round all numeric outputs to 2 decimal places.

## Output Example

```csv
document,fk_grade,fog_index,cli,avg_sentence_length,avg_word_length
doc1.txt,8.23,10.45,7.89,15.67,4.32
doc2.txt,12.01,14.56,11.23,20.12,4.75
```

Ensure your solution is efficient and handles all documents in the directory.
