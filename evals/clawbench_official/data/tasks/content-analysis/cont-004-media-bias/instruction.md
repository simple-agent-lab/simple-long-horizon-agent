# News Article Bias Analysis

## Task Description

You are given a collection of news articles stored as text files in the `workspace/articles/` directory. All articles discuss the same topic but come from different sources.

Your task is to analyze each article to assess potential media bias by performing the following analyses:

1. **Loaded Language Count:** Identify and count words or phrases that indicate emotionally charged or biased language (e.g., "radical", "unprecedented", "allegedly").

2. **Source Diversity:** Extract the source name from each article's metadata (the first line of each file is formatted as `Source: <source_name>`). Track how many unique sources are present.

3. **Sentiment Polarity:** Compute the sentiment polarity score of the article's main text (excluding the source line). Use a standard sentiment analysis approach.

4. **Subjectivity Indicators:** Compute a subjectivity score indicating how subjective or objective the article is.

5. **Bias Scoring:** Combine the above metrics into a bias score for each article. Higher loaded language counts, extreme sentiment polarity (very positive or very negative), and high subjectivity should increase the bias score.


## Input

- Multiple text files located in `workspace/articles/`.
- Each file starts with a source line: `Source: <source_name>`.
- The rest of the file is the article text.


## Output

- Write a JSON file `workspace/bias_report.json` containing:
  - For each article:
    - Filename
    - Source
    - Loaded language count
    - Sentiment polarity score
    - Subjectivity score
    - Computed bias score
  - Overall statistics:
    - Number of unique sources
    - Average bias score across all articles


## Requirements

- Implement text processing to identify loaded language from a predefined list.
- Use sentiment analysis and subjectivity scoring (e.g., via TextBlob or similar).
- Combine metrics into a bias score with a reasonable formula.
- Write the output JSON file in a human-readable format.


## Evaluation

Your solution will be evaluated on:
- Correctness of loaded language counting.
- Accurate extraction of source names.
- Correct sentiment and subjectivity computation.
- Proper bias score calculation.
- Correct output JSON structure and data.


## Notes

- The list of loaded language words to consider includes: `radical`, `unprecedented`, `allegedly`, `controversial`, `extreme`, `biased`, `sensational`, `propaganda`, `manipulative`, `exaggerated`.
- Sentiment polarity ranges from -1 (very negative) to 1 (very positive).
- Subjectivity ranges from 0 (objective) to 1 (subjective).
- Bias score can be computed as:

  ```
  bias_score = loaded_language_count * 0.5 + abs(sentiment_polarity) * 2 + subjectivity * 3
  ```

- Use UTF-8 encoding for reading and writing files.


Good luck!