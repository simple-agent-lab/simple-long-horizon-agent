# Document Corpus Topic Extraction

## Description

You are provided with a corpus of 20+ text documents located in the `workspace/corpus/` directory. Your task is to analyze this document collection to discover underlying topics.

Specifically, you must:

1. Read all text files from `workspace/corpus/`.
2. Compute the TF-IDF matrix for the entire corpus.
3. Apply K-Means clustering with `k=5` to the TF-IDF vectors to identify 5 distinct topics.
4. For each topic, identify the top 10 words that characterize it.
5. Assign each document to one of the 5 topics.

## Outputs

Your program must write two files:

- `workspace/topic_words.csv`: A CSV file with two columns:
  - `topic`: integer topic ID (0 to 4)
  - `top_words`: a space-separated string of the top 10 words for that topic

- `workspace/doc_topics.json`: A JSON object mapping each document filename (e.g., `doc_01.txt`) to its assigned topic ID (0 to 4).

## Requirements

- Use standard TF-IDF vectorization (e.g., scikit-learn's `TfidfVectorizer` with default parameters).
- Use K-Means clustering with `k=5`.
- Extract top words per topic by selecting words with highest centroid values.
- The documents are plain text files with English content.

## Example

If `doc_01.txt` is assigned to topic 2, the JSON will contain:

```json
{"doc_01.txt": 2, ...}
```

If topic 0's top words are "data machine learning model algorithm ...", the CSV row will be:

```
0,data machine learning model algorithm ...
```

## Evaluation

Your submission will be verified by:

- Checking that the two output files exist and are correctly formatted.
- Validating that the topic assignments correspond to the clustering.
- Ensuring the top words per topic are correctly extracted.

Good luck!