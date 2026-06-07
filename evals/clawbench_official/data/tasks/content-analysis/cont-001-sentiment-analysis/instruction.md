# Product Review Sentiment Classification

## Description

You are given a CSV file `workspace/reviews.csv` containing product reviews with the following columns:

- `review_id`: Unique identifier for each review
- `product`: The product name
- `rating`: Numeric rating (1 to 5)
- `text`: The review text

Your task is to classify the sentiment of each review based on keyword scoring using predefined positive and negative word lists. Then, compute an aggregate sentiment score per product.

Finally, write the results to `workspace/sentiment_dashboard.json`.


## Details

1. **Sentiment Classification:**
   - Use the following keyword lists:
     - Positive words: `good`, `great`, `excellent`, `amazing`, `love`, `wonderful`, `best`, `fantastic`, `perfect`, `nice`
     - Negative words: `bad`, `terrible`, `awful`, `worst`, `hate`, `poor`, `disappointing`, `boring`, `slow`, `ugly`
   - For each review, count the occurrences of positive and negative words (case-insensitive).
   - Calculate the sentiment score for the review as: `positive_count - negative_count`.
   - Classify the review sentiment as:
     - `positive` if score > 0
     - `negative` if score < 0
     - `neutral` if score == 0

2. **Aggregate per Product:**
   - For each product, compute:
     - `total_reviews`: total number of reviews
     - `positive_reviews`: number of positive reviews
     - `negative_reviews`: number of negative reviews
     - `neutral_reviews`: number of neutral reviews
     - `average_sentiment_score`: average of sentiment scores across all its reviews (float)

3. **Output:**
   - Write a JSON file `workspace/sentiment_dashboard.json` with a dictionary keyed by product names.
   - Each product maps to an object with the above aggregate statistics.


## Requirements

- Read `reviews.csv` from the workspace.
- Perform sentiment classification as described.
- Write the JSON output to `sentiment_dashboard.json` in the workspace.
- Use only the provided keyword lists for sentiment scoring.


## Example

If `reviews.csv` contains:

| review_id | product | rating | text                      |
|-----------|---------|--------|---------------------------|
| 1         | PhoneX  | 5      | "I love this phone, it is perfect and amazing!" |
| 2         | PhoneX  | 2      | "The battery life is bad and the screen is ugly." |

Then `sentiment_dashboard.json` would include an entry for `PhoneX` with counts and average sentiment score accordingly.


## Files

- Input: `workspace/reviews.csv`
- Output: `workspace/sentiment_dashboard.json`
