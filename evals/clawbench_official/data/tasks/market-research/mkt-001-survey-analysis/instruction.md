# Analyze Customer Survey and Compute NPS Score

You are provided with a CSV file named `survey_responses.csv` located in the workspace directory. This file contains customer survey responses with the following columns:

- `respondent_id`: Unique identifier for each respondent
- `age_group`: Age group of the respondent (e.g., "18-25", "26-35", etc.)
- `gender`: Gender of the respondent
- `satisfaction_1_10`: Satisfaction rating on a scale from 1 to 10
- `recommend_0_10`: Likelihood to recommend the product/service on a scale from 0 to 10
- `comment`: Optional free-text comment from the respondent

Your task is to analyze this survey data and compute the Net Promoter Score (NPS). The NPS is calculated as follows:

- **Promoters**: Respondents who gave a `recommend_0_10` score of 9 or 10
- **Detractors**: Respondents who gave a `recommend_0_10` score from 0 to 6
- **Passives**: Respondents who gave a `recommend_0_10` score of 7 or 8

You must:

1. Calculate the overall NPS score:
   
   \[ \text{NPS} = \%\text{Promoters} - \%\text{Detractors} \]

2. Calculate the NPS score segmented by `age_group`.
3. Calculate the total number of responses.
4. Calculate the percentage of promoters, detractors, and passives overall.

Finally, write the results to a JSON file named `nps_report.json` in the workspace directory with the following structure:

```json
{
  "overall_nps": <float>,
  "segment_nps": {
    "18-25": <float>,
    "26-35": <float>,
    ...
  },
  "total_responses": <int>,
  "promoter_pct": <float>,
  "detractor_pct": <float>,
  "passive_pct": <float>
}
```

**Notes:**
- Percentages and NPS values should be represented as floats rounded to two decimal places.
- The `segment_nps` object should include all age groups present in the data.
- Use the data in `survey_responses.csv` only.

Your solution should read the input file, perform the computations, and write the output file exactly as specified.

---

Good luck!