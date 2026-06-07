# Task: Technology Selection Analysis

You are given a set of project constraints and must produce a structured technology comparison and recommendation.

## Input Files

- `workspace/constraints.json` — project constraints including budget, team size, timeline, and requirements

## Goal

Analyze the constraints and produce a technology selection analysis comparing at least 3 technology stack options.

## Requirements

1. Read the constraints file carefully.
2. Create `workspace/analysis.json` with the following structure:
   ```json
   {
     "options": [
       {
         "name": "Option name",
         "pros": ["advantage 1", "advantage 2"],
         "cons": ["disadvantage 1"],
         "cost_estimate": 50000,
         "recommendation_score": 8,
         "recommended": false
       }
     ]
   }
   ```
3. Include at least 3 technology stack options.
4. Each option must have:
   - `name` — descriptive name of the technology stack
   - `pros` — array of advantages (at least 2)
   - `cons` — array of disadvantages (at least 1)
   - `cost_estimate` — estimated cost as a number (must be within budget)
   - `recommendation_score` — integer from 1 to 10
   - `recommended` — boolean, exactly one option should be `true`
5. The recommended option should have the highest recommendation_score.

## Notes

- Consider the team's existing skills when evaluating options.
- Cost estimates should be realistic given the timeline and team size.
- At least one option should be budget-friendly and one should be premium.
- Scores should reflect how well the option fits ALL constraints.
