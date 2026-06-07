# Pairwise Protein Sequence Alignment

## Description

You are given a FASTA file `proteins.fasta` containing multiple protein sequences in the workspace directory. Your task is to implement the Needleman-Wunsch global alignment algorithm to compute pairwise alignments for all unique pairs of sequences.

Use the following scoring scheme:

- Match: +2
- Mismatch: -1
- Gap penalty: -2

For each pair of sequences, compute:

- **Identity**: The percentage of aligned positions with identical amino acids.
- **Similarity**: The percentage of aligned positions that are either identical or similar (for this task, consider only exact matches as similar; mismatches are not similar).

Write the results as a JSON array to `alignment_results.json` in the workspace directory. Each element in the array should be a JSON object with the following keys:

- `seq1_id`: The identifier of the first sequence.
- `seq2_id`: The identifier of the second sequence.
- `identity`: Identity percentage (float, rounded to 2 decimal places).
- `similarity`: Similarity percentage (float, rounded to 2 decimal places).

## Input

- `workspace/proteins.fasta`: A FASTA file with at least 20 protein sequences.

## Output

- `workspace/alignment_results.json`: JSON array with pairwise alignment results.

## Requirements

- Implement Needleman-Wunsch global alignment with the specified scoring.
- Compute identity and similarity as described.
- Output must be valid JSON.

## Example

If `proteins.fasta` contains sequences with IDs `seq1` and `seq2`, one element of the output might be:

```json
{
  "seq1_id": "seq1",
  "seq2_id": "seq2",
  "identity": 75.00,
  "similarity": 75.00
}
```

## Notes

- Use zero-based indexing for sequences internally.
- Only unique pairs (seq1 < seq2) need to be computed.
- Similarity here equals identity since only exact matches count as similar.
- Round percentages to two decimal places.
- The sequences are protein sequences (amino acids).

Good luck!