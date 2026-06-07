# DNA Sequence GC Content and ORF Analysis

## Description

You are given a file `workspace/sequences.fasta` containing multiple DNA sequences in FASTA format. Your task is to:

1. **Compute GC content** for each sequence. The GC content is the percentage of nucleotides in the sequence that are either G or C.

2. **Identify Open Reading Frames (ORFs)** in each sequence. An ORF is defined as a DNA subsequence that:
   - Starts with the start codon `ATG`.
   - Ends with one of the stop codons: `TAA`, `TAG`, or `TGA`.
   - Has a length of at least 100 base pairs (including start and stop codons).

3. **Output two files:**
   - `workspace/sequence_analysis.csv`: A CSV file with columns:
     - `sequence_id`: The identifier of the sequence from the FASTA header (without the `>`).
     - `length`: Length of the sequence in base pairs.
     - `gc_content`: GC content percentage rounded to two decimal places.
     - `orf_count`: Number of ORFs found in the sequence.

   - `workspace/orf_summary.json`: A JSON file summarizing all ORFs found across all sequences. The JSON structure should be a dictionary where keys are sequence IDs and values are lists of ORFs. Each ORF should be represented as a dictionary with:
     - `start`: 1-based start position of the ORF in the sequence.
     - `end`: 1-based end position of the ORF in the sequence.
     - `length`: Length of the ORF in base pairs.

## Input

- `workspace/sequences.fasta`: FASTA format file with 20+ DNA sequences.

## Output

- `workspace/sequence_analysis.csv`
- `workspace/orf_summary.json`

## Notes

- Positions are 1-based (the first nucleotide of the sequence is position 1).
- ORFs do not overlap in the output; report all valid ORFs found in the sequence reading frame.
- You only need to analyze the given sequences as-is (no reverse complement or other frames).
- Use uppercase letters for DNA sequences.

## Example

If a sequence with ID `seq1` is 300 bp long with 45% GC content and 2 ORFs found, the CSV row would be:

```
seq1,300,45.00,2
```

The JSON entry for `seq1` might look like:

```json
"seq1": [
  {"start": 10, "end": 150, "length": 141},
  {"start": 200, "end": 320, "length": 121}
]
```

## Requirements

- Your solution must read the input file from `workspace/sequences.fasta`.
- Write the two output files exactly as specified.
- Ensure the output files are valid and well-formatted.
- Use only the standard Python 3 libraries or bash utilities.

Good luck!