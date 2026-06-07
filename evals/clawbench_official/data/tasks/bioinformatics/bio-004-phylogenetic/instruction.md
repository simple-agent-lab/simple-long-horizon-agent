# Phylogenetic Tree Construction from Sequences

## Description

You are given a FASTA file named `species_sequences.fasta` located in the workspace directory. This file contains DNA sequences of multiple species.

Your tasks are:

1. **Read** the sequences from `workspace/species_sequences.fasta`.
2. **Compute** the pairwise distance matrix between all sequences using the normalized Hamming distance. The normalized Hamming distance between two sequences is the number of differing nucleotides divided by the sequence length.
3. **Construct** a phylogenetic tree using the Neighbor-Joining (NJ) algorithm based on the computed distance matrix.
4. **Write** the resulting distance matrix to `workspace/distance_matrix.csv` in CSV format, where rows and columns are labeled by species names.
5. **Write** the resulting phylogenetic tree in Newick format to `workspace/tree.newick`.


## Input

- `workspace/species_sequences.fasta`: A FASTA file containing 20+ DNA sequences of equal length.


## Output

- `workspace/distance_matrix.csv`: CSV file with species names as headers and rows, containing normalized Hamming distances.
- `workspace/tree.newick`: A text file containing the phylogenetic tree in Newick format.


## Notes

- You may assume all sequences are aligned and of equal length.
- Use the Neighbor-Joining algorithm for tree construction.
- The output files must be created exactly as specified.


## Example

If the input FASTA contains sequences for species A, B, and C, the `distance_matrix.csv` should look like:

```
,SpeciesA,SpeciesB,SpeciesC
SpeciesA,0.0,0.1,0.2
SpeciesB,0.1,0.0,0.15
SpeciesC,0.2,0.15,0.0
```

and `tree.newick` should contain a valid Newick tree string representing the relationships.


## Evaluation

Your submission will be evaluated on:

- Correctness of the distance matrix.
- Correctness and validity of the Newick tree.
- Proper file reading and writing.
- Handling of the provided data.


Good luck!