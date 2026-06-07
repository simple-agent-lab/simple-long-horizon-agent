# Genomic Variant Annotation and Classification

## Description

You are provided with two data files in the workspace:

- `variants.tsv`: Contains genomic variants with columns:
  - `chrom`: Chromosome (e.g., chr1, chr2, ...)
  - `pos`: Position on the chromosome (integer)
  - `ref`: Reference allele (string)
  - `alt`: Alternate allele (string)
  - `qual`: Quality score (float)

- `gene_regions.csv`: Contains gene region annotations with columns:
  - `gene`: Gene name (string)
  - `chrom`: Chromosome (e.g., chr1, chr2, ...)
  - `start`: Start position of the gene region (integer)
  - `end`: End position of the gene region (integer)
  - `function`: Functional region type (e.g., exon, intron, promoter)


## Task

1. **Read** the `variants.tsv` and `gene_regions.csv` files from the workspace.
2. **Annotate** each variant with the gene and region it falls into by matching the variant's chromosome and position to the gene region intervals.
   - If a variant does not fall into any gene region, annotate gene and function as `null`.
3. **Classify** the pathogenicity of each variant based on its quality score (`qual`):
   - `qual >= 90`: `Pathogenic`
   - `70 <= qual < 90`: `Likely Pathogenic`
   - `50 <= qual < 70`: `Uncertain`
   - `qual < 50`: `Benign`
4. **Write** the annotated variants to a JSON file named `annotated_variants.json` in the workspace.


## Output Format

The output JSON file should be a list of variant objects with the following fields:

- `chrom` (string)
- `pos` (integer)
- `ref` (string)
- `alt` (string)
- `qual` (float)
- `gene` (string or null)
- `function` (string or null)
- `pathogenicity` (string)


## Example

```json
[
  {
    "chrom": "chr1",
    "pos": 12345,
    "ref": "A",
    "alt": "G",
    "qual": 95.2,
    "gene": "BRCA1",
    "function": "exon",
    "pathogenicity": "Pathogenic"
  },
  {
    "chrom": "chr2",
    "pos": 54321,
    "ref": "T",
    "alt": "C",
    "qual": 45.0,
    "gene": null,
    "function": null,
    "pathogenicity": "Benign"
  }
]
```


## Notes

- Positions are 1-based and inclusive.
- Variants may overlap multiple gene regions; annotate with the first matching gene region found in the gene_regions.csv file order.
- Use exact chromosome string matching (e.g., "chr1" matches "chr1" only).
- The workspace directory is provided as the first argument to your scripts or defaults to `/workspace`.


Good luck!