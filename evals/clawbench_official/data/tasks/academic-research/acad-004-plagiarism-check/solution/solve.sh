#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

# Required python packages: scikit-learn

python3 - <<EOF
import os
import json
import csv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read documents
docs_dir = Path(os.path.join("$WORKSPACE", "documents"))
doc_files = sorted([f for f in docs_dir.iterdir() if f.is_file() and f.suffix == ".txt"])

texts = []
filenames = []
for f in doc_files:
    with f.open("r", encoding="utf-8") as file:
        texts.append(file.read())
    filenames.append(f.name)

# Function to get character 3-grams set

def char_3grams(text):
    text = text.replace('\n', ' ')
    n = 3
    grams = set()
    for i in range(len(text) - n + 1):
        grams.add(text[i:i+n])
    return grams

# Precompute 3-gram sets
three_gram_sets = [char_3grams(t) for t in texts]

# Compute TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Prepare output data
pairs = []

for i in range(len(filenames)):
    for j in range(i+1, len(filenames)):
        doc1 = filenames[i]
        doc2 = filenames[j]

        # Jaccard 3-gram similarity
        set1 = three_gram_sets[i]
        set2 = three_gram_sets[j]
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_sim = intersection / union if union > 0 else 0.0

        # Cosine similarity on TF-IDF
        cos_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0,0]

        pairs.append({
            "doc1": doc1,
            "doc2": doc2,
            "jaccard_3gram": round(jaccard_sim, 6),
            "cosine_tfidf": round(cos_sim, 6)
        })

# Write similarity_matrix.csv
csv_path = Path("$WORKSPACE") / "similarity_matrix.csv"
with csv_path.open("w", encoding="utf-8", newline='') as csvfile:
    fieldnames = ["doc1", "doc2", "jaccard_3gram", "cosine_tfidf"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for pair in pairs:
        writer.writerow(pair)

# Write plagiarism_report.json
report_path = Path("$WORKSPACE") / "plagiarism_report.json"
flagged = [p for p in pairs if p["jaccard_3gram"] > 0.3 or p["cosine_tfidf"] > 0.3]

with report_path.open("w", encoding="utf-8") as f:
    json.dump(flagged, f, indent=2)
EOF
