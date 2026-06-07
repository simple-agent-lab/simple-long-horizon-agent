#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 - <<EOF
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

corpus_dir = os.path.join("$WORKSPACE", "corpus")
documents = []
doc_names = []

for fname in sorted(os.listdir(corpus_dir)):
    if fname.endswith(".txt"):
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(fname)

# Compute TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# KMeans clustering
k = 5
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)
labels = model.labels_

# Extract top words per cluster
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

topic_words = []
for i in range(k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    topic_words.append((i, " ".join(top_terms)))

# Write topic_words.csv
with open(os.path.join("$WORKSPACE", "topic_words.csv"), "w", encoding="utf-8") as f:
    for topic_id, words_str in topic_words:
        f.write(f"{topic_id},{words_str}\n")

# Write doc_topics.json
doc_topic_map = {doc_names[i]: int(labels[i]) for i in range(len(doc_names))}
with open(os.path.join("$WORKSPACE", "doc_topics.json"), "w", encoding="utf-8") as f:
    json.dump(doc_topic_map, f, indent=2)
EOF
