#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE/corpus"

python3 - <<'EOF'
import random
import os
random.seed(42)

# Sample vocabulary and topics
topics = {
    0: ["data", "analysis", "statistics", "model", "regression", "variable", "distribution", "sample", "mean", "variance"],
    1: ["machine", "learning", "algorithm", "neural", "network", "training", "deep", "layer", "feature", "classification"],
    2: ["natural", "language", "processing", "text", "token", "syntax", "semantic", "corpus", "parsing", "entity"],
    3: ["computer", "hardware", "processor", "memory", "storage", "input", "output", "device", "circuit", "digital"],
    4: ["biology", "cell", "genome", "protein", "dna", "organism", "evolution", "species", "gene", "enzyme"]
}

# Generate 25 documents, each mostly about one topic with some noise
num_docs = 25
words_per_doc = 100
noise_words = ["the", "and", "is", "in", "of", "to", "with", "for", "on", "by"]

os.makedirs("$WORKSPACE/corpus", exist_ok=True)

for i in range(1, num_docs + 1):
    topic_id = i % 5
    topic_words = topics[topic_id]
    doc_words = []
    for _ in range(words_per_doc):
        if random.random() < 0.7:
            # Pick a topic word
            w = random.choice(topic_words)
        else:
            # Pick noise word
            w = random.choice(noise_words)
        doc_words.append(w)
    content = " ".join(doc_words)
    filename = f"doc_{i:02d}.txt"
    with open(os.path.join("$WORKSPACE/corpus", filename), "w") as f:
        f.write(content + "\n")
EOF
