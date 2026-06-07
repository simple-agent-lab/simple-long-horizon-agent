#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE/documents"

python3 - <<'EOF'
import random
random.seed(42)
import os

# Sample sentences and phrases to generate documents
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Data science is an interdisciplinary field.",
    "Machine learning models can detect patterns.",
    "Natural language processing involves computational linguistics.",
    "Plagiarism detection is important in academia.",
    "Text similarity measures help find copied content.",
    "TF-IDF stands for term frequency-inverse document frequency.",
    "Jaccard similarity is a statistic used for comparing sets.",
    "Cosine similarity measures the cosine of the angle between vectors.",
    "Document clustering groups similar documents together.",
    "Deep learning has revolutionized many AI applications.",
    "The cat sat on the mat near the fireplace.",
    "Artificial intelligence is transforming industries.",
    "Students should avoid plagiarism in their assignments.",
    "Text preprocessing includes tokenization and normalization.",
    "Semantic similarity captures meaning beyond words.",
    "Vector space models represent documents numerically.",
    "Computational methods enable large scale text analysis.",
    "The dog barked loudly at the stranger.",
    "Research papers require proper citation and originality."
]

# Create 7 documents with some overlapping content to simulate plagiarism
num_docs = 7
os.makedirs(os.path.join("$WORKSPACE", "documents"), exist_ok=True)

for i in range(1, num_docs + 1):
    lines = []
    # Each doc has 15 sentences
    for _ in range(15):
        # Introduce some overlap for docs 3 and 4 to simulate plagiarism
        if i in (3, 4) and random.random() < 0.4:
            # Pick from a small subset to increase similarity
            lines.append(random.choice(sentences[:5]))
        else:
            lines.append(random.choice(sentences))
    # Slightly shuffle lines
    random.shuffle(lines)
    content = "\n".join(lines)
    filename = f"doc{i}.txt"
    with open(os.path.join("$WORKSPACE", "documents", filename), "w", encoding="utf-8") as f:
        f.write(content)
EOF
