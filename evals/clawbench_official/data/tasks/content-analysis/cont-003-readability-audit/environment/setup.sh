#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE/documents"

python3 - <<'EOF'
import random
import os
random.seed(42)

# Sample sentences with varying complexity
simple_sentences = [
    "The cat sat on the mat.",
    "She likes to read books.",
    "He went to the store.",
    "It is a sunny day.",
    "They play football every weekend.",
    "Birds are flying in the sky.",
    "We enjoy watching movies.",
    "The dog barked loudly.",
    "Children are playing outside.",
    "I love eating ice cream."
]

complex_sentences = [
    "The internationalization of the organization requires comprehensive planning.",
    "Photosynthesis is a process used by plants to convert light energy.",
    "The juxtaposition of the two theories highlights their differences.",
    "Understanding the ramifications of the policy is essential.",
    "The multifaceted approach improved the overall efficiency.",
    "Technological advancements have revolutionized communication.",
    "The hypothesis was tested through rigorous experimentation.",
    "Environmental sustainability is a critical global issue.",
    "The architecture of the building is both modern and functional.",
    "Philosophical debates often involve abstract reasoning."
]

# Generate 25 documents with mixed sentences
num_docs = 25
os.makedirs("$WORKSPACE/documents", exist_ok=True)

for i in range(1, num_docs + 1):
    filename = f"doc{i}.txt"
    lines = []
    # Each document will have 20-30 sentences
    num_sentences = random.randint(20, 30)
    for _ in range(num_sentences):
        # Randomly pick simple or complex sentence
        if random.random() < 0.6:
            sentence = random.choice(simple_sentences)
        else:
            sentence = random.choice(complex_sentences)
        lines.append(sentence)
    text = " ".join(lines)
    with open(os.path.join("$WORKSPACE/documents", filename), "w", encoding="utf-8") as f:
        f.write(text + "\n")
EOF
