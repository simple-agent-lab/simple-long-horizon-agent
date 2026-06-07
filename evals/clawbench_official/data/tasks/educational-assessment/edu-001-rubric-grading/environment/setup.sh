#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE/essays"

# Generate rubric.csv
cat > "$WORKSPACE/rubric.csv" << EOF
criterion,max_points,keywords
Clarity,10,clear;concise;understandable;simple;direct
Argument Strength,15,evidence;reason;logic;support;claim
Grammar,10,grammar;syntax;punctuation;sentence;structure
EOF

# Generate a simple English word list for spelling check
# Using a small subset of common English words
cat > "$WORKSPACE/english_words.txt" << EOF
this
is
a
sample
text
with
some
common
english
words
and
simple
sentences
structure
logic
support
claim
evidence
reason
clear
concise
understandable
direct
grammar
syntax
punctuation
sentence
structure
for
students
writing
about
education
EOF

# Generate 25 student essays with some variability
python3 - << EOF
import random
import os
random.seed(42)

keywords = {
    "Clarity": ["clear", "concise", "understandable", "simple", "direct"],
    "Argument Strength": ["evidence", "reason", "logic", "support", "claim"],
    "Grammar": ["grammar", "syntax", "punctuation", "sentence", "structure"]
}

common_words = ["this", "is", "a", "sample", "text", "with", "some", "common", "english", "words", "and", "simple", "sentences", "for", "students", "writing", "about", "education"]

# Helper to generate sentences

def generate_sentence(keywords_for_criterion, include_keywords_prob=0.6):
    sentence_words = []
    # Decide to include keywords or not
    if random.random() < include_keywords_prob:
        # Include 1 or 2 keywords randomly
        num_keywords = random.choice([1, 2])
        chosen_keywords = random.sample(keywords_for_criterion, num_keywords)
        sentence_words.extend(chosen_keywords)
    # Add filler words
    num_filler = random.randint(5, 12)
    filler_words = random.choices(common_words, k=num_filler)
    sentence_words.extend(filler_words)
    random.shuffle(sentence_words)
    sentence = " ".join(sentence_words).capitalize() + "."
    return sentence

os.makedirs(os.path.join("$WORKSPACE", "essays"), exist_ok=True)

for i in range(1, 26):
    student_id = f"student{i}"
    lines = []
    # For each criterion, generate 3-5 sentences
    for criterion, kws in keywords.items():
        num_sentences = random.randint(3, 5)
        for _ in range(num_sentences):
            # Vary keyword inclusion probability per student to simulate quality
            prob = random.uniform(0.4, 0.9)
            sentence = generate_sentence(kws, include_keywords_prob=prob)
            lines.append(sentence)
    # Add some random filler sentences without keywords
    for _ in range(2):
        filler_len = random.randint(6, 12)
        filler_words = random.choices(common_words, k=filler_len)
        sentence = " ".join(filler_words).capitalize() + "."
        lines.append(sentence)
    # Shuffle all sentences
    random.shuffle(lines)
    essay_text = "\n".join(lines)
    with open(os.path.join("$WORKSPACE", "essays", f"{student_id}.txt"), "w") as f:
        f.write(essay_text)
EOF
