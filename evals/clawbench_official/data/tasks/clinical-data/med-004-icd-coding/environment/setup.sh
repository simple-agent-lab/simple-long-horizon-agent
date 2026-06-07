#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE/clinical_notes"

# Generate icd_mapping.csv with realistic keywords and ICD-10 codes
python3 - <<'EOF'
import csv
import random
random.seed(42)

icd_data = [
    ("hypertension", "I10", "Essential (primary) hypertension"),
    ("diabetes", "E11.9", "Type 2 diabetes mellitus without complications"),
    ("asthma", "J45.909", "Unspecified asthma, uncomplicated"),
    ("chest pain", "R07.9", "Chest pain, unspecified"),
    ("headache", "R51", "Headache"),
    ("fever", "R50.9", "Fever, unspecified"),
    ("cough", "R05", "Cough"),
    ("shortness of breath", "R06.02", "Shortness of breath"),
    ("obesity", "E66.9", "Obesity, unspecified"),
    ("depression", "F32.9", "Major depressive disorder, single episode, unspecified"),
    ("anxiety", "F41.9", "Anxiety disorder, unspecified"),
    ("back pain", "M54.5", "Low back pain"),
    ("allergy", "T78.40XA", "Allergy, unspecified, initial encounter"),
    ("pneumonia", "J18.9", "Pneumonia, unspecified organism"),
    ("migraine", "G43.909", "Migraine, unspecified, not intractable, without status migrainosus"),
    ("arthritis", "M19.90", "Osteoarthritis, unspecified site"),
    ("anemia", "D64.9", "Anemia, unspecified"),
    ("infection", "B99", "Other and unspecified infectious diseases"),
    ("fracture", "S52.501A", "Unspecified fracture of lower end of right radius, initial encounter"),
    ("insomnia", "G47.00", "Insomnia, unspecified"),
]

with open(f"{"$WORKSPACE"}/icd_mapping.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["keyword", "icd_code", "description"])
    for row in icd_data:
        writer.writerow(row)
EOF

# Generate 25 clinical note text files with some keywords randomly inserted
python3 - <<'EOF'
import random
random.seed(42)
import os

keywords = [
    "hypertension", "diabetes", "asthma", "chest pain", "headache", "fever", "cough", "shortness of breath",
    "obesity", "depression", "anxiety", "back pain", "allergy", "pneumonia", "migraine", "arthritis", "anemia",
    "infection", "fracture", "insomnia"
]

templates = [
    "The patient presents with {}.",
    "History of {} noted.",
    "Symptoms include {} and {}.",
    "No signs of {} or {}.",
    "Patient denies {} but reports {}.",
    "Clinical examination reveals {}.",
    "Past medical history significant for {}.",
    "{} has been managed with medication.",
    "{} was observed during the visit.",
    "{} and {} are concerns for this patient."
]

notes_dir = f"{"$WORKSPACE"}/clinical_notes"

for i in range(1, 26):
    num_keywords = random.randint(0, 4)
    chosen_keywords = random.sample(keywords, num_keywords) if num_keywords > 0 else []
    lines = []
    for _ in range(5):
        template = random.choice(templates)
        # Fill template with 1 or 2 keywords as needed
        if template.count("{}") == 2:
            if len(chosen_keywords) >= 2:
                kw1, kw2 = chosen_keywords[:2]
            elif len(chosen_keywords) == 1:
                kw1, kw2 = chosen_keywords[0], "no symptoms"
            else:
                kw1, kw2 = "no symptoms", "no symptoms"
            line = template.format(kw1, kw2)
        else:
            if len(chosen_keywords) >= 1:
                line = template.format(chosen_keywords[0])
            else:
                line = template.format("no relevant findings")
        lines.append(line)
    content = "\n".join(lines)
    with open(os.path.join(notes_dir, f"note_{i}.txt"), "w") as f:
        f.write(content + "\n")
EOF
