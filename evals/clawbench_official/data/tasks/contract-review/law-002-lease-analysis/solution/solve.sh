#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

input_file="$WORKSPACE/lease_agreement.txt"
output_file="$WORKSPACE/lease_analysis.json"

# Use python to parse and extract required fields
python3 - <<EOF
import re
import json

with open("$input_file", "r", encoding="utf-8") as f:
    text = f.read()

# Helper function to extract a line or paragraph containing a keyword

def extract_clause(keyword_list, text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for kw in keyword_list:
            if kw.lower() in line.lower():
                # Return line and next line if relevant
                snippet = line.strip()
                # Try to append next line if it looks like continuation
                if i + 1 < len(lines) and len(lines[i+1].strip()) > 0 and not any(k.lower() in lines[i+1].lower() for k in keyword_list):
                    snippet += " " + lines[i+1].strip()
                return snippet
    return "None"

# Extract base_rent
base_rent = extract_clause(["base rent", "base_rent"], text)

# Extract cam_charges
cam_charges = extract_clause(["cam charges", "common area maintenance", "cam_charge"], text)

# Extract escalation_rate
escalation_rate = extract_clause(["escalation rate", "annual increase", "rent escalation"], text)

# Extract lease_term
lease_term = extract_clause(["lease term", "term of the lease", "lease duration"], text)

# Extract renewal_options
renewal_options = extract_clause(["renewal option", "renewal period", "renewal term"], text)

# Extract termination_clauses
termination_clauses = extract_clause(["termination clause", "termination right", "terminate the lease", "termination"], text)

# Identify risk clauses
risk_keywords = ["penalty", "early termination", "responsible", "increase cam", "fee", "liability", "breach", "default"]
risk_clauses_flagged = []

lines = text.splitlines()
for line in lines:
    low = line.lower()
    if any(k in low for k in risk_keywords):
        clause = line.strip()
        if clause and clause not in risk_clauses_flagged:
            risk_clauses_flagged.append(clause)

# If no renewal options found, set to None
if renewal_options == "None":
    renewal_options = "None"

# If no termination clauses found, set to None
if termination_clauses == "None":
    termination_clauses = "None"

output = {
    "base_rent": base_rent,
    "cam_charges": cam_charges,
    "escalation_rate": escalation_rate,
    "lease_term": lease_term,
    "renewal_options": renewal_options,
    "termination_clauses": termination_clauses,
    "risk_clauses_flagged": risk_clauses_flagged
}

with open("$output_file", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
EOF
