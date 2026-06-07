#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

input_file="$WORKSPACE/merger_agreement.txt"
output_file="$WORKSPACE/due_diligence.json"

# Define section headers and corresponding JSON keys
declare -A section_map=(
  ["CONDITIONS PRECEDENT"]="conditions_precedent"
  ["REPRESENTATIONS AND WARRANTIES"]="representations_warranties"
  ["INDEMNIFICATION TERMS"]="indemnification_terms"
  ["CLOSING TIMELINE"]="closing_timeline"
  ["MATERIAL ADVERSE CHANGE CLAUSE"]="material_adverse_change_clause"
)

# Initialize associative array to hold extracted text
for key in "${section_map[@]}"; do
  declare "$key"=""
done

current_section=""

while IFS= read -r line || [[ -n "$line" ]]; do
  # Trim trailing and leading whitespace
  trimmed_line="$(echo "$line" | sed 's/^\s*//;s/\s*$//')"

  # Check if line matches a known section header
  if [[ -n "$trimmed_line" && ${section_map[$trimmed_line]+_} ]]; then
    current_section="${section_map[$trimmed_line]}"
    continue
  fi

  # Check if line is a header underline (dashes), skip
  if [[ "$line" =~ ^-+$ ]]; then
    continue
  fi

  # If current_section is set, append line
  if [[ -n "$current_section" ]]; then
    # Append line with newline
    eval "$current_section+=\"$line\n\""
  fi

done < "$input_file"

# Prepare JSON output
# Escape backslashes and double quotes properly
json_escape() {
  python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"
}

# Compose JSON
jq -n \
  --arg cp "$(printf "%b" "$conditions_precedent" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
  --arg rw "$(printf "%b" "$representations_warranties" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
  --arg it "$(printf "%b" "$indemnification_terms" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
  --arg ct "$(printf "%b" "$closing_timeline" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
  --arg mac "$(printf "%b" "$material_adverse_change_clause" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
  '{
    conditions_precedent: $cp,
    representations_warranties: $rw,
    indemnification_terms: $it,
    closing_timeline: $ct,
    material_adverse_change_clause: $mac
  }' > "$output_file"
