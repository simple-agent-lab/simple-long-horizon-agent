#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

input_file="$WORKSPACE/license_agreement.txt"
csv_file="$WORKSPACE/rights_matrix.csv"
json_file="$WORKSPACE/license_summary.json"

# Read the license text
text=$(cat "$input_file" | tr '\n' ' ' | tr '[:upper:]' '[:lower:]')

# Helper function to check if any of the phrases appear in the text
function contains_any() {
  local text="$1"
  shift
  for phrase in "$@"; do
    if echo "$text" | grep -q "$phrase"; then
      return 0
    fi
  done
  return 1
}

# Permissions
# Define phrases indicating permission granted or denied
# We'll check for presence of positive phrases, and absence of negative phrases

# commercial_use
if contains_any "$text" "commercial purposes" "commercial use" "use it for commercial" && ! contains_any "$text" "commercial use is prohibited" "commercial use is strictly prohibited"; then
  commercial_use=true
else
  commercial_use=false
fi

# modification
if contains_any "$text" "modification of the software" "modify the software" "modifications are allowed" "modifying the software" && ! contains_any "$text" "may not modify" "not allowed to modify" "you may not modify"; then
  modification=true
else
  modification=false
fi

# distribution
if contains_any "$text" "distribution of the software" "distribute the software" "redistribution is allowed" "redistribute" && ! contains_any "$text" "distribution is prohibited" "redistribution is prohibited"; then
  distribution=true
else
  distribution=false
fi

# sublicense
if contains_any "$text" "sublicensing rights are granted" "sublicense rights are granted" "grant sublicenses" "sublicense" && ! contains_any "$text" "sublicensing rights are not granted" "no sublicense" "sublicense rights are not granted"; then
  sublicense=true
else
  sublicense=false
fi

# patent_use
if contains_any "$text" "patent rights are granted" "use any patents" "patent use" && ! contains_any "$text" "patent use is not included" "no patent use"; then
  patent_use=true
else
  patent_use=false
fi

# Restrictions

# attribution
if contains_any "$text" "attribution to the original author" "must be given" "attribution is required" "credit the author" && ! contains_any "$text" "attribution is optional" "no attribution required"; then
  attribution=true
else
  attribution=false
fi

# share_alike
if contains_any "$text" "share alike" "licensed under the same terms" "derivative works must be licensed" && ! contains_any "$text" "no share alike" "share alike requirement does not apply"; then
  share_alike=true
else
  share_alike=false
fi

# no_trademark
if contains_any "$text" "use of the licensor's trademarks is prohibited" "trademark use requires prior written permission" "no trademark" "no use of trademarks"; then
  no_trademark=true
else
  no_trademark=false
fi

# no_liability
if contains_any "$text" "disclaims all liability" "no warranty" "liability is limited" "no liability"; then
  no_liability=true
else
  no_liability=false
fi

# Write CSV
{
  echo "Category,Permission/Restriction,Granted"
  echo "Permission,commercial_use,$( [ "$commercial_use" = true ] && echo Yes || echo No )"
  echo "Permission,modification,$( [ "$modification" = true ] && echo Yes || echo No )"
  echo "Permission,distribution,$( [ "$distribution" = true ] && echo Yes || echo No )"
  echo "Permission,sublicense,$( [ "$sublicense" = true ] && echo Yes || echo No )"
  echo "Permission,patent_use,$( [ "$patent_use" = true ] && echo Yes || echo No )"
  echo "Restriction,attribution,$( [ "$attribution" = true ] && echo Yes || echo No )"
  echo "Restriction,share_alike,$( [ "$share_alike" = true ] && echo Yes || echo No )"
  echo "Restriction,no_trademark,$( [ "$no_trademark" = true ] && echo Yes || echo No )"
  echo "Restriction,no_liability,$( [ "$no_liability" = true ] && echo Yes || echo No )"
} > "$csv_file"

# Write JSON
jq -n \
  --argjson commercial_use "$commercial_use" \
  --argjson modification "$modification" \
  --argjson distribution "$distribution" \
  --argjson sublicense "$sublicense" \
  --argjson patent_use "$patent_use" \
  --argjson attribution "$attribution" \
  --argjson share_alike "$share_alike" \
  --argjson no_trademark "$no_trademark" \
  --argjson no_liability "$no_liability" \
  '{commercial_use: $commercial_use, modification: $modification, distribution: $distribution, sublicense: $sublicense, patent_use: $patent_use, attribution: $attribution, share_alike: $share_alike, no_trademark: $no_trademark, no_liability: $no_liability}' > "$json_file"
