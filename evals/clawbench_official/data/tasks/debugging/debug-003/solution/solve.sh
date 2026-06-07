#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 << PYEOF
with open("$WORKSPACE/processor.py", "r") as f:
    code = f.read()

# Fix 1: get_user_name — guard against None and missing keys
code = code.replace(
    '''def get_user_name(user):
    """Extract the full name from a user dict."""
    return user["first_name"] + " " + user["last_name"]''',
    '''def get_user_name(user):
    """Extract the full name from a user dict."""
    if user is None:
        return "Unknown"
    first = user.get("first_name", "Unknown")
    last = user.get("last_name", "Unknown")
    return first + " " + last'''
)

# Fix 2: get_config_value — use .get() with default
code = code.replace(
    '''def get_config_value(config, key):
    """Get a configuration value by key."""
    return config[key]''',
    '''def get_config_value(config, key):
    """Get a configuration value by key."""
    return config.get(key, "default")'''
)

# Fix 3: get_first_element — guard against empty list
code = code.replace(
    '''def get_first_element(items):
    """Get the first element from a list."""
    return items[0]''',
    '''def get_first_element(items):
    """Get the first element from a list."""
    if not items:
        return None
    return items[0]'''
)

with open("$WORKSPACE/fixed.py", "w") as f:
    f.write(code)
PYEOF

echo "Fixed file written to $WORKSPACE/fixed.py"
