"""Print the first CLI argument unchanged (fixture skill helper)."""

import sys

print(sys.argv[1] if len(sys.argv) > 1 else "")
