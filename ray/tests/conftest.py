"""Make `import code.<module>` find ray/code instead of Python's standard-library `code` module.

pytest imports pdb, and pdb imports the standard `code` module, before any test is collected.
So ray/ goes first on sys.path and the standard module is dropped from the import cache."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not hasattr(sys.modules.get("code"), "__path__"):
    sys.modules.pop("code", None)
