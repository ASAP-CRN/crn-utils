"""CRN-UTILS"""

__version__ = "0.3.0"

# All consumers use explicit submodule imports (e.g. from crn_utils.util import X).
# Star-imports removed to prevent eager loading of heavy/env-specific dependencies
# (doi, file_metadata) as a side effect of importing any crn_utils submodule.