"""
QUBEKit
"""
# Handel versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision = versions["full-revisionid"]
del get_versions, versions
