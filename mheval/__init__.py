"""Mental Health Evaluation Harness."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mheval")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"
