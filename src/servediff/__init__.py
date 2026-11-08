"""servediff: diff what an LLM inference server actually ran with."""

from servediff.cli import _version, diff, main, parse

# Read from the installed distribution rather than a literal: a hardcoded copy drifts
# silently, and `servediff.__version__` still said 0.1.0 on the 0.4.9 wheel.
__version__ = _version()
__all__ = ["diff", "main", "parse", "__version__"]
