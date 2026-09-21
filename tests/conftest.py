"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add project root to sys.path so bench module can be imported
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
