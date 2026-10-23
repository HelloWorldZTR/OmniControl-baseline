#!/usr/bin/env python3
"""Run in the frankenmotion Python environment; see docs/command_benchmark.md."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.export import main

if __name__ == "__main__":
    main()
