#!/usr/bin/env python3
"""Run in the frankenmotion Python environment, reusing its task metrics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.score import main

if __name__ == "__main__":
    main()
