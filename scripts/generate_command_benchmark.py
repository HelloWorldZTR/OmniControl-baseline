#!/usr/bin/env python3
"""Run in the native OmniControl environment; no frankenmotion imports required."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_benchmark.generate import main

if __name__ == "__main__":
    main()
