"""
ARJUN - Hybrid World Model Training Entry Point

Run from the ARJUN project root:

    python train_hybrid.py

This script calls the new delta-based training pipeline in:

    world_model/hybrid_trainer.py
"""

from __future__ import annotations

import sys
from pathlib import Path


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# IMPORT TRAINER
# ================================================================

from world_model.hybrid_trainer import train


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 72)
    print(
        "ARJUN HYBRID WORLD MODEL"
    )
    print("=" * 72)

    train()


if __name__ == "__main__":
    main()