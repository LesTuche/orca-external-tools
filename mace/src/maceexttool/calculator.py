from __future__ import annotations

import warnings
import os
MODEL_PATH = "/cluster/home/jlandis/ml_potentials/MACE-matpes-r2scan-omat-ft.model"

with warnings.catch_warnings():
    # suppress the missing PySisiphus warning
    warnings.simplefilter("ignore")
    from mace.calculators import MACECalculator
    from mace.calculators import mace_mp


def init(model):
    """Initialize an MACE r2scan foundation model instance."""
    print(
        f"Initializing MACE calculator with model: {model} for optimizations")

    # Try the foundation model first, fall back to local model loading if it fails
    try:
        return mace_mp(model=MODEL_PATH, device="cpu", default_dtype="float64")
    except Exception as e:
        print(f"Failed to load with mace_mp: {e}")
        print("Falling back to MACECalculator...")
        try:
            return MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
        except Exception as e2:
            print(f"Failed to load with MACECalculator: {e2}")
            raise
