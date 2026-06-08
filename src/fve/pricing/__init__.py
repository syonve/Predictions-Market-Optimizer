"""Pricing layer: devigging and consensus fair value.

Pure functions only. No IO. The devig method is pluggable and selected by
config / the calibration harness — never silently hardcoded.
"""

from fve.pricing.devig import (
    DevigMethod,
    booksum,
    devig,
    implied_probs,
    overround,
    power,
    proportional,
    shin,
)

__all__ = [
    "DevigMethod",
    "booksum",
    "devig",
    "implied_probs",
    "overround",
    "power",
    "proportional",
    "shin",
]
