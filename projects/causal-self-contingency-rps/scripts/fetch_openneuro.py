#!/usr/bin/env python3
"""Record canonical source for OpenNeuro ds006761.

The ~78 GB raw EEG dataset is intentionally not duplicated in Git history.
"""

from pathlib import Path

DATASET = "ds006761"
VERSION = "1.0.0"
DOI = "10.18112/openneuro.ds006761.v1.0.0"
URL = "https://openneuro.org/datasets/ds006761"

text = f"""Canonical dataset: {URL}
Dataset DOI: https://doi.org/{DOI}
Version: {VERSION}
License: CC0

Use OpenNeuro's documented download methods and keep the BIDS tree outside
Git history, e.g. data/external/{DATASET}/.
"""
print(text)
Path("data").mkdir(exist_ok=True)
Path("data/openneuro_ds006761_SOURCE.txt").write_text(text, encoding="utf-8")
