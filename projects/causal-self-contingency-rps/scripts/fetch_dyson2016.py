#!/usr/bin/env python3
"""Record authoritative Dyson et al. (2016) supplementary-data locations.

Raw XLS mirroring is deferred until file-level license/credit lines are checked.
"""

from pathlib import Path

ARTICLE_DOI = "https://doi.org/10.1038/srep20479"
ARTICLE_URL = "https://www.nature.com/articles/srep20479"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4740902/"
SUPPLEMENT_NAME = "srep20479-s2.xls"

text = f"""Dyson et al. (2016)
Article DOI: {ARTICLE_DOI}
Official article: {ARTICLE_URL}
PMC record: {PMC_URL}
Official supplementary filename: {SUPPLEMENT_NAME}

The article is CC BY 4.0. Before mirroring the XLS, inspect supplementary
credit/license lines and record SHA256.
"""
print(text)
Path("data").mkdir(exist_ok=True)
Path("data/dyson2016_SOURCE.txt").write_text(text, encoding="utf-8")
