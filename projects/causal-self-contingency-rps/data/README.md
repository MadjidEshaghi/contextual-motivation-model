# External data provenance

This folder does **not** replace the authoritative repositories.

## 1. Wang–Xu–Zhou RPS data

Original article:
- Wang Z, Xu B, Zhou H-J (2014), *Scientific Reports* 4:5830.
- DOI: https://doi.org/10.1038/srep05830

Public trial-level subset:
- CRAN package `stratEst`, object `WXZ2014`
- Documentation: https://search.r-project.org/CRAN/refmans/stratEst/html/WXZ2014.html
- Package DOI: https://doi.org/10.32614/CRAN.package.stratEst
- Documentation states 72 participants × 300 periods = 21,600 rows.

Important: the original article reports 360 participants. The CRAN object is therefore treated as a public subset, not the complete original study.

Raw bytes are not mirrored here until the package `COPYRIGHTS`/dataset-level redistribution terms are checked. Use `scripts/fetch_wxz2014.R`.

## 2. Dyson et al. 2016

Article:
- DOI: https://doi.org/10.1038/srep20479
- Official article: https://www.nature.com/articles/srep20479
- PMC article record: https://pmc.ncbi.nlm.nih.gov/articles/PMC4740902/

The official/PMC records list supplementary file `srep20479-s2.xls`. The article is CC BY 4.0, but redistribution here is deferred until the supplementary file is checked for any separate credit/license restriction.

## 3. OpenNeuro ds006761

Canonical source:
- https://openneuro.org/datasets/ds006761
- DOI: https://doi.org/10.18112/openneuro.ds006761.v1.0.0
- License: CC0

Associated paper:
- https://doi.org/10.1093/scan/nsaf101

The raw dataset is approximately 78 GB, so this repository stores retrieval instructions rather than duplicating the bytes.

## Citation rule for the manuscript

Always cite the **original article/dataset DOI** first. The GitHub repository is cited separately as the analysis/reproducibility location.
