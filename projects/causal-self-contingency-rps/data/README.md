# External data provenance

This folder never replaces the authoritative repositories. The manuscript must cite the original article/dataset DOI separately from this GitHub analysis repository.

## 1. Wang–Xu–Zhou (WXZ2014)

Original article:
- https://doi.org/10.1038/srep05830
- https://www.nature.com/articles/srep05830

Public trial-level subset:
- CRAN `stratEst::WXZ2014`
- https://search.r-project.org/CRAN/refmans/stratEst/html/WXZ2014.html
- Package DOI: https://doi.org/10.32614/CRAN.package.stratEst

The public object has 72 participants × 300 periods = 21,600 rows. The original article reports 360 participants, so this object is explicitly treated as a subset.

### Redistribution rule

The `stratEst` COPYRIGHT file identifies the WXZ data as copyright Zhijian Wang, Bin Xu, and Hai-Jun Zhou but does not attach the CC BY license that is explicitly attached to some other datasets in the same package. The package paper thanks the original authors for permission to include the data in `stratEst`. That package-specific permission is not treated here as a general downstream redistribution license.

**Therefore this repository does not mirror WXZ raw bytes.** Use `scripts/fetch_wxz2014.R` to obtain the authoritative public subset and record a local checksum.

## 2. Dyson et al. 2016

Article:
- https://doi.org/10.1038/srep20479
- https://www.nature.com/articles/srep20479
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4740902/

Official supplementary file:
- `srep20479-s2.xls` (55 KB), listed by the article/PMC record.

The article and listed supplementary material are distributed under CC BY 4.0 unless a file-specific credit line states otherwise. Available evidence indicates that the XLS contains participant-level/derived summary tables, so it is **not classified as trial-level raw data** without further file-structure verification.

It may be mirrored with attribution after file-level inspection; the authoritative publisher/PMC location remains the primary source.

## 3. OpenNeuro ds006761

Canonical source:
- https://openneuro.org/datasets/ds006761
- Dataset DOI: https://doi.org/10.18112/openneuro.ds006761.v1.0.0
- Associated paper: https://doi.org/10.1093/scan/nsaf101
- License: CC0

The dataset contains 64-channel EEG from 62 participants (31 pairs) playing 480 RPS games. OpenNeuro is the primary source. Because the dataset is about 78 GB, GitHub stores metadata/retrieval instructions rather than duplicating the raw BIDS tree.

## Manuscript citation rule

For every empirical number:
1. cite the original article and/or canonical dataset DOI;
2. state whether the analysis uses raw, processed, or summary data;
3. cite this GitHub repository separately for analysis code and reproducibility materials.
