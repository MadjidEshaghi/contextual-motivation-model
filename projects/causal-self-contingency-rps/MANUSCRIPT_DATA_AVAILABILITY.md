# Manuscript-ready Data and Code Availability

## Data availability

The study distinguishes newly collected experimental data from previously published public data.

For behavioral replication and descriptive validation, the public `WXZ2014` object distributed with the R package `stratEst` is used. The original experiment is reported by Wang, Xu, and Zhou (2014), DOI: https://doi.org/10.1038/srep05830. The public `stratEst` object contains 72 participants observed for 300 periods (21,600 rows) and is therefore treated as a public subset rather than the complete 360-participant experiment. The authoritative data documentation is https://search.r-project.org/CRAN/refmans/stratEst/html/WXZ2014.html. Because the package copyright file does not grant a general downstream redistribution license for the WXZ data, the data are retrieved from the authoritative package and are not mirrored in this repository.

The supplementary material associated with Dyson et al. (2016), DOI: https://doi.org/10.1038/srep20479, is available from the official Scientific Reports/PMC record at https://pmc.ncbi.nlm.nih.gov/articles/PMC4740902/. The listed spreadsheet `srep20479-s2.xls` is treated as supplementary/summary material unless trial-level structure is independently verified; it is not represented as a raw trial-level dataset.

Exploratory EEG analyses use the canonical OpenNeuro dataset `ds006761`, version 1.0.0, DOI: https://doi.org/10.18112/openneuro.ds006761.v1.0.0, associated with Moerel et al., DOI: https://doi.org/10.1093/scan/nsaf101. OpenNeuro is the primary source. The raw dataset is released under CC0 and is not duplicated in GitHub because of its size.

For every external dataset, the repository records the original DOI/URL, version, license, data status (raw/processed/summary), and retrieval procedure in `data/data_manifest.csv`.

## Code availability

Simulation, model-recovery, preprocessing, statistical-analysis, and figure/table reproduction code is maintained at:

https://github.com/MadjidEshaghi/contextual-motivation-model/tree/main/projects/causal-self-contingency-rps

The executable implementation is governed by `simulator_spec.md` and all deviations and Gate decisions are recorded in `locked_log.md`. A versioned release/tag should be cited in the accepted manuscript so that the exact analysis state remains immutable.

## Reproducibility rule

The GitHub repository is the analysis/code location, not the primary source for external datasets. Original dataset DOIs and publisher/repository URLs must remain in the article even when a redistributable copy or derived file is also present in the analysis repository.
