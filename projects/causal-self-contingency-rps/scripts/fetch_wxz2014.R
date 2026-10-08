# Public WXZ2014 trial-level subset from CRAN stratEst.
# Primary scientific citation: https://doi.org/10.1038/srep05830
# Public data documentation:
# https://search.r-project.org/CRAN/refmans/stratEst/html/WXZ2014.html
#
# Run from project root:
#   Rscript scripts/fetch_wxz2014.R

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("stratEst", quietly = TRUE)) {
  install.packages("stratEst", repos = "https://cloud.r-project.org")
}

data("WXZ2014", package = "stratEst", envir = environment())
stopifnot(nrow(WXZ2014) == 21600)

write.csv(
  WXZ2014,
  "data/raw/WXZ2014_stratEst.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

message("Wrote data/raw/WXZ2014_stratEst.csv")
message("Cite DOI 10.1038/srep05830 and the CRAN stratEst data documentation.")
