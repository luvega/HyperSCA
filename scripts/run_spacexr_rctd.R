#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(spacexr)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  idx <- match(name, args)
  if (is.na(idx) || idx >= length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

as_bool <- function(value) {
  tolower(as.character(value)) %in% c("1", "true", "t", "yes", "y")
}

read_count_bundle <- function(input_dir, prefix) {
  counts <- readMM(file.path(input_dir, paste0(prefix, "_counts.mtx")))
  genes <- readLines(file.path(input_dir, paste0(prefix, "_genes.tsv")), warn = FALSE)
  barcodes <- readLines(file.path(input_dir, paste0(prefix, "_barcodes.tsv")), warn = FALSE)
  rownames(counts) <- genes
  colnames(counts) <- barcodes
  as(counts, "dgCMatrix")
}

input_dir <- get_arg("--input-dir")
output_dir <- get_arg("--output-dir")
doublet_mode <- get_arg("--doublet-mode", "doublet")
max_cores <- as.integer(get_arg("--max-cores", "1"))
test_mode <- as_bool(get_arg("--test-mode", "false"))
reference_min_umi <- as.numeric(get_arg("--reference-min-umi", "100"))
umi_min <- as.numeric(get_arg("--umi-min", "100"))
counts_min <- as.numeric(get_arg("--counts-min", "10"))
umi_min_sigma <- as.numeric(get_arg("--umi-min-sigma", "300"))
cell_min_instance <- as.integer(get_arg("--cell-min-instance", "25"))

if (is.null(input_dir) || is.null(output_dir)) {
  stop("--input-dir and --output-dir are required")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

reference_counts <- read_count_bundle(input_dir, "reference")
query_counts <- read_count_bundle(input_dir, "query")

labels <- fread(file.path(input_dir, "reference_labels.csv"))
cell_types <- factor(labels$cell_type)
names(cell_types) <- labels$barcode

coords <- fread(file.path(input_dir, "query_coords.csv"))
coords <- as.data.frame(coords)
rownames(coords) <- coords$barcode
coords <- coords[, c("x", "y"), drop = FALSE]

reference <- Reference(reference_counts, cell_types, min_UMI = reference_min_umi)
puck <- SpatialRNA(coords, query_counts)
my_rctd <- create.RCTD(
  puck,
  reference,
  max_cores = max_cores,
  test_mode = test_mode,
  UMI_min = umi_min,
  counts_MIN = counts_min,
  UMI_min_sigma = umi_min_sigma,
  CELL_MIN_INSTANCE = cell_min_instance
)
my_rctd <- run.RCTD(my_rctd, doublet_mode = doublet_mode)

results_df <- as.data.frame(my_rctd@results$results_df)
results_df$obs_id <- rownames(results_df)
setcolorder(results_df, c("obs_id", setdiff(colnames(results_df), "obs_id")))
fwrite(results_df, file.path(output_dir, "results_df.csv.gz"))

cell_type_names <- levels(reference@cell_types)
abundance <- matrix(0, nrow = nrow(results_df), ncol = length(cell_type_names))
rownames(abundance) <- rownames(results_df)
colnames(abundance) <- cell_type_names

if (doublet_mode == "doublet") {
  weights_doublet <- as.matrix(my_rctd@results$weights_doublet)
  obs_ids <- rownames(results_df)
  spot_class <- as.character(results_df$spot_class)
  first_type <- as.character(results_df$first_type)
  second_type <- as.character(results_df$second_type)

  first_weight <- rep(1, length(obs_ids))
  second_weight <- rep(0, length(obs_ids))
  weight_rows <- rownames(weights_doublet)
  has_weight <- !is.null(weight_rows) & obs_ids %in% weight_rows
  if (any(has_weight) && ncol(weights_doublet) >= 1) {
    weight_idx <- match(obs_ids[has_weight], weight_rows)
    first_weight[has_weight] <- as.numeric(weights_doublet[weight_idx, 1])
    if (ncol(weights_doublet) > 1) {
      second_weight[has_weight] <- as.numeric(weights_doublet[weight_idx, 2])
    }
  }

  valid_first <- !is.na(spot_class) & spot_class != "reject" & !is.na(first_type) & first_type %in% cell_type_names
  first_rows <- which(valid_first)
  if (length(first_rows) > 0) {
    first_values <- ifelse(spot_class[first_rows] == "singlet", 1, first_weight[first_rows])
    abundance[cbind(first_rows, match(first_type[first_rows], cell_type_names))] <- first_values
  }

  valid_second <- spot_class %in% c("doublet_certain", "doublet_uncertain") &
    !is.na(second_type) & second_type %in% cell_type_names
  second_rows <- which(valid_second)
  if (length(second_rows) > 0) {
    abundance[cbind(second_rows, match(second_type[second_rows], cell_type_names))] <- second_weight[second_rows]
  }
} else {
  weights <- as.matrix(my_rctd@results$weights)
  if (nrow(weights) == nrow(results_df)) {
    abundance <- weights[rownames(results_df), cell_type_names, drop = FALSE]
  } else if (ncol(weights) == nrow(results_df)) {
    abundance <- t(weights[cell_type_names, rownames(results_df), drop = FALSE])
  } else {
    stop("Unsupported RCTD weights shape for non-doublet mode")
  }
}

abundance_df <- as.data.frame(abundance)
abundance_df$obs_id <- rownames(abundance_df)
setcolorder(abundance_df, c("obs_id", setdiff(colnames(abundance_df), "obs_id")))
fwrite(abundance_df, file.path(output_dir, "abundance.csv.gz"))
