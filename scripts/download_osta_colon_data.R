#!/usr/bin/env Rscript
# =============================================================================
# download_osta_colon_data.R
# 下载 OSTA.data 中所有 HumanColon_Oliveira 数据集到 data/ 目录
# 用法: Rscript HyperSCA/scripts/download_osta_colon_data.R
# =============================================================================

# ---------- 0. 设置目标目录（使用绝对路径避免歧义） ----------
data_root <- normalizePath(file.path("data"), mustWork = FALSE)
dir.create(data_root, showWarnings = FALSE, recursive = TRUE)

# ---------- 1. 安装依赖（若缺失） ----------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("OSTA.data", quietly = TRUE)) {
  BiocManager::install("OSTA.data", ask = FALSE, update = FALSE)
}

library(OSTA.data)

# ---------- 辅助函数: 使用 PowerShell Expand-Archive 解压 ----------
ps_unzip <- function(zip_path, dest_dir) {
  zip_abs  <- normalizePath(zip_path, winslash = "\\")
  dest_abs <- normalizePath(dest_dir, winslash = "\\", mustWork = FALSE)
  dir.create(dest_abs, showWarnings = FALSE, recursive = TRUE)

  # 构建 PowerShell 命令
  ps_script <- sprintf(
    "Expand-Archive -LiteralPath '%s' -DestinationPath '%s' -Force",
    zip_abs, dest_abs
  )
  cat(sprintf("  [解压] PowerShell Expand-Archive -> %s\n", dest_abs))

  exit_code <- system2(
    "powershell",
    args = c("-NoProfile", "-NonInteractive", "-Command", shQuote(ps_script)),
    stdout = TRUE, stderr = TRUE
  )

  # system2 在 stdout=TRUE 时返回字符向量，status 在 attr
  status <- attr(exit_code, "status")
  if (!is.null(status) && status != 0) {
    stop(sprintf("Expand-Archive 失败 (exit %d): %s",
                 status, paste(exit_code, collapse = "\n")))
  }
  invisible(TRUE)
}

# ---------- 辅助函数: 校验解压结果 ----------
validate_extraction <- function(out_dir, dataset_id) {
  files <- list.files(out_dir, recursive = TRUE, full.names = TRUE)

  if (length(files) == 0) {
    stop(sprintf("解压目录为空: %s", out_dir))
  }

  # 检查文件大小 — 全部为 0 字节说明解压失败
  sizes <- file.size(files)
  zero_files <- files[!is.na(sizes) & sizes == 0]
  real_files <- files[!is.na(sizes) & sizes > 0]

  cat(sprintf("  [校验] 文件总数: %d, 有效文件: %d, 空文件: %d\n",
              length(files), length(real_files), length(zero_files)))

  # 检查关键文件是否存在
  expected <- list(
    Chromium   = "filtered_feature_bc_matrix.h5",
    Visium     = "filtered_feature_bc_matrix.h5",
    VisiumHD   = "filtered_feature_bc_matrix.h5",
    Xenium     = "experiment.xenium"
  )

  # 根据数据集类型确定关键文件
  key <- NULL
  if (grepl("^Chromium_", dataset_id))  key <- expected$Chromium
  if (grepl("^Visium_",   dataset_id))  key <- expected$Visium
  if (grepl("^VisiumHD_", dataset_id))  key <- expected$VisiumHD
  if (grepl("^Xenium_",   dataset_id))  key <- expected$Xenium

  if (!is.null(key)) {
    matches <- grep(key, files, value = TRUE, fixed = TRUE)
    if (length(matches) == 0) {
      warning(sprintf("关键文件缺失: %s", key))
    } else {
      sz <- file.size(matches[1])
      cat(sprintf("  [校验] 关键文件 %s 大小: %s\n",
                  key, format(sz, big.mark = ",")))
      if (is.na(sz) || sz == 0) {
        stop(sprintf("关键文件 %s 大小为 0, 解压可能失败", key))
      }
    }
  }

  # 返回相对路径列表
  list.files(out_dir, recursive = TRUE)
}

# ---------- 2. 定义待下载数据集 ----------
datasets <- c(
  "Chromium_HumanColon_Oliveira",
  "Visium_HumanColon_Oliveira",
  "VisiumHD_HumanColon_Oliveira",
  "Xenium_HumanColon_Oliveira"
)

cat("========================================\n")
cat("OSTA.data 结直肠癌数据下载器\n")
cat("目标目录:", data_root, "\n")
cat("解压方式: PowerShell Expand-Archive\n")
cat("数据集数量:", length(datasets), "\n")
cat("========================================\n\n")

# ---------- 3. 逐个下载并解压 ----------
results <- data.frame(
  dataset  = character(),
  status   = character(),
  n_files  = integer(),
  message  = character(),
  stringsAsFactors = FALSE
)

for (id in datasets) {
  cat(sprintf("[%s] 开始下载...\n", id))
  out_dir <- file.path(data_root, id)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  tryCatch({
    # 对 Xenium/CosMx 可传 mol/pol 参数；其他数据集不需要
    if (grepl("^Xenium_", id)) {
      pa <- OSTA.data_load(id, mol = TRUE, pol = TRUE)
    } else if (grepl("^CosMx", id)) {
      pa <- OSTA.data_load(id, mol = FALSE, pol = FALSE)
    } else {
      pa <- OSTA.data_load(id)
    }

    cat(sprintf("  zip 路径: %s\n", pa))
    cat(sprintf("  zip 大小: %s bytes\n", format(file.size(pa), big.mark = ",")))

    # 使用 PowerShell 解压（替代 R 内置 unzip）
    ps_unzip(pa, out_dir)

    # 校验
    files <- validate_extraction(out_dir, id)
    cat(sprintf("  解压完成, 共 %d 个文件:\n", length(files)))
    for (f in files) cat(sprintf("    - %s\n", f))

    results <- rbind(results, data.frame(
      dataset = id, status = "SUCCESS",
      n_files = length(files), message = "",
      stringsAsFactors = FALSE
    ))
  }, error = function(e) {
    msg <- conditionMessage(e)
    cat(sprintf("  [ERROR] %s\n", msg))
    results <<- rbind(results, data.frame(
      dataset = id, status = "FAILED",
      n_files = 0L, message = msg,
      stringsAsFactors = FALSE
    ))
  })
  cat("\n")
}

# ---------- 4. 汇总报告 ----------
cat("========================================\n")
cat("下载汇总报告\n")
cat("========================================\n")
print(results)
cat(sprintf("\n成功: %d / %d\n",
            sum(results$status == "SUCCESS"), nrow(results)))

if (any(results$status == "FAILED")) {
  cat("\n失败的数据集:\n")
  failed <- results[results$status == "FAILED", ]
  for (i in seq_len(nrow(failed))) {
    cat(sprintf("  - %s: %s\n", failed$dataset[i], failed$message[i]))
  }
}

# 列出 data 目录最终状态
cat("\n========================================\n")
cat("data/ 目录最终状态:\n")
cat("========================================\n")
for (id in datasets) {
  d <- file.path(data_root, id)
  if (dir.exists(d)) {
    ff <- list.files(d, recursive = TRUE, full.names = TRUE)
    total_size <- sum(file.size(ff), na.rm = TRUE)
    cat(sprintf("  %s: %d 文件, 总大小 %s bytes\n",
                id, length(ff), format(total_size, big.mark = ",")))
  } else {
    cat(sprintf("  %s: [目录不存在]\n", id))
  }
}

cat("\n完成!\n")
