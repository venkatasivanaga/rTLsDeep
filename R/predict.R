#' @title 3D PointNet Functions for rTLsDeep
#'
#' @description
#' Drop-in 3D equivalents of the rTLsDeep 2D workflow functions.
#' Works directly on 3D TLS point clouds — no 2D projection needed.
#'
#' Workflow mirror:
#'
#' \strong{Original 2D workflow:}
#' \preformatted{
#'   model       <- get_dl_model(...)
#'   weights     <- fit_dl_model(...)
#'   predictions <- predict_treedamage(model, weights, ...)
#'   cm          <- confmatrix_treedamage(predictions, ...)
#'   gcmplot(cm)
#' }
#'
#' \strong{New 3D workflow (this file):}
#' \preformatted{
#'   model       <- get_dl_model_3d(...)
#'   weights     <- fit_dl_model_3d(model, ...)
#'   predictions <- predict_treedamage_3d(model, weights, ...)
#'   cm          <- confmatrix_treedamage(predictions, ...)
#'   gcmplot(cm)
#' }
#'
#' @author Venkata Siva Naga, Carlos Alberto Silva
#' @name rTLsDeep_3d
NULL


# ============================================================
# INTERNAL HELPERS
# ============================================================

#' @noRd
.get_python_script <- function() {
  script <- system.file("python", "tree_classifier.py",
                        package = "rTLsDeep")
  if (script == "") {
    stop("tree_classifier.py not found. Please reinstall rTLsDeep.")
  }
  return(script)
}


#' @noRd
.get_model_dir <- function() {
  model_dir <- system.file("extdata", "output",
                           package = "rTLsDeep")
  if (model_dir == "") {
    stop("Model weights directory not found. Please reinstall rTLsDeep.")
  }
  return(model_dir)
}


#' @noRd
.setup_python <- function(conda_env = "pyn310") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' required: install.packages('reticulate')")
  }
  tryCatch(
    reticulate::use_condaenv(conda_env, required = FALSE),
    error = function(e) {
      message("Note: conda env '", conda_env,
              "' not found. Using default Python.")
    }
  )
}


#' @noRd
.load_classifier <- function(conda_env = "pyn310") {
  .setup_python(conda_env)
  script <- .get_python_script()
  classifier <- reticulate::import_from_path(
    "tree_classifier",
    path = dirname(script)
  )
  classifier$OUTPUT_DIR <- .get_model_dir()
  return(classifier)
}


# ============================================================
# 1. get_dl_model_3d()
#    Mirrors: get_dl_model()
# ============================================================

#' Get 3D PointNet model
#'
#' @description
#' Initialises a PointNet deep learning model for direct 3D point cloud
#' classification. This is the 3D equivalent of \code{get_dl_model()}.
#'
#' @param num_classes Integer. Number of damage classes. Default 6 (C1-C6).
#' @param num_points Integer. Points sampled per tree. Default 1024.
#' @param dropout Numeric. Dropout rate (0-1). Default 0.4.
#' @param conda_env Character. Conda environment with PyTorch. Default "pyn310".
#'
#' @return A list representing the PointNet model configuration, to be
#'   passed to \code{fit_dl_model_3d()} and \code{predict_treedamage_3d()}.
#'
#' @examples
#' \dontrun{
#' # Mirrors: model <- get_dl_model(model_type="vgg", ...)
#' model <- get_dl_model_3d(num_classes = 6)
#' }
#'
#' @seealso \code{\link{fit_dl_model_3d}}, \code{\link{predict_treedamage_3d}}
#' @export
get_dl_model_3d <- function(num_classes = 6L,
                             num_points  = 1024L,
                             dropout     = 0.4,
                             conda_env   = "pyn310") {

  classifier <- .load_classifier(conda_env)

  model_obj <- classifier$PointNet(
    num_classes = as.integer(num_classes),
    dropout     = dropout
  )

  message(sprintf(
    "PointNet 3D model initialised\n  Classes  : %d\n  Points   : %d\n  Dropout  : %.1f",
    num_classes, num_points, dropout
  ))

  return(list(
    model      = model_obj,
    classifier = classifier,
    num_classes= as.integer(num_classes),
    num_points = as.integer(num_points),
    conda_env  = conda_env
  ))
}


# ============================================================
# 2. fit_dl_model_3d()
#    Mirrors: fit_dl_model()
# ============================================================

#' Train the 3D PointNet model
#'
#' @description
#' Trains the PointNet model using 5-fold cross-validation on your
#' labeled tree dataset. Saves trained weights to the output directory.
#' This is the 3D equivalent of \code{fit_dl_model()}.
#'
#' Only needs to be run \strong{once} — trained weights are reused for
#' all future calls to \code{predict_treedamage_3d()}.
#'
#' @param model A model object from \code{get_dl_model_3d()}.
#' @param train_input_path Character. Path to training data folder
#'   containing C1..C6 subfolders with .las/.laz files.
#' @param epochs Integer. Number of training epochs. Default 150.
#' @param batch_size Integer. Batch size. Default 4.
#' @param lr_rate Numeric. Learning rate. Default 0.001.
#' @param label_smoothing Numeric. Label smoothing factor. Default 0.1.
#'
#' @return Character. Path to the output directory containing saved
#'   model weights (pointnet_fold1.pt ... pointnet_fold5.pt).
#'
#' @examples
#' \dontrun{
#' # Mirrors: weights <- fit_dl_model(model, train_path, test_path, ...)
#' model   <- get_dl_model_3d(num_classes = 6)
#' weights <- fit_dl_model_3d(
#'   model            = model,
#'   train_input_path = "data/",
#'   epochs           = 150L,
#'   batch_size       = 4L,
#'   lr_rate          = 0.001
#' )
#' }
#'
#' @seealso \code{\link{get_dl_model_3d}}, \code{\link{predict_treedamage_3d}}
#' @export
fit_dl_model_3d <- function(model,
                             train_input_path,
                             epochs          = 150L,
                             batch_size      = 4L,
                             lr_rate         = 0.001,
                             label_smoothing = 0.1) {

  if (!dir.exists(train_input_path)) {
    stop("Training data folder not found: ", train_input_path)
  }

  script    <- .get_python_script()
  model_dir <- .get_model_dir()

  message("Starting PointNet 3D training...")
  message("  Data dir   : ", train_input_path)
  message("  Output dir : ", model_dir)
  message("  Epochs     : ", epochs)
  message("  Batch size : ", batch_size)
  message("  LR         : ", lr_rate)
  message("\nThis will take ~2 hours on GPU. Models saved on completion.\n")

  cmd <- sprintf('python "%s" --mode train', script)
  ret <- system(cmd)

  if (ret == 0) {
    pt_files <- list.files(model_dir,
                           pattern = "pointnet_fold.*\\.pt$",
                           full.names = TRUE)
    message(sprintf("\nTraining complete! %d model files saved to:\n  %s",
                    length(pt_files), model_dir))
    return(invisible(model_dir))
  } else {
    warning("Training may have encountered errors (exit code: ", ret, ")")
    return(invisible(NULL))
  }
}


# ============================================================
# 3. predict_treedamage_3d()
#    Mirrors: predict_treedamage()
# ============================================================

#' Predict post-hurricane tree damage using 3D PointNet
#'
#' @description
#' Classifies trees into damage classes (C1-C6) directly from 3D TLS
#' point clouds. This is the 3D equivalent of \code{predict_treedamage()}.
#'
#' Uses an ensemble of 5 trained models with test-time augmentation
#' (20 rotations) for robust, stable predictions.
#'
#' @param model A model object from \code{get_dl_model_3d()}.
#' @param input_file_path Character. Either:
#'   \itemize{
#'     \item Path to a single .las/.laz file, OR
#'     \item Path to a folder of .las/.laz files (batch prediction)
#'   }
#' @param weights Character. Path to model weights directory containing
#'   pointnet_fold*.pt files. Default uses bundled package weights.
#' @param class_list Character vector. Damage classes to predict.
#'   Default c("C1","C2","C3","C4","C5","C6").
#' @param batch_size Integer. Not used (kept for API compatibility).
#' @param verbose Logical. Print prediction details. Default TRUE.
#'
#' @return Character vector of predicted damage classes, one per tree.
#'   Named by filename. Same format as \code{predict_treedamage()}.
#'
#' @examples
#' \dontrun{
#' # Mirrors: tree_damage <- predict_treedamage(model, path, weights, ...)
#'
#' model   <- get_dl_model_3d()
#' weights <- fit_dl_model_3d(model, "data/")
#'
#' # Single tree
#' pred <- predict_treedamage_3d(
#'   model           = model,
#'   input_file_path = "data/C3/Tree5_c3.laz",
#'   weights         = weights,
#'   class_list      = c("C1","C2","C3","C4","C5","C6")
#' )
#' print(pred)  # "C3"
#'
#' # Folder of trees
#' preds <- predict_treedamage_3d(
#'   model           = model,
#'   input_file_path = "data/validation/",
#'   weights         = weights,
#'   class_list      = c("C1","C2","C3","C4","C5","C6")
#' )
#' print(preds)
#' # Tree1_c3.laz  Tree2_c1.laz  Tree3_c5.laz
#' #        "C3"          "C1"          "C5"
#' }
#'
#' @seealso \code{\link{get_dl_model_3d}}, \code{\link{fit_dl_model_3d}},
#'   \code{\link{confmatrix_treedamage}}
#' @export
predict_treedamage_3d <- function(model,
                                   input_file_path,
                                   weights    = NULL,
                                   class_list = c("C1","C2","C3",
                                                  "C4","C5","C6"),
                                   batch_size = 4L,
                                   verbose    = TRUE) {

  classifier <- model$classifier

  # Override output dir if custom weights path provided
  if (!is.null(weights) && dir.exists(weights)) {
    classifier$OUTPUT_DIR <- weights
  } else {
    classifier$OUTPUT_DIR <- .get_model_dir()
  }

  # Load ensemble models
  models <- classifier$load_models()

  # Determine if input is a file or folder
  is_folder <- dir.exists(input_file_path)
  is_file   <- file.exists(input_file_path) &&
               grepl("\\.(las|laz)$", input_file_path, ignore.case = TRUE)

  if (!is_folder && !is_file) {
    stop("input_file_path must be a .las/.laz file or a folder: ",
         input_file_path)
  }

  # Collect files
  if (is_file) {
    las_files <- input_file_path
  } else {
    las_files <- c(
      list.files(input_file_path, pattern = "\\.las$",
                 full.names = TRUE, ignore.case = TRUE),
      list.files(input_file_path, pattern = "\\.laz$",
                 full.names = TRUE, ignore.case = TRUE)
    )
    if (length(las_files) == 0) {
      stop("No .las/.laz files found in: ", input_file_path)
    }
  }

  if (verbose) {
    message(sprintf("\nPredicting %d tree(s)...\n", length(las_files)))
  }

  # Predict each tree
  predictions <- character(length(las_files))
  names(predictions) <- basename(las_files)

  for (i in seq_along(las_files)) {
    result     <- classifier$predict_tree(las_files[i], models)
    pred_class <- result[[1]]
    probs      <- unlist(result[[2]])
    confidence <- max(probs)
    low_conf   <- confidence < 0.35

    predictions[i] <- pred_class

    if (verbose) {
      flag <- ifelse(low_conf, " [LOW CONF]", "")
      cat(sprintf("  %-30s  ->  %s  (%.1f%%)%s\n",
                  basename(las_files[i]),
                  pred_class,
                  confidence * 100,
                  flag))
    }
  }

  if (verbose) cat("\n")
  return(predictions)
}


# ============================================================
# 4. get_validation_classes_3d()
#    Mirrors: get_validation_classes()
# ============================================================

#' Get true damage classes from a labelled folder
#'
#' @description
#' Extracts true damage class labels from filenames or folder structure.
#' This is the 3D equivalent of \code{get_validation_classes()}.
#'
#' Supports two structures:
#' \itemize{
#'   \item Files named like \code{Tree1_c3.laz} → class parsed from filename
#'   \item Files inside \code{C3/} subfolders → class parsed from folder name
#' }
#'
#' @param file_path Character. Path to folder of .las/.laz files.
#' @param class_list Character vector. Valid class names. Default C1-C6.
#'
#' @return Named character vector of true class labels, one per file.
#'
#' @examples
#' \dontrun{
#' # Mirrors: test_classes <- get_validation_classes(file_path = "validation/")
#' true_classes <- get_validation_classes_3d("data/validation/")
#' }
#'
#' @export
get_validation_classes_3d <- function(file_path,
                                       class_list = paste0("C", 1:6)) {

  las_files <- c(
    list.files(file_path, pattern = "\\.las$",
               full.names = TRUE, ignore.case = TRUE),
    list.files(file_path, pattern = "\\.laz$",
               full.names = TRUE, ignore.case = TRUE)
  )

  if (length(las_files) == 0) {
    stop("No .las/.laz files found in: ", file_path)
  }

  true_classes <- character(length(las_files))
  names(true_classes) <- basename(las_files)

  for (i in seq_along(las_files)) {
    fname  <- basename(las_files[i])
    parent <- basename(dirname(las_files[i]))

    # Try folder name first (e.g. file is inside C3/)
    if (toupper(parent) %in% toupper(class_list)) {
      true_classes[i] <- toupper(parent)
      next
    }

    # Try filename pattern (e.g. Tree1_c3.laz)
    m <- regmatches(fname,
                    regexpr("_c([1-6])", fname, ignore.case = TRUE))
    if (length(m) > 0) {
      true_classes[i] <- toupper(gsub("_", "", m))
      next
    }

    true_classes[i] <- NA_character_
    warning("Could not determine class for: ", fname)
  }

  return(true_classes)
}