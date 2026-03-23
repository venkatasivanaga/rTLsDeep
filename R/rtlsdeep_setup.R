#' Setup Python and TensorFlow environment
#'
#' Installs Python and TensorFlow if they are not already available.
#' This function is intended for interactive use only.
#'
#' @examples
#' \dontrun{
#' tensorflow_dir = NA
#' model_type = "simple"
#' train_image_files_path = system.file('extdata', 'train', package='rTLsDeep')
#' test_image_files_path = system.file('extdata', 'validation', package='rTLsDeep')
#' img_width <- 256
#' img_height <- 256
#' class_list_train = unique(list.files(train_image_files_path))
#' class_list_test = unique(list.files(test_image_files_path))
#' lr_rate = 0.0001
#' target_size <- c(img_width, img_height)
#' channels = 4
#'
#' rtlsdeep_setup()
#' }
#' @return Invisible TRUE on success.
#' @export
rtlsdeep_setup <- function() {
  if (!interactive()) {
    stop("rtlsdeep_setup() is for interactive use only and should not be run during package checks.")
  }
  
  if (!reticulate::py_available(initialize = FALSE)) {
    message("Python not found. Installing Python...")
    reticulate::install_python()
  }
  
  if (!reticulate::py_module_available("tensorflow")) {
    message("TensorFlow not found. Installing TensorFlow...")
    tensorflow::install_tensorflow()
  }
  
  invisible(TRUE)
}