#' Setup Python and TensorFlow environment
#'
#' Installs Python and TensorFlow if they are not already available.
#' This function is intended for interactive use only.
#'
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