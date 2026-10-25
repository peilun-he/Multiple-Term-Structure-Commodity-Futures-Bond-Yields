pca <- function(data, Q) {
  # Principal Component Analysis
  
  data <- as.matrix(data)
  
  # PCA
  pca_machine <- prcomp(
    data,
    center = TRUE,
    scale. = FALSE
  )
  
  # Extracted factors: T x Q
  U <- pca_machine$x[, 1:Q, drop = FALSE]
  
  # Eigenvectors: Q x 5
  phi_tilde_t <- t(
    pca_machine$rotation[, 1:Q, drop = FALSE]
  )
  
  return(
    list(
      U = U,
      phi_tilde_t = phi_tilde_t
    )
  )
}