fpca <- function(data, Q) {
  # Functional Principal Component Analysis
  
  # Functional grid
  grid <- c(1, 3, 6, 9, 12)
  
  # B-spline basis
  basis <- create.bspline.basis(
    rangeval = range(grid),
    nbasis = dim(data)[2],
    norder = 4
  )
  
  # Convert data into functional data
  data_fd <- smooth.basis(
    argvals = grid,
    y = t(as.matrix(data)),
    fdParobj = basis
  )$fd
  
  # FPCA
  fpca_machine <- pca.fd(
    data_fd,
    nharm = Q,
    centerfns = TRUE
  )
  
  # Evaluate FPCA eigenfunctions on the original grid
  # Q x number of columns of data
  phi_tilde_t <- t(
    eval.fd(grid, fpca_machine$harmonics)
  )
  
  # Extracted factors
  U <- as.matrix(fpca_machine$scores)
  
  return(
    list(
      U = U,
      phi_tilde_t = phi_tilde_t
    )
  )
}