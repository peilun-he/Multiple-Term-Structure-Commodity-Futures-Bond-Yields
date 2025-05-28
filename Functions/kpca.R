kpca <- function(data, kernel, gamma, Q) {
  # Kernal Principal Component Analysis
  kpca_machine <- sk_dec$KernelPCA(kernel = kernel,
                                   gamma = gamma,
                                   n_components = as.integer(Q))
  
  X_kpca <- kpca_machine$fit_transform( t(as.matrix(data)) ) # principal component - A
  eig_values <- kpca_machine$eigenvalues_ # eigen values 
  eig_vector <- kpca_machine$eigenvectors_ # eigen vectors
  X_fit <- kpca_machine$X_fit_
  Lambda <- t(X_kpca) %*% X_kpca
  W <- X_kpca %*% solve(Lambda)
  V <- t(eig_vector)
  
  rbf_kernel <- function(x, y, gamma) exp( -gamma * sum((x-y)^2) )
  
  K <- apply(t(data), 1, function(x) 
    apply(t(data), 1, function(y)
      rbf_kernel(x, y, gamma = kpca_machine$gamma_)
    )
  )
  
  phi_tilde_t <- 0 
  for (t in 1: Q){
    phi_tilde_t <- phi_tilde_t + as.vector(matrix(K[t, ], nrow = 1) %*% W) * V
  }
  
  U <- matrix(0, nrow = dim(data)[1], ncol = Q) # extracted factors
  for (t in 1: dim(data)[1]){
    for (j in 1: Q) {
      U[t, j] <- sum(data[t, ] * phi_tilde_t[j, ])
    }
  }
  
  return(list(U = U, 
              phi_tilde_t = phi_tilde_t))
}