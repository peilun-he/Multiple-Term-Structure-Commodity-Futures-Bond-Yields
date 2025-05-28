simulate_data_NS <- function(par, mats, n_obs, n_contract, lam, seed) {
  # Simulate data, state variables and time to maturities from Nelson Siegel model. 
  # Inputs: 
  #   par: a vector of parameters
  #   mats: a vector of maturities 
  #   n_obs: number of observations
  #   n_contract: number of contracts
  #   seed: seed to generate random values
  # Outputs: 
  #   yt: data
  #   xt: state variables
  
  require(MASS)
  
  if (length(par) != 9+n_contract) {
    stop("Incorrect number of parameters. ")
  }
  
  phi0 <- c(par[1], par[2], par[3])
  phi1 <- diag(c(par[4], par[5], par[6]))
  #lam <- par[7]
  sigma_w <- diag( par[7:9]^2 )
  sigma_v <- diag( par[10: length(par)]^2 )
  
  # Random noises
  set.seed(seed)
  noise_xt <- mvrnorm(n_obs, c(0,0,0), sigma_w)
  noise_yt <- mvrnorm(n_obs, rep(0, n_contract), sigma_v)
  
  # Simulate xt
  xt <- matrix(0, nrow = n_obs+1, ncol = 3)
  xt[1, ] <- phi0 / (1 - diag(phi1))
  for (i in 1: n_obs) {
    xt[i+1, ] <- phi0 + phi1 %*% xt[i, ] +  noise_xt[i, ]
  }
  xt <- xt[-1, ]
  
  
  # Simulate yt
  yt <- matrix(0, nrow = n_obs, ncol = n_contract)
  for (i in 1: n_obs) {
    loading <- NS_loading(lam, mats)
    yt[i, ] <- t(loading) %*% xt[i, ] + noise_yt[i, ]
    
  }
  return(list(yt = yt, xt = xt))
}

