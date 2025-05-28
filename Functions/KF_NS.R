KF_NS <- function(par, yt, mats, lam) {
  # Standard Kalman Filter for Nelson Siegel model
  # Model: 
  #   y_t = d_t + F_t' x_t + f_t + v_t, v_t ~ N(0, V), observation equation
  #   x_t = c + G x_{t-1} + w_t, w_t ~ N(0, W), state equation
  #   f_t = b*t + beta*cos(2*pi*t*dt) + eta*sin(2*pi*t*dt), seasonal effect
  # Inputs: 
  #   par: a vector of parameters
  #   yt: the logarihm of futures prices
  #   T: maturities
  #   delivery_time: a vector of date, which is necessary if seasonality is "Constant"
  #   dt: delta t
  #   smoothing: a boolean variable indicate if Kalman Smoothing is required
  #   seasonality: "Constant" or "None"
  # Outputs: 
  #   nll: the negative log likelihood 
  #   ll_table: a vector to store cumulative log-likelihood at each time point - used to calculate Sandwich variance
  #   table_at_filter: a nT*2 matrix gives the filtered values of state variables. 
  #   table_at_prediction: a (nT+1)*2 matrix gives the predicted values of state variables. 
  #   table_at_smoother: a nT*2 matrix gives the smoothed values of state variables. The algorithm of Kalman Smoother is given by Bierman (1973) and De Jong (1989). 
  #   ft: seasonal effect
  
  require(lubridate)
  
  n_obs <- dim(yt)[1]
  n_contract <- dim(yt)[2]
  
  table_xt_filter <- matrix(0, nrow = n_obs, ncol = 3) # a_t|t
  table_Pt_filter <- array(0, dim = c(3, 3, n_obs)) # P_t|t
  table_xt_prediction <- matrix(0, nrow = n_obs+1, ncol = 3) # a_t|t-1
  table_Pt_prediction <- array(0, dim = c(3, 3, n_obs+1)) # P_t|t-1
  
  nll <- 0 # negative log-likelihood 
  ll_table <- matrix(0, nrow = 1, ncol = n_obs) # table of log-likelihood
  
  table_et <- matrix(0, nrow = n_obs, ncol = n_contract) # e_t
  table_L <- array(0, dim = c(n_contract, n_contract, n_obs)) # Covariance of y
  table_y <- matrix(0, nrow = n_obs, ncol = n_contract) # y_hat
  
  # Parameters
  if (length(par) != 9+n_contract) {
    stop("Incorrect number of parameters. ")
  }
  
  phi0 <- c(par[1], par[2], par[3])
  phi1 <- diag(c(par[4], par[5], par[6]))
  x0 <- phi0 / (1 - diag(phi1))
  #lam <- par[7]
  sigma_w <- diag( par[7:9]^2 )
  sigma_v <- diag( par[10: length(par)]^2 )
  
  # Initialization
  xt_filter <- x0
  Pt_filter <- diag( diag(sigma_w) / (1 - diag(phi1)^2) )
  
  # Kalman Filter
  for (i in 1:n_obs) {
    lambda <- rbind(rep(1, n_contract), 
                    (1 - exp(-lam * mats[i, ])) / (lam * mats[i, ]), 
                    (1 - exp(-lam * mats[i, ])) / (lam * mats[i, ]) - exp(-lam * mats[i, ]))
    
    # Prediction step
    xt_prediction  <- phi0 + phi1 %*% xt_filter # a_t+1|t 
    Pt_prediction <- phi1 %*% Pt_filter %*% t(phi1) + sigma_w # P_t+1|t
    y_prediction <- t(lambda) %*% xt_prediction # ytilde_t|t-1 = d_t + F_t a_t|t-1
    
    # Filter step
    et <- yt[i, ] - t(y_prediction) # e_t = y_t - ytilde_t|t-1
    L <- t(lambda) %*% Pt_prediction %*% lambda + sigma_v # Covariance matrix of et
    invL <- solve(L) # inverse of L 
    K <- Pt_prediction %*% lambda %*% invL # Kalman gain matrix: K_t
    
    xt_filter <- xt_prediction + K %*% t(et) # a_t
    Pt_filter <- (diag(3) - K %*% t(lambda)) %*% Pt_prediction # P_t
    #Pt_filter <- (diag(3) - K %*% t(lambda)) %*% Pt_prediction %*% t(diag(3) - K %*% t(lambda)) + K %*% sigma_v %*% t(K) 
    
    # Update tables
    table_xt_filter[i, ] <- t(xt_filter)
    table_Pt_filter[, , i] <- Pt_filter
    table_xt_prediction[i+1, ] <- t(xt_prediction)
    table_Pt_prediction[, , i+1] <- Pt_prediction
    table_et[i, ] <- et
    table_L[, , i] <- L
    table_y[i, ] <- y_prediction
    
    if (det(L)<0) {
      message(i)
      message("matrix is not semi positive definite (KF)")
    }
    
    # Update likelihood 
    nll <- nll + 0.5*length(yt[i, ])*log(2*pi) + 0.5*log(det(L)) + 0.5*et %*% solve(L) %*% t(et)
    ll_table[i] <- -(0.5*length(yt[i, ])*log(2*pi) + 0.5*log(det(L)) + 0.5*et %*% solve(L) %*% t(et))
  }
  
  return(list(nll = nll, 
              ll_table = ll_table, 
              xt_filter = table_xt_filter,
              Pt_filter = table_Pt_filter, 
              xt_prediction = table_xt_prediction, 
              cov_y = table_L, 
              y_hat = table_y))
}