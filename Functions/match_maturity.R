match_maturity <- function(data, original_maturity, target_maturity) {
  # Match maturity through Static Nelson-Siegel model
  
  require(YieldCurve)
  
  n_obs <- dim(data)[1]
  n_contract <- dim(data)[2]
  
  new_maturity <- target_maturity[!(target_maturity %in% original_maturity)]
  n_new <- length(new_maturity) # number of new contracts
  
  NS_parameters <- Nelson.Siegel(rate = data, maturity = original_maturity)
  SNS <- matrix(0, nrow = n_obs, ncol = n_contract + n_new) # static Nelson-Siegel
  
  for (i in 1: n_obs){
    beta <- NS_parameters[i, 1: 3]
    lambda <- NS_parameters[i, 4]
    SNS[i, ] <- as.matrix(beta) %*% NS_loading(lambda, c(original_maturity, new_maturity))
  }
  
  rmse <- sqrt(colMeans( (SNS[, 1: n_contract] - data)^2 ))
  
  target_data <- data
  
  for(i in 1: n_new){
    maturity <- new_maturity[i]
    if (maturity > 12) {
      maturity <- maturity/12
      name <- paste("close.", maturity, "y", sep = "")
    } else {
      name <- paste("close.", maturity, "m", sep = "")
    }
    target_data[[name]] <- SNS[, n_contract + i]
  }
  
  target_data <- target_data[, order(c(original_maturity, new_maturity))]
  
  return(list(target_data = target_data, 
              SNS_rmse = rmse))
}