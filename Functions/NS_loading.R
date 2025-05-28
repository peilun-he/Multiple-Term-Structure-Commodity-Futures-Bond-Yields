NS_loading <- function(lambda, mats) {
  loading <- rbind(rep(1, length(mats)), 
                   (1 - exp(-lambda*mats)) / (lambda*mats), 
                   (1 - exp(-lambda*mats)) / (lambda*mats) - exp(-lambda*mats))
  return(loading)
}