library(reticulate) # import Python packages
library(ggplot2)
library(dplyr)
library(expm)
library(tidyr)
library(YieldCurve)
library(zoo) 
library(pracma)
library(plotly)
library(ftsa)
library(glmnet)
library(lmridge)
library(lmtest)
library(statespacer)
library(scales)
library(lubridate)
library(fda)

setwd("/Users/HPL/Documents/GitHub/Multiple-Term-Structure-Commodity-Futures-Bond-Yields/")
source("Functions/NS_loading.R")
source("Functions/match_maturity.R")
source("Functions/KF_NS.R")
source("Functions/kpca.R")
source("Functions/fpca.R")
source("Functions/pca.R")

########################
##### Prepare data #####
########################
# US monthly data 
US <- read.csv("Data/USTreasury_20250513.csv")

US$time <- as.Date(US$time)
US <- US %>%
  filter(time >= as.Date("2003-01-01") & time <= as.Date("2022-12-31"))

rownames(US) <- US$time
date <- US$time
US <- US[, c(2, 4, 6, 7)]
mat_US <- c(1, 3, 6, 12) # maturity of US data

# Match maturity through Static Nelson-Siegel model
maturity <- c(1, 3, 6, 9, 12)
US <- match_maturity(data = US, 
                     original_maturity = mat_US, 
                     target_maturity = maturity)$target_data
US$time <- date

# WTI crude oil futures data
WTI <- read.csv("Data/WTI_Formatted_20250513.csv")
WTI$Trade.Date <- as.Date(WTI$Trade.Date)
WTI <- WTI %>%
  filter(Trade.Date >= as.Date("2003-01-01") & Trade.Date <= as.Date("2022-12-31"))

index <- !duplicated(year(WTI$Trade.Date) + month(WTI$Trade.Date)/1000)
WTI <- WTI[index, 1: 13]

# Period 1
US1 <- US %>%
  filter(time >= as.Date("2003-01-01") & time <= as.Date("2007-09-18"))
WTI1 <- WTI %>%
  filter(Trade.Date >= as.Date("2003-01-01") & Trade.Date <= as.Date("2007-09-18"))

# Period 2
US2 <- US %>%
  filter(time >= as.Date("2007-09-18") & time <= as.Date("2008-12-16"))
WTI2 <- WTI %>%
  filter(Trade.Date >= as.Date("2007-09-18") & Trade.Date <= as.Date("2008-12-16"))

# Period 3
US3 <- US %>%
  filter(time >= as.Date("2008-12-16") & time <= as.Date("2020-03-03"))
WTI3 <- WTI %>%
  filter(Trade.Date >= as.Date("2008-12-16") & Trade.Date <= as.Date("2020-03-03"))

# Period 4
US4 <- US %>%
  filter(time >= as.Date("2020-03-03") & time <= as.Date("2022-12-31"))
WTI4 <- WTI %>%
  filter(Trade.Date >= as.Date("2020-03-03") & Trade.Date <= as.Date("2022-12-31"))

# Save data
write.csv(WTI1, "Data/WTI1.csv", row.names = FALSE)
write.csv(US1[, 1:5], "Data/USTreasury_period1.csv", row.names = FALSE)

###########################
##### FPCA on US data #####
###########################
dat_fpca <- US1[, 1:5] # data for KPCA
Q <- 3
U <- pca(data = dat_fpca, Q = Q)$U

write.csv(U, "Data/Data1/US_PCA_3factors_period1.csv", row.names = FALSE)

###########################
##### KPCA on US data #####
###########################
dat_kpca <- US4[, 1:5] # data for KPCA
use_python("/Users/HPL/anaconda3/bin/python3") # select python version
pd <- import("pandas")
np <- import("numpy")
sk_dec <- import("sklearn.decomposition")
sk_met <- import("sklearn.metrics")
sk_ms <- import("sklearn.model_selection")

# Estimate hyper-parameters
score <- function(estimator, X, Y = NULL) {
  X_reduced <- estimator$fit_transform(X)
  X_preimage <- estimator$inverse_transform(X_reduced)
  return(-sk_met$mean_squared_error(X, X_preimage))
}

param_grid <- list(gamma = seq(from = 0.001, to = 1, by = 0.001))

Q <- 3 # number of factors. (max = 12)
kpca_machine <- sk_dec$KernelPCA(kernel = "rbf",  
                                 n_components = as.integer(Q), 
                                 fit_inverse_transform = TRUE)

set.seed(1234)
grid_search <- sk_ms$GridSearchCV(kpca_machine, 
                                  param_grid, 
                                  cv = as.integer(3), scoring = score)
hp_fit <- grid_search$fit(t(as.matrix(dat_kpca))) # hyper-parameter 

hp_fit$best_params_

# Extract factors
U <- kpca(data = dat_kpca, kernel = "rbf", gamma = hp_fit$best_params_$gamma, Q = Q)$U

# heatmap of U
colnames(U) <- paste("PC", 1: Q, sep = "")
U %>%
  as.data.frame() %>%
  gather(key = "PC", value = "values") %>%
  ggplot(mapping = aes(y = rep(as.Date(rownames(dat_kpca)), Q), x = PC, fill = values)) + 
  geom_tile() + 
  ylab("Time") + 
  ggtitle("Heatmap of U")

# Save factors
write.csv(U, "Data/US_2factors_period1.csv", row.names = FALSE)

###################################
##### Functional coefficients #####
###################################
n_contract <- 12
gamma <- as.matrix(read.csv("Data/Coe/coe_period4_3factors.csv", header = FALSE))
phi_tilde_t <- kpca(data = dat_kpca, kernel = "rbf", gamma = hp_fit$best_params_$gamma, Q = 3)$phi_tilde_t
gamma_functional <- as.data.frame(t(gamma %*% phi_tilde_t))
colnames(gamma_functional) <- paste("Contract", 1: n_contract)

colors = c(rgb(252,174,145, maxColorValue = 255), 
           rgb(251,106,74, maxColorValue = 255),
           rgb(222,45,38, maxColorValue = 255),
           rgb(165,15,21, maxColorValue = 255),
           rgb(189,215,231, maxColorValue = 255),
           rgb(107,174,214, maxColorValue = 255),
           rgb(49,130,189, maxColorValue = 255),
           rgb(8,81,156, maxColorValue = 255),
           rgb(186,228,179, maxColorValue = 255),
           rgb(116,196,118, maxColorValue = 255),
           rgb(49,163,84, maxColorValue = 255),
           rgb(0,109,44, maxColorValue = 255))

gamma_functional %>% 
  pivot_longer(cols = 1: n_contract, names_to = "Contract", values_to = "Values") %>%
  mutate(US_maturity = rep(maturity, each = n_contract)) %>% 
  mutate(Contract = factor(Contract, levels = paste("Contract", 1: n_contract))) %>%
  ggplot(aes(x = US_maturity, y = Values, color = Contract)) + 
  geom_line() + 
  theme_classic(base_size = 20) + 
  scale_color_manual(labels = c(expression(gamma[1](tau)), 
                                expression(gamma[2](tau)), 
                                expression(gamma[3](tau)),
                                expression(gamma[4](tau)),
                                expression(gamma[5](tau)),
                                expression(gamma[6](tau)),
                                expression(gamma[7](tau)),
                                expression(gamma[8](tau)),
                                expression(gamma[9](tau)),
                                expression(gamma[10](tau)),
                                expression(gamma[11](tau)),
                                #expression(gamma[12](tau))), values = hue_pal()(12)) + 
                                expression(gamma[12](tau))), values = colors) + 
  scale_x_continuous(breaks = c(3, 6, 9, 12)) + 
  xlab("Time to maturity") + 
  labs(color = expression(paste(gamma, " function")))

gamma_functional %>%
  gather(key = "Contract", value = "Values") %>%
  mutate(US_maturity = rep(maturity, times = dim(US)[2])) %>%
  plot_ly(x = ~US_maturity, y = ~Values, type = "scatter", mode = "lines", color = ~Contract) %>%
  layout(title = "Functional coefficients", xaxis = list(title = "US Contract"))

##########################
##### Visualisations #####
##########################
library(ggplot2)
library(scales)

# FOMC interest rates change
rate_changes <- data.frame(
  Date = as.Date(c(
    "2003-06-25", "2004-06-30", "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20",
    "2005-11-01", "2005-12-13", "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2007-09-18", "2007-10-31", "2007-12-11", "2008-01-22", "2008-01-30", "2008-03-18",
    "2008-04-30", "2008-10-08", "2008-10-29", "2008-12-16", "2015-12-16", "2016-12-14",
    "2017-03-15", "2017-06-14", "2017-12-13", "2018-03-21", "2018-06-13", "2018-09-26",
    "2018-12-19", "2019-07-31", "2019-09-18", "2019-10-30", "2020-03-03", "2020-03-15",
    "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02",
    "2022-12-14"
  )),
  Change_bps = c(
    -25, 25, 25, 25, 25, 25,
    25, 25, 25, 25, 25, 25,
    25, 25, 25, 25, 25, 25,
    -50, -25, -25, -75, -50, -75,
    -25, -50, -50, -100, 25, 25,
    25, 25, 25, 25, 25, 25,
    25, -25, -25, -25, -50, -100,
    25, 50, 75, 75, 75, 75,
    50), 
  Rate = c(
    1.00, 1.25, 1.50, 1.75, 2.00, 2.25,
    2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25,
    4.50, 4.75, 5.00, 5.25, 4.75, 4.50, 4.25,
    3.50, 3.00, 2.25, 2.00, 1.50, 1.00, 0.25,
    0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.25, 2.50,
    2.25, 2.00, 1.75, 1.25, 0.25,
    0.50, 1.00, 1.75, 2.50, 3.25, 4.00, 4.50)
)

period_lines <- as.Date(c("2007-09-18", "2008-12-16", "2020-03-03"))

ggplot(rate_changes, aes(x = Date, y = Change_bps)) +
  geom_col(fill = ifelse(rate_changes$Change_bps > 0, "green", "red")) +
  geom_hline(yintercept = 0, linetype = "solid") +
  geom_vline(xintercept = period_lines, linetype = "dashed", colour = "black") +
  annotate("text", x = as.Date("2005-01-01"), y = 110, label = "Period 1", size = 5) +
  annotate("text", x = as.Date("2008-03-01"), y = 110, label = "Period 2", size = 5) +
  annotate("text", x = as.Date("2015-01-01"), y = 110, label = "Period 3", size = 5) +
  annotate("text", x = as.Date("2021-06-01"), y = 110, label = "Period 4", size = 5) +
  scale_y_continuous(
    breaks = seq(-100, 100, by = 25),
    labels = function(x) paste0(x, " bps")
  ) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(x = "Date of FOMC Meeting", y = "Change (bps)") +
  theme_minimal(base_size = 14)

# Create step plot
ggplot(rate_changes, aes(x = Date, y = Rate)) +
  geom_step(direction = "hv", linewidth = 1, color = "steelblue") +
  geom_vline(xintercept = period_lines, linetype = "dashed", color = "black") +
  annotate("text", x = as.Date("2005-01-01"), y = 6, label = "Period 1", size = 5) +
  annotate("text", x = as.Date("2008-03-01"), y = 6, label = "Period 2", size = 5) +
  annotate("text", x = as.Date("2015-01-01"), y = 6, label = "Period 3", size = 5) +
  annotate("text", x = as.Date("2021-06-01"), y = 6, label = "Period 4", size = 5) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(x = "Date", y = "Target Rate") +
  theme_minimal(base_size = 14)

# Crude Oil futures price
colors = c(rgb(252,174,145, maxColorValue = 255), 
           rgb(251,106,74, maxColorValue = 255),
           rgb(222,45,38, maxColorValue = 255),
           rgb(165,15,21, maxColorValue = 255),
           rgb(189,215,231, maxColorValue = 255),
           rgb(107,174,214, maxColorValue = 255),
           rgb(49,130,189, maxColorValue = 255),
           rgb(8,81,156, maxColorValue = 255),
           rgb(186,228,179, maxColorValue = 255),
           rgb(116,196,118, maxColorValue = 255),
           rgb(49,163,84, maxColorValue = 255),
           rgb(0,109,44, maxColorValue = 255))

period_lines <- as.Date(c("2007-09-18", "2008-12-16", "2020-03-03"))

WTI %>%
  pivot_longer(col = 2:13, names_to = "Maturity", values_to = "Price") %>%
  mutate(Maturity = as.numeric(substring(Maturity, 4))) %>%
  mutate(Maturity = paste(Maturity, ifelse(Maturity>1, "months", "month"))) %>%
  mutate(Maturity = factor(Maturity, levels = paste(1: 12, c("month", rep("months", 11))))) %>%
  ggplot(mapping = aes(x = Trade.Date, y = Price, color = Maturity)) +
  geom_line() + 
  theme_classic(base_size = 20) + 
  geom_vline(xintercept = period_lines, linetype = "dashed", colour = "black") +
  annotate("text", x = as.Date("2005-01-01"), y = 150, label = "Period 1", size = 5) +
  annotate("text", x = as.Date("2008-03-01"), y = 150, label = "Period 2", size = 5) +
  annotate("text", x = as.Date("2015-01-01"), y = 150, label = "Period 3", size = 5) +
  annotate("text", x = as.Date("2021-08-01"), y = 150, label = "Period 4", size = 5) +
  scale_color_manual(values = colors) + 
  xlab("Date") 

# US Treasury
colors <- c(
  rgb(228, 26, 28, maxColorValue = 255),   # red
  rgb(55, 126, 184, maxColorValue = 255),  # blue
  rgb(77, 175, 74, maxColorValue = 255),   # green
  rgb(152, 78, 163, maxColorValue = 255),  # purple
  rgb(255, 127, 0, maxColorValue = 255)    # orange
)

US %>%
  rename(`1 month` = close.1m, `3 months` = close.3m, `6 months` = close.6m, `9 months` = close.9m, `12 months` = close.1y) %>%
  pivot_longer(cols = 1: 5, names_to = "Maturity", values_to = "Yield") %>%
  mutate(Maturity = factor(Maturity, levels = c("1 month", "3 months", "6 months", "9 months", "12 months"))) %>%
  ggplot(aes(x = time, y = Yield, color = Maturity)) + 
  geom_line() + 
  theme_classic(base_size = 20) + 
  geom_vline(xintercept = period_lines, linetype = "dashed", colour = "black") +
  annotate("text", x = as.Date("2005-01-01"), y = 5.5, label = "Period 1", size = 5) +
  annotate("text", x = as.Date("2008-03-01"), y = 5.5, label = "Period 2", size = 5) +
  annotate("text", x = as.Date("2015-01-01"), y = 5.5, label = "Period 3", size = 5) +
  annotate("text", x = as.Date("2021-08-01"), y = 5.5, label = "Period 4", size = 5) +
  scale_color_manual(values = colors) + 
  xlab("Date") 

# Panel charts
for (i in 1: 12) {
  df <- data.frame(Period1 = gamma_functional1[, i], 
                   Period2 = gamma_functional2[, i],
                   Period3 = gamma_functional3[, i],
                   Period4 = gamma_functional4[, i])
  
  colors <- c(rgb(0, 114, 178, maxColorValue = 255), 
              rgb(213, 94, 0, maxColorValue = 255), 
              rgb(0, 158, 115, maxColorValue = 255), 
              rgb(240, 228, 66, maxColorValue = 255))
  
  p <- df %>%
    pivot_longer(cols = 1: 4, names_to = "Period", values_to = "Values") %>%
    mutate(US_maturity = rep(maturity, each = 4)) %>%
    ggplot(aes(x = US_maturity, y = Values, color = Period)) + 
    geom_line() + 
    theme_classic(base_size = 20) + 
    scale_color_manual(labels = c("Period 1", 
                                  "Period 2", 
                                  "Period 3",
                                  "Period 4"), values = colors) + 
    scale_x_continuous(breaks = c(3, 6, 9, 12)) + 
    xlab("Time to maturity") + 
    labs(color = "Period") + 
    ylim(c(-0.3, 0.2))
  
  print(p)
}


