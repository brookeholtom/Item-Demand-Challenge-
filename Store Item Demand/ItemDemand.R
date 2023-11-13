library(patchwork)
library(timetk)
library(tidyverse)
library(ggplot2)
library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)



train <- read.csv("C:/Users/brook/Downloads/STAT348/Store Item Demand/train.csv")
test <- read.csv("C:/Users/brook/Downloads/STAT348/Store Item Demand/test.csv")

plot1 <- train %>%
  filter(store==1, item==1) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365)

plot2 <- train %>%
  filter(store==2, item==2) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365)

plot3 <- train %>%
  filter(store==3, item==3) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365)

plot4 <- train %>%
  filter(store==4, item==4) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max=2*365)


wrap_plots(plot1, plot2, plot3, plot4)
