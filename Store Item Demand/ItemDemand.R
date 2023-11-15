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



train <- vroom("train.csv")
test <-  vroom("test.csv")

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


storeItem <- train %>%
filter(store==7, item==17) 

#Random Forest 
my_recipe <- recipe(sales ~., data=storeItem) %>%
step_date(date, features="doy") %>%
step_date(date, features="dow") #%>%
#step_range(date_doy, min=0, max=pi) %>%
#step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) 

prep <- prep(my_recipe)
bake(prep, new_data = storeItem)

my_mod_forest <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
workflow_forest <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_forest)


## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1, ncol(storeItem) - 1)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities



## Set up K-fold CV
folds <- vfold_cv(storeItem, v = 10, repeats=1)


CV_results_forest <- workflow_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))


## Find best tuning parameters
bestTune <- CV_results_forest %>%
  select_best("smape")

collect_metrics(CV_results_forest) 

## Finalize workflow and predict

final_wf_forest <-
  workflow_forest %>%
  finalize_workflow(bestTune) %>%
  fit(data=storeItem)


predictions_forest <- final_wf_forest %>%
  predict(test, type = "prob")

predictions_forest <- predictions_forest %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_forest, file="predictions_forest_final.csv", delim=",")





