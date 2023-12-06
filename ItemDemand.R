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
library(forecast)
library(modeltime)



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
step_date(date, features="dow") %>%
step_naomit()
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


#Exponential Smoothing
library(modeltime)
library(timetk)

Item1 <- train %>%
  filter(store==7, item==8)

Item2 <- train %>%
  filter(store==1, item==8)


cv_split <- time_series_split(Item1, assess='3 months', cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


cv_split2 <- time_series_split(Item2, assess='3 months', cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


es_model <- exp_smoothing() %>%
set_engine("ets") %>%
fit(sales~date, data=training(cv_split))

es_model2 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split2))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

cv_results2 <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split2))

## Visualize CV results
Plot1_Nov17 <- cv_results %>%
modeltime_forecast(
                   new_data = testing(cv_split),
                   actual_data = Item1
) %>%
plot_modeltime_forecast(.interactive=TRUE)

Plot2_Nov17 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = Item2
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)


## Evaluate the accuracy
cv_results %>%
modeltime_accuracy() %>%
table_modeltime_accuracy(
                         .interactive = FALSE
)
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )


es_fullfit <- cv_results %>%
modeltime_refit(data = Item1)

es_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = Item2)


es_preds <- es_fullfit %>%
modeltime_forecast(h = "3 months") %>%
rename(date=.index, sales=.value) %>%
select(date, sales) %>%
full_join(., y=test, by="date") %>%
select(id, sales)

es_preds2 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)



Plot3_Nov17 <-  es_fullfit %>%
modeltime_forecast(h = "3 months", actual_data = Item1) %>%
plot_modeltime_forecast(.interactive=FALSE)

Plot4_Nov17 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = Item1) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(Plot1_Nov17, Plot2_Nov17, Plot3_Nov17, Plot4_Nov17, nrows=2)



#Nov 27 Arima Model
Item1 <- train %>%
  filter(store==1, item==17)

Item2 <- train %>%
  filter(store==1, item==18)


cv_split <- time_series_split(Item1, assess='3 months', cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


cv_split2 <- time_series_split(Item2, assess='3 months', cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


arima_recipe <- recipe(sales ~., data=Item1) %>% # For the linear model part
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_naomit()

arima_recipe_2 <- recipe(sales ~., data=Item2) %>% # For the linear model part
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_naomit()

arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
set_engine("auto_arima")


arima_wf <- workflow() %>%
add_recipe(arima_recipe) %>%
add_model(arima_model) %>%
fit(data=training(cv_split))

arima_wf_2 <- workflow() %>%
  add_recipe(arima_recipe_2) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split2))


## Cross-validate to tune model
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))
cv_results_2 <- modeltime_calibrate(arima_wf_2,
                                  new_data = testing(cv_split2))

## Visualize CV results

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

cv_results_2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )


arima_fullfit <- cv_results %>%
  modeltime_refit(data = Item1)

arima_fullfit2 <- cv_results_2 %>%
  modeltime_refit(data = Item2)

Item1_test <- test %>%
  filter(store==1, item==17)
Item2_test <- test %>%
  filter(store==1, item==18)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(new_data = Item1_test) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

arima_preds2 <- arima_fullfit2 %>%
  modeltime_forecast(new_data = Item2) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)


#plots for Nov 27 
Plot1_Nov27 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = Item1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

Plot2_Nov27 <- arima_fullfit %>%
  modeltime_forecast(new_data = Item1_test, actual_data = Item1) %>%
  plot_modeltime_forecast(.interactive=FALSE)

Plot3_Nov27 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = Item2
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)


Plot4_Nov27 <- arima_fullfit2 %>%
  modeltime_forecast(new_data = Item2_test, actual_data = Item2) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(Plot1_Nov27, Plot2_Nov27, Plot3_Nov27, Plot3_Nov27, nrows=2)



#November 29th 
Item1 <- train %>%
  filter(store==1, item==17)

Item2 <- train %>%
  filter(store==1, item==18)

cv_split <- time_series_split(Item1, assess='3 months', cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


cv_split2 <- time_series_split(Item2, assess='3 months', cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


prophet_model <- prophet_reg() %>%
set_engine(engine = "prophet") %>%
fit(sales ~ date, data = training(cv_split))

prophet_model2 <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split2))


cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))
cv_results_2 <- modeltime_calibrate(prophet_model2,
                                    new_data = testing(cv_split2))
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

cv_results_2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

prophet_fullfit <- cv_results %>%
  modeltime_refit(data = Item1)

prophet_fullfit2 <- cv_results_2 %>%
  modeltime_refit(data = Item2)

Item1_test <- test %>%
  filter(store==1, item==17)
Item2_test <- test %>%
  filter(store==1, item==18)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(new_data = Item1_test) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

prophet_preds2 <- prophet_fullfit2 %>%
  modeltime_forecast(new_data = Item2) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)


Plot1_Nov29 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = Item1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

Plot2_Nov29 <- prophet_fullfit %>%
  modeltime_forecast(new_data = Item1_test, actual_data = Item1) %>%
  plot_modeltime_forecast(.interactive=FALSE)

Plot3_Nov29 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = Item2
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)


Plot4_Nov29 <- prophet_fullfit2 %>%
  modeltime_forecast(new_data = Item2_test, actual_data = Item2) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(Plot1_Nov29, Plot3_Nov29, Plot2_Nov29, Plot4_Nov29, nrows=2)



#Submission Code 

nStores <- max(train$store)
nItems <- max(train$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
    filter(store==s, item==i)
    storeItemTest <- test %>%
    filter(store==s, item==i)
    
    ## Fit storeItem models here
    
    ## Predict storeItem sales
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(all_preds, file=..., delim=...)




