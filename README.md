# Predict Future Sales 💰⏰
[![alt text](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/kaggle.png "title")](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

> In this project I work with a **time series** dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - [1C Company](http://www.1c.com/)🇷🇺. The goal for this project is to predict total sales for every product and store in the next month. 

* [Kaggle Link](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

> Based on my business sense and business investigation, I use Python to detect and split text variables, and implement **label encoding**, **mean encoding** on the categorical variables extracted from original text information. Finally, I Incremented features from only 4 to 39 and using **XGboost**, **LightGBM** and Random Forest for the prediction. The ranking on Kaggle is at top 25% with rmse = 0.90912.


+ See more details and data visualization at `prediction.ipynb` or [Link](https://www.kaggle.com/jiakaicheng/final)



## Content
* Look at Data Quickly
* EDA
* Feature Engineering
* Methodology
* Result
* End


## Look at Data Quickly


<img src="https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/data.png" width="800">

The data told us that **`sales_train.csv`** can be merged with **`items.csv.csv`**, **`item_categories.csv`**, **`shops.csv`** by shop_id、item_id、item_category_id. After we finish that, you can see the dataset below.  

Prediction Goal: **`item_cnt_month`** True target values are clipped into [0,20] range.

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/merge.png)

## EDA

### ***sales vs price***
Since we know sales is highly connected wth product prices, for example, promotion on price can be a great way to stimulate sales, so I analyze its relationship first.

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/price.png)

As you can see, most products' prices are below 100 thousand, and daily sales are below 1000. Also price has a **`negative relationship`** with sales, and it's what I expected. Then, remove the **`outliers`** and the wrong data, for instance, the product with negative number of price. 
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/outlier1.png)
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/outlier2.png)


### ***sales vs month***
As time goes, by here are two discoveries:
1/ The total sales is decling every year.
2/ The total sales has periodic feature
3/ Sales peak occurs at Decemeber, like it's relevent to Christmas Day.  

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/time.png)


### ***shop name***
Although shop name is written in Russian, we can still find something regular here. Actually every shop name is divided by blank space ‘’, and first one is the **`city name`** , the second one, if existed, is probably the shop type (Ц、ТРЦ、ТК..), and the last one is the actual name for every unique store.

At the same time, I found some shop names is similar, which can be regarded as typo, let's revise them.
  
Here I modified string to int type, hoping to save space.

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/shop.png)


### ***item and cat name***

Same logic above, just extract more info from item name and category name

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/cat.png)
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/item.png)
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/item1.png)



## Feature Engineering

Content
- Lag features
- Mean Encoding
- Price trend
- Revenue trend
- Dummy feacture & other
- Feature Mergence

Before doing this part, converted daily sales to monthly sales, and merge all datasets I've modified above.
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/all.png)

### ***Lag features + Mean Encoding***
Since it's a time series quetsion, I include lag features to help prediction. When doing label encoding, it has some shortcomings. Since I have tem-million observations, the value of observations that have been encoded could have passive influences on the prediction result because they have real number meanings. To deal with the wide-scattered observations, I can process them with mean encoding.

Actually every feature has real meanings instead of just raw data processing.
```
date_avg_item_cnt——【1】：mean monthly sales for 1C company 
date_item_avg_item_cnt——【1/2/3/6/12】: mean monthly sales per ITEM
date_shop_avg_item_cnt——【1/2/3/6/12】: mean monthly sales per SHOP
date_cat_avg_item_cnt——【1】: mean monthly sales per ITEM-CATEGORY
date_cat_item_f1_avg_item_cnt——【1】: mean monthly sales per ITEM-CATEGORY-F1
date_cat_item_f2_avg_item_cnt——【1】: mean monthly sales per ITEM-CATEGORY-F2
date_shop_cat_avg_item_cnt——【1】: mean monthly sales per SHOP per ITEM-CATEGORY
date_shop_type_avg_item_cnt——【1】: mean monthly sales per SHOP-TYPE
date_shop_subtype_avg_item_cnt——【1】: mean monthly sales per SHOP-SUBTYPE
date_city_avg_item_cnt——【1】: mean monthly sales per CITY
```

### ***Price trend***
Then use lag features to analyze sales trends.
```
item_avg_item_price: mean prices of an item in all time
date_item_avg_item_price : mean price of an item in every month
delta_price_lag——【1/2/3/4/5/6】: price fluctuation by month
```


### ***Revenue trend***
same logic above
```
date_shop_revenue : revenue of a shop by month
shop_avg_revenue : mean revenue of a shop in every month 
delta_revenue : revenue trend of a shop in every month
delta_revenue_lag_1——【1】 : revenue trend of a shop from the last month
```
### ***Dummy feature & other***

As we have noticed from EDA that every December has the highest sales of the year, we can use dummy variable to deal with this cycle. At first, transfer“date_block_num” to “month”, and then assign value 1 to December and 0 to the rest of eleven months.

Beside, I think that the lowest and highest price of an item in history, the first sales of an item can be a great reflection on new products’ performances, so I incremented 3 new features here. Finally, we can independently introduce some external data, the number of holidays on each month, to help predict the seasonality of sales.

![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/dummy.png)

### ***Feature Mergence***

In the end, we can successfully increment 35 new features through all the work done above, merge them together, and enrich the original training set to a 11,056,276 rows plus 39 columns matrix. We believe that these features can greatly improve the prediction accuracy.





## Methodology

### ***Algorithms***
I have employed three different algorithms:
**`Random Forest`**, **`lightGBM`**, **`XGBoost`**



### ***Result Evaluation***
In this part, I choose Root Mean Square Error(RMSE) as evaluation metric for testing dataset.
To compare among three different algorithms, I build and train all three models with default parameters. And the results are shown below:
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/result.png)
Among these three models, Random Forest plays worst with underfitting problem. LightGBM and XGBoost both work very well without overfitting or underfitting problems. However, LightGBM need less time and iterations for training model than XGBoost.


### ***Parameter Tuning***
XGBoost algorithm has become the ultimate weapon of many data scientists. It’s a highly sophisticated algorithm, powerful enough to deal with all sorts of irregularities of data.

Building a model using XGBoost is easy. However, improving the model using XGBoost is difficult. This algorithm uses multiple parameters. To improve the model, parameter tuning is must. Hence, in this part I’ll present the steps of parameter tuning along with some useful information about XGBoost's parameters.

* Step 1. Fix number of estimators  
In order to decide on boosting parameters, I need to set some initial values of other parameters. Therefore, I take the following values:  
**`learning_rate = 0.1 `** : Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems.   
**`max_depth = 5  `** : This should be between 3-10. I’ve started with 5 but a different number can be chosen as well. 4-6 can be good starting points.  
**`min_child_weight = 1 `** : We choose the default value for this parameter.
**`gamma = 0 `** : I choose the default value for this parameter.
**`subsample, colsample_bytree = 0.8 `** :This is a commonly used as start value. Typical values range between 0.5-0.9.  


All the above settings are just initial estimates and will be tuned later. Then I check the optimum number of estimators using both cv function[1]and early-stopping method of XGBoost.
First, I use early-stopping method to find out the number of estimators until the model hasn't improved in 100 rounds.  


```python
params = {'learning_rate': 0.1, 'max_depth': 5, 
          'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'eval_metric':'rmse'}
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
rgs = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
verbose_eval=10)
```  

  
  
Result:
Stopping. Best iteration:[74]  

Then I use GridSearchCV to find out the optimum number of estimators around 74.  
```python
cv_params = {'n_estimators': [70,71,72,73,74]}
other_params={'objective':'reg:squarederror','learning_rate':0.1,
'n_estimators':500,'max_depth':5,'min_child_weight':1,
'subsample':0.8, 'colsample_bytree': 0.8,
                'gamma': 0,'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
optimized_xgb=GridSearchCV(estimator=model,param_grid=cv_params, 					verbose=1,n_jobs=-1)
optimized_xgb.fit(x_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('best value for the parameter：{0}'.format(optimized_xgb.best_params_))
```  

Result:
best value for the parameter：{'n_estimators': 72}  
Hence, I will set 72 as the value of n_estimators for tuning tree-based parameters in the following steps.


* Step 2. Tune max_depth and min_child_weight  
I tune these first as they will have the highest impact on model outcome. To start with, I set wide ranges: from 3 to 10 for max_depth and 1 to 6 for min_child_weight.
```python
cv_params = {'max_depth':range(3,11), 'min_child_weight':range(1,7)}
Result:
best value for the parameter：{'max_depth': 5, 'min_child_weight': 2}
```

Here, I have run 48 combinations with wide intervals between values. The ideal values are 5 for max_depth and 2 for min_child_weight.


* Step 3.Tune gamma  
Then I tune gamma value using the parameters already tuned above. Gamma can take various values but I only check for 5 values here.

```python
cv_params = {'gamma':[i/10.0 for i in range(0,5)]}
```
Result:
best value for the parameter：{'gamma': 0.0}  
This shows that our original value of gamma, i.e. 0 is the optimum one. 

* Step 4.Tune subsample and colsamle_bytree  
The next step would be try different subsample and colsample_bytree values. Lets do this in 2 stages as well and take values 0.6,0.7,0.8,0.9 for both to start with.

```python
cv_params = {'subsample':[i/10.0 for i in range(6,10)],
             'colsample_bytree':[i/10.0 for i in range(6,10)]}
```  
Here, I find 0.8 as the optimum value for both subsample and colsample_bytree. Then I'll try values in 0.05 interval around these.

```python
cv_params = {'subsample':[i/100.0 for i in range(75,90,5)],
             'colsample_bytree':[i/100.0 for i in range(75,90,5)]}
```
Result:
best value for the parameter：{'colsample_bytree': 0.8, 'subsample': 0.8}

Again get the same values as before. Thus the optimum values are 0.8 for both parameters.

* Step 5.Tune regularization parameters
Next step is to apply regularization to reduce overfitting. 

```python
cv_params = {'reg_alpha':[0,0.1,0.6,1,100],'reg_lambda':[0.1,0.3,0.6,1,100]}
```
Result:
best value for the parameter：{'reg_alpha': 0.6, 'reg_lambda': 0.3}

Then fix alpha and try more values for lambda. 

```
cv_params = {'reg_lambda':[0.2,0.3,0.4,0.5]}
```
Result:
best value for the parameter：{'reg_lambda': 0.3}

Thus the optimum values are 0.6 for reg_alpha and 0.3 for reg_lambda.

* Step 6.Reduce learning rate
Lastly, lower the learning rate and add more trees. Therefore, I use the cv function to do the job again.

```python
cv_params = {'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.3]}
```
Result:
best value for the parameter：{'learning_rate': 0.01}

Then the final step is to re-calibrate the number of boosting rounds for the updated parameters by using early-stopping method.

```python
params = {'learning_rate': 0.01, 'max_depth': 5, 
          'min_child_weight': 2, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'gamma': 0, 'reg_alpha': 0.6, 'reg_lambda': 0.3, 'eval_metric':'rmse'}
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
rgs = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
verbose_eval=10)
```

Result:
Stopping. Best iteration:
[850]	train-rmse:0.67492	valid-rmse:0.88371

test_rmse = 0.90912

## Result:
best value for the parameter：{'colsample_bytree': 0.8, 'subsample': 0.8}






  


## END

During the process of data analysis and preprocessing, I find that doing data analysis is of vital importance to the understanding of the data, which is the cornerstone of the building of the model. I find that the data preprocessing can improve the performance of the model as the raw data can be quite confusing for the model to train. 

I then focus on the parameters tuning for XGBoost algorithm and improve future sales prediction model based on them. Comparing with the model improvement by data preprocessing, parameter tuning does not feed back significant benefits which means parameter tuning can improve the performance of model but not a big leap. Besides, more efforts should be focus on feature engineering or other ways. For instance, it would be better if I crawl more specific features for the stores outside the dataset from Kaggle, like the weather information of the stores' location in the certain month. And I would like to do more work on this part in the future.
