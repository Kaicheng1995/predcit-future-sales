# Predict Future Sales ⏰

> In this project I work with a time series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - [1C Company](http://www.1c.com/)🇷🇺. The goal for this project is to predict total sales for every product and store in the next month.

* [Kaggle Link](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview):

[![alt text](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/kaggle.png "title")](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

```diff
! Please check out more details and data visualization at [prediction.ipynb], especially for Feature Engineering Part.
```

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
Since it's a time series quetsion, I include lag features to help prediction. When doing label encoding, it has some shortcomings. Since we have tem-million observations, the value of observations that have been encoded could have passive influences on the prediction result because they have real number meanings. To deal with the wide-scattered observations, we can process them with mean encoding.

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



## END

During the process of data analysis and preprocessing, I find that doing data analysis is of vital importance to the understanding of the data, which is the cornerstone of the building of the model. I find that the data preprocessing can improve the performance of the model as the raw data can be quite confusing for the model to train. 

I then focus on the parameters tuning for XGBoost algorithm and improve future sales prediction model based on them. Comparing with the model improvement by data preprocessing, parameter tuning does not feed back significant benefits which means parameter tuning can improve the performance of model but not a big leap. Besides, more efforts should be focus on feature engineering or other ways. For instance, it would be better if I crawl more specific features for the stores outside the dataset from Kaggle, like the weather information of the stores' location in the certain month. And I would like to do more work on this part in the future.
