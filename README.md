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
- lag features
- 增加mean encoding features
- 增加price trend features
- 增加the resident or new item features
- 增加year、month

Before doing this part, converted daily sales to monthly sales, and merge all datasets I've modified above.
![](https://github.com/Kaicheng1995/predict-future-sales/blob/master/img/all.png)

### Lag features


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
### Mean encoding

不管是哪种机器学习模型，它都需要将categorical特征（str）转化为数字（int or float），这个过程称为编码。Mean encoding就是常用的编码技术，和它对应的有one-hot encoding、label encoding。

‘shop_id'，'item_id'，'city_code'，'shop_type_code‘，'type_code'等都是label code，例如总共有50个city，就用[0, 49]给每个city一个编号。[【Predict Future Sales】用深度学习玩转销量预测](https://www.jianshu.com/p/f0d34d1952f0)已经聊过label code和one-hot code的弊端，即code之间无法建立关联，而mean encoding则在一定程度上解决了这个问题。

![Figure 3](https://upload-images.jianshu.io/upload_images/13575947-2edf00a00641aedf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 3是我从数据集中随机摘取的一个片段，从图中可以看到，相同'item_id'的'item_cnt_month'是随机分布的，通常情况下，数据的随机性越大，数据的误差就越大，不利于机器学习算法。

![Figure 4](https://upload-images.jianshu.io/upload_images/13575947-b4263b9a82bef121.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 4中的'item_id_code'字段就是item_id的mean encoding，即将所有相同item_id的item_cnt_month相加求和，再除以item_id出现的频率。可以看到，item_id_code的图形比Figure 3的随机性要小得多，有利于算法利用item_id这个特征来区分样本。

```
g = df.groupby(by='item_id').item_cnt_month.agg({'item_cnt_month': 'mean'})
g.columns = ['item_id_code']
df2 = df.merge(g.reset_index(), on='item_id', how='left')
```

实际上，item_id_code既有item_id的频率特征也包含了因变量（item_cnt_month）与item_id的关联。然而，正因为它囊括了因变量的部分数据，因此有data leakage的可能，但这个风险在time series数据集中并不存在，毕竟数据集的特征都是从过去的time中获取的。

### Price trend features

前面已经分析过，销量和售价呈反比，因此我们为每个样本增加的售价涨跌趋势特征（前一个月的）。

### Revenue trend features

我们已经知道validation set和test set都会出现train set所没有的新商品，借助商店revenue趋势特征可以帮助算法预测新商品在现有店铺的销量。

### The resident or new item features

'item_shop_last_sale'和'item_last_sale'用以纪录距离最近一次销售之间隔了几个月，通过它可以和之前月份的数据建立关联。'item_shop_first_sale'和'item_first_sale'则是用于表示新商品的特征。

---

## Trainning

和前面的流程一样，通过proc_df()将category数据数字化、填充NaN数据。用最后两个月的数据分别构建validation set和test set。

```
trn_idxs = df[df.date_block_num < 22].index.values
val_idxs = df[df.date_block_num == 22].index.values
test_idxs = df[df.date_block_num == 23].index.values
trn_x, trn_y = df.loc[trn_idxs].copy(), y[trn_idxs].copy()
val_x, val_y = df.loc[val_idxs].copy(), y[val_idxs].copy()
test_x = df.loc[test_idxs].copy()
```

如果你的机器算力有限，或是想要减少模型训练的时间，可以通过set_rf_samples()来设置每颗决策树最大的随机采样数。当然如果条件允许，用完整的数据集训练效果更好。

```
# set_rf_samples(50000)
# len(trn_x), 50000 / len(trn_x)

# (6115195, 0.008176354147332997)
```

```
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)
%time m.fit(trn_x, trn_y)
show_results(m)

[0.5325591159079617,
 0.3752106515417507,
 0.5227038231220662,
 0.8979843084539196]
```

我用完整的数据集训练的结果：valid set的RMSE为0.898。相比“Look at Data Quickly”时的模型有了很大的进步。另外，模型超参也增加几个新面孔：
- **n_estimators**:  决策树的数量，一般来说决策树越多模型就越准确。Forest，指的就是由大量决策树共同作用，这也称为Model Ensembling。
- **min_samples_leaf**:  可以把它理解为模型在做出预测前所要经过的决策次数，**min_samples_leaf**的值越大表示决策次数越少，过少的决策会降低模型预测准确率，但过多的决策又容易overfitting。常用参数为：1、3、5、10。
- **max_features**:  模型做决策的过程就是在决策树上建立分叉的过程，每个分叉都是对一个特征做判断的结果，如Figure 5所示，RF为了增加随机性，每个分叉所选取的特征都是来自于一个随机的特征子集，而**max_features**可以控制特征子集的建立。max_features=0.5，指的是每次决策时按1/2的比例随机划分待决策的特征为两个子集。常用的参数为：0.5、log2、sqrt。

![Figure 5](https://upload-images.jianshu.io/upload_images/13575947-ddfd974d03028a36.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Feature importance

从Figure 5可以看出，RF是通过判断特征来做预测的，那最好的情况是，所选取的所有特征都是核心特征。例如，要判断样本是猫还是狗，那么，“有没有毛”这个特征就属于没用的特征，而“耳朵下垂”这个特征就可以区分猫和狗。Feature importance就是用来判断特征有用与否的指标。

```
fi = rf_feat_importance(m, trn_x)
fi.plot('cols', 'imp', 'barh', figsize=(10, 8))
```
![](https://upload-images.jianshu.io/upload_images/13575947-907d0cc00a5f2433.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到数据集中很多特征的重要性接近于0，因此接下来我们只选取feature importance大于0.004的特征。

```
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=4, max_features=0.5, oob_score=True)
%time m.fit(trn_x, trn_y)
show_results(m)

[0.5283864898099666,
 0.37657590348293346,
 0.5200241416189362,
 0.8970026613695433]
```
重新训练后，可以看到score所有提升。

接着我经过调整'min_samples_leaf'和'max_features'，我发现min_samples_leaf=3，max_features=0.5的超参效果最佳，并将决策树增加到120后重新训练。

```
m = RandomForestRegressor(n_estimators=120, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)
%time m.fit(trn_x, trn_y)
show_results(m)

[0.5384021046588424,
 0.38571421105968695,
 0.5286384392316963,
 0.8904041458506617]
```

可以看到，选取更多的决策树效果明显，但训练时间也更长。为了减少训练时间，我把决策数增加到400的同时将采样数限制在50万。

```
set_rf_samples(500000)
m = RandomForestRegressor(n_estimators=400, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)
%time m.fit(trn_x, trn_y)
show_results(m)

[0.6248771505427018,
 0.40001388910585356,
 0.5843704869177022,
 0.8799794652507853]
```

最终RMSE是0.8799，在LB上排名16，对于没有完整训练的模型来说还是不错的。

### Submission

```
yp = m.predict(test_x)
test  = pd.read_csv(f'{PATH}/test.csv').set_index('ID')
sub = pd.DataFrame({'ID': test.index.values, 'item_cnt_month': yp})
sub.to_csv('submission.csv', index=False)
```

在预测test set之前，你最好还是把valid set重新加入train set，并重新训练，我这里是为了节省时间所以没有重新训练模型。我在LB上的score比validation set's score要差一些，这其实也属于正常情况，毕竟我也只是训练了部分数据，而且正如我在[【Predict Future Sales】用深度学习玩转销量预测](https://www.jianshu.com/p/f0d34d1952f0) 所说的那样，Public LB的可信度并不比validation set's score更高。

另外，你如果有兴趣，也可以用其他模型试试，如xgboost、LightGBM以及前作介绍的[深度神经网络](https://www.jianshu.com/p/f0d34d1952f0)，或者把它们的结果ensembling起来，这样预测的效果可能会更好。

## END

总结一下，本文主要介绍了Random Forest的用法，以及时间序列数据集的特征工程方法。通过这个案例，你应该可以感觉到，对于真实世界的问题，尤其是time series、tabular data等问题，通过特征工程挖掘数据特征以及特征间的关联，比用哪个算法要重要得多得多得多。

## Refences
