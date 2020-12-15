import pandas as pd
import catboost
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

train_data = pd.read_csv('zhengqi_train.txt',sep='\t')
test_data = pd.read_csv('zhengqi_test.txt',sep='\t')

# 通过观察，删除分布差异较大的属性
drop_columns = ['V5','V11','V13','V14','V17','V19','V20', 'V21', 'V22', 'V27','V35']
train_data.drop(columns=drop_columns,inplace=True)
test_data.drop(columns=drop_columns,inplace=True)

# 数据分割
X = train_data.iloc[:,:-1]
y = train_data['target']

model = catboost.CatBoostRegressor(
                           loss_function="RMSE",
                           eval_metric="RMSE",
                           task_type="GPU",
                           learning_rate=0.005,
                           iterations=12000,
                           random_seed=42,
                           od_type="Iter",
                           depth=7,
                           early_stopping_rounds=50,
                           l2_leaf_reg=3
                          )
rfecv = RFECV(estimator = model,
              cv = KFold(5),
              scoring = 'neg_mean_squared_error')
rfecv.fit(X, y)
df = pd.DataFrame(rfecv.predict(test_data))
df.to_csv("my.txt", index=False, header=False)

