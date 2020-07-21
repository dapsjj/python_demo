import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import joblib


train = pd.read_csv(r'D:/for_garbage_train.csv',encoding='utf-8')#训练集
test = pd.read_csv(r'D:/for_garbage_test.csv',encoding='utf-8')#需要预测的数据
val = pd.read_csv(r'D:/for_garbage_val.csv',encoding='utf-8')#验证集
X_train = train.iloc[:,:-1]
X_val = val.iloc[:,:-1]
Y_train = train['Y']
Y_val = val['Y']
light = LGBMRegressor()
light.fit(X_train,Y_train,eval_set=[(X_val, Y_val)],eval_metric='l1',early_stopping_rounds=20)
# save model
joblib.dump(light, 'light.pkl')
# load model
lightgbm_pickle = joblib.load('light.pkl')
Y_test = lightgbm_pickle.predict(test,num_iteration=lightgbm_pickle.best_iteration_)
print(Y_test)
print('Feature importances:', list(lightgbm_pickle.feature_importances_))

estimator = LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10, 20, 30, 40, 50]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, Y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
