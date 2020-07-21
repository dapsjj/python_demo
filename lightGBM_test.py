import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


train = pd.read_csv(r'D:/for_garbage_train.csv',encoding='utf-8')
test = pd.read_csv(r'D:/for_garbage_test.csv',encoding='utf-8')
X_train = train.iloc[:,:-1]
Y_train = train['Y']
light = LGBMRegressor()
light.fit(X_train,Y_train)
Y_test = light.predict(test,num_iteration=light.best_iteration_)
print(Y_test)
print('Feature importances:', list(light.feature_importances_))

estimator = LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10, 20, 30, 40, 50]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, Y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
