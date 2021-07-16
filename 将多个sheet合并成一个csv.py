import pandas as pd
iris = pd.read_excel(r'2015.xlsx',None)
keys = list(iris.keys())
iris_concat = pd.DataFrame()
for i in keys:
    iris1 = iris[i]
    iris_concat = pd.concat([iris_concat,iris1])
iris_concat.to_csv('./2015.csv',index=0)
