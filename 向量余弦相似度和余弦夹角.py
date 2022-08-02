import numpy as np

vec1 = np.array([1,0,0])
vec2 = np.array([-1,0,0])
dist1 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print("余弦距离(相似度)为：\t" + str(dist1))

import numpy as np
vector_1 = np.array([2,0,0])
vector_2 = np.array([0,-2,0])

unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
dot_product = np.dot(unit_vector_1, unit_vector_2)
radian = np.arccos(dot_product)#弧度3.14是180度,1.57是90度
print('弧度：'+str(radian))
angle = np.degrees(radian)#角度
print('角度：'+str(angle))



############################################################################################
#从csv里边读取数据
#csv数据如下：
#峰号	20220621混合水样1	20220621企业A	20220621企业B	测试的
#78	572568229.9	173488.87	606972090.5	1
#111	132512959.7	135497442.1	700767.818	2
#74	124972225.6			3
#225	120353205.8	97973574.57		4
#26	105057332.9		73714195.12	5
#101	75818854.51	76144029.81	916998.952	6

import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv(r'溯源结果.csv', encoding='utf-8', header=0)
df.fillna(value=0, inplace=True)

# 归一化
'''
min_max_scaler = preprocessing.MinMaxScaler()
df[:] = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df)
'''

df_compare_data = df.iloc[:, 2:]  # df.iloc[:, 2:].copy()避免修改df1时影响df,获取第三列之后的列
df_standard = df.iloc[:, 1]  # 获取混合水样,df.iloc[:, 1]得到的是series,df.iloc[:, 1:2]得到的是dataframe
np_array_standard = np.array(df_standard)
vector_standard = np_array_standard / np.linalg.norm(np_array_standard)
company_name_and_cosine_angle = []  # 企业名称、余弦角度
for (colname, colval) in df_compare_data.iteritems():
    # print(colname, colval.values)
    np_array_this = np.array(colval)
    vector_this = np_array_this / np.linalg.norm(np_array_this)
    dot_vector = np.dot(vector_standard, vector_this)
    radian = np.arccos(dot_vector)  # 弧度3.14是180度,1.57是90度
    # print('弧度：' + str(radian))
    angle = np.degrees(radian)  # 角度
    print('列名：' + colname, ',余弦角度：' + str(angle),'余弦值：' + str(dot_vector))
    company_name_and_cosine_angle.append([colname, angle])

result_df = pd.DataFrame(company_name_and_cosine_angle, columns=['企业名称', '余弦角度'])
result_df = result_df.sort_values(by=['余弦角度'], ascending=[True])
result_df.to_csv('结果.csv', index=False, mode='w', header=True, encoding='utf-8')

