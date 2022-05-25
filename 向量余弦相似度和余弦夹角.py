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
