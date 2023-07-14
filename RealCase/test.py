import numpy as np
import pandas as pd
from util import load_data, get_mt_matrix, cal_C_max, FJS_GA, global_initialization

# data_path = "E:\Task\Intelligent Technology in Manufacturing System\Work3\代码\RealCase\实际案例 - 零件特征及加工时间.xlsx"
# order_data, workshop_data = load_data(data_path)

# machines, times = get_mt_matrix(order_data,workshop_data)

# population = global_initialization(N=50,machines=machines, times=times)

# print(population.shape)
# print(population[1])


# # 创建一个示例数组
# arr = np.array([1, 2, 3, 2, 4, 5, 1, 6, 2])

# # 要删除的第一个重复值
# value_to_remove = 2

# # 找到第一个重复值的索引
# first_duplicate_index = np.where(arr == value_to_remove)[0][0]

# # 创建布尔索引
# mask = np.ones_like(arr, dtype=bool)
# mask[first_duplicate_index] = False  # 将第一个重复值的索引设为 False

# # 使用布尔索引来获取删除第一个重复值后的新数组
# new_arr = np.delete(arr, first_duplicate_index)

# print(new_arr)
# m = 0
# a = np.tanh(np.sqrt(m))

# 创建一个示例数组
arr = np.array([1, 2, 3, 2, 3, 4, 5, 1, 2])

# 要删除的特定值
value_to_remove = 2

# 找到第一个特定值的索引
index_to_remove = np.argmax(arr == value_to_remove)

# 使用布尔索引来获取不包含特定值的新数组
new_arr = np.delete(arr, index_to_remove)

print(index_to_remove)