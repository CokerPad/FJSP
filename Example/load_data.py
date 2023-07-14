# %% [markdown]
# 导入数据

# %%
import pandas as pd
import numpy as np
import os

# %% [markdown]
# 导入原始数据

# %%
data_path = "实际案例 - 零件特征及加工时间.xlsx"
order_data = pd.read_excel(data_path,sheet_name="订单信息")
workshop_data = pd.read_excel(data_path,sheet_name="车间信息")

# %% [markdown]
# 首先对车间信息表进行处理，即将列名称换成序号、ID以及描述

# %%
row_name = workshop_data.loc[0,:].tolist()
workshop_data = workshop_data.drop(axis=0,index=0)
workshop_data.columns = row_name
workshop_data.set_index('ID',inplace=True)


# %% [markdown]
# 接下来对订单信息进行修改，将订单信息分割成加工时间以及加工机器两个信息表

# %%
workshop_order_row_name = ["零件","数量","特征","工序"]
workshop_order_row_name = workshop_order_row_name + [s.strip() for s in order_data.iloc[0,4:].tolist()]
order_data = order_data.drop(axis=0,index=0)
order_data.columns = workshop_order_row_name
order_data.set_index('零件',inplace=True)

# %% [markdown]
# 得到零件的名字

# %%
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# %%
parts_name = order_data.index.dropna().to_list()
parts_name = remove_duplicates(parts_name)

# %% [markdown]
# 得到零件加工的数量

# %%
parts_num = order_data.loc[:,"数量"].dropna().to_numpy()

# %%
machines = np.zeros((21,18),dtype='object')
times = np.zeros((21,18),dtype='object')
for i in range(len(parts_name)):
    parts_data = order_data.loc[parts_name[i],:]
    parts_data.set_index("工序",inplace=True)
    num_process = order_data.loc[parts_name[i],"工序"].tolist()
    for j in range(len(num_process)):
        process_parts_data = parts_data.loc[num_process[j],"C1":"QG1"].dropna()
        machines[i,j] = process_parts_data.index.to_numpy()
        times[i,j] = process_parts_data.values





# %%
workshop_data.head()
mapping_dict = dict(zip(workshop_data.index.tolist(),workshop_data.loc[:,"序号"].to_list()))

# %%
def map_function(element):
    return mapping_dict.get(element, element)

mapped_machines = np.vectorize(map_function)
for i in range(len(parts_name)):
    for j in range(len(num_process)):
        machines[i,j] = mapped_machines(machines[i,j])


