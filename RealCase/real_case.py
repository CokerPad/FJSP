import numpy as np
import pandas as pd
from util import load_data, get_mt_matrix, cal_C_max, FJS_GA

data_path = "E:\Task\Intelligent Technology in Manufacturing System\Work3\代码\RealCase\实际案例 - 零件特征及加工时间.xlsx"
order_data, workshop_data = load_data(data_path)

machines, times = get_mt_matrix(order_data,workshop_data)

best_C_max, best_stat_df, C_max_list = FJS_GA(func=cal_C_max,machines=machines,times=times,Max_iter=1000,global_init=False,dynamic_change=False,TLBO=False)
C_max_list = pd.DataFrame(C_max_list)
writer = pd.ExcelWriter('Solution_RealCase_1000.xlsx')#文件的名字也要改下最好把运行次数也加到文件里去
best_stat_df.to_excel(writer,sheet_name="best_sol")
C_max_list.to_excel(writer,sheet_name="C_max")
writer.save()
writer.close()

print(f"最优值为{best_C_max}")


