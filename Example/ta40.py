import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.getcwd(),"util"))

from init import load_data,cal_Cmax
from GA import JS_GA

data_path = "E:\\Task\\Intelligent Technology in Manufacturing System\\Work3\\算例\\ta40.xlsx"#文件的位置要修改一下

times, machines = load_data(data_path)

best_C_max, best_sol, machine_start_time, machine_end_time, C_max_list = JS_GA(func=cal_Cmax,machines=machines,times=times,Max_iter=1000,dynamic_change=False,TLBO=False)  #记得改这里，要修改次数
best_sol, machine_start_time, machine_end_time, C_max_list = pd.DataFrame(best_sol), pd.DataFrame(machine_start_time), pd.DataFrame(machine_end_time), pd.DataFrame(C_max_list)

writer = pd.ExcelWriter('Solution_ta40_1000.xlsx')#文件的名字也要改下最好把运行次数也加到文件里去
best_sol.to_excel(writer,sheet_name="best_sol")
machine_start_time.to_excel(writer,sheet_name="machine_start_time")
machine_end_time.to_excel(writer,sheet_name="machine_end_time")
C_max_list.to_excel(writer,sheet_name="C_max")
writer.save()
writer.close()

print("请用你可爱的小手，记一下下面这个值")
print(best_C_max)




