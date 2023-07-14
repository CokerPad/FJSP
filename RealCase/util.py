import numpy as np
import pandas as pd
from tqdm import tqdm
import copy



def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



#加载数据
def load_data(data_path):
    order_data = pd.read_excel(data_path,sheet_name="订单信息")
    workshop_data = pd.read_excel(data_path,sheet_name="车间信息") #这里对原文件调整了位置
    return order_data, workshop_data

 
#返回的数据形式是单行,总共476列，每列表示每个工件的对应工序可用的机器或需要的时间，如工件1占用前10列，工件2占11-28列
#返回的total_machines,total_times矩阵是每个工件每个工序(总共476行，其中计算了要加工多个的零件)的机器或者时间。
#且其中total_machines返回的机器索引从0开始
def get_mt_matrix(order_data,workshop_data):
    

    #首先对车间信息表进行处理，即将列名称换成序号、ID以及描述
    workshop_row_name = workshop_data.loc[0,:].tolist()
    workshop_data = workshop_data.drop(axis=0,index=0)
    workshop_data.columns = workshop_row_name
    workshop_data.set_index('ID',inplace=True)

    #接下来对订单信息进行修改，将订单信息分割成加工时间以及加工机器两个信息表
    order_row_name = ["零件","数量","特征","工序"]
    order_row_name = order_row_name + [s.strip() for s in order_data.iloc[0,4:].tolist()]
    
    order_data = order_data.drop(axis=0,index=0)
    order_data.columns = order_row_name
    order_data.set_index('零件',inplace=True)
    

    #得到零件的名字
    parts_name = order_data.index.dropna().to_list()
    parts_name = remove_duplicates(parts_name)
    #得到零件加工的数量
    parts_num = order_data.loc[:,"数量"].dropna().to_numpy()

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

    mapping_dict = dict(zip(workshop_data.index.tolist(),(workshop_data.loc[:,"序号"]-1).to_list()))

    def map_function(element):
        return mapping_dict.get(element, element)
    
    mapped_machines = np.vectorize(map_function)
    for i in range(len(parts_name)):
        for j in range(18):
            machines[i,j] = mapped_machines(machines[i,j])

    total_machines = np.zeros((56,18),dtype='object')
    total_times = np.zeros((56,18),dtype='object')
    sum_num = int(0)
    for i in range(len(parts_num)):
        for j in range(int(parts_num[i])):
            total_machines[int(sum_num+j)] = machines[i]
            total_times[int(sum_num+j)] = times[i]
        sum_num += parts_num[i]

    total_machines = total_machines.reshape(-1)
    total_times = total_times.reshape(-1)
    total_machines.squeeze()
    total_times.squeeze()

    delete_index = []
    for i in range(1008):
        if np.any(total_machines[i]) == False:
            delete_index.append(i)

    delete_index = np.array(delete_index,dtype=np.int16)
    total_machines = np.delete(total_machines,delete_index)
    total_times = np.delete(total_times,delete_index)

    return total_machines, total_times  


def get_sum_list():
    #得到总共56个零件所对应的工序数最开始的索引 即第一个零件是从0开始，第二个零件是从10开始
    parts_process_num = np.array([10,18,7,10,9,11,9,7,8,9,9,8,4,5,4,7,5,7,7,8,10])  #每类零件的工序数
    parts_num = np.array([4,4,8,2,2,2,2,2,4,4,2,2,2,2,2,2,2,2,2,2,2])  #每类零件的数量
    sum_process_num_list = []  #累积索引列表
    sum_process_num = 0
    for i in range(parts_num.shape[0]):
        for j in range(parts_num[i]):
            sum_process_num_list.append(sum_process_num)
            sum_process_num += parts_process_num[i]

    return sum_process_num_list


#将加工顺序编码解码成(total_process_num，3)的矩阵，3个元素分别表示加工机器、零件编号以及工序
def decoder(chromosome,machines):
    #chromosome代表每个加工顺序编码

    total_process_num = int(machines.shape[0])
    schedule = np.zeros((total_process_num,3))

    #得到总共56个零件所对应的工序数最开始的索引
    sum_process_num_list = get_sum_list()

    #零件计数步骤矩阵--每个元素表示该零件需要安排到了第几步 如第一个位置的0表示 需要安排第0步
    count_process = np.zeros(56,dtype=np.int16)
    
    
    for i in range(total_process_num):
        #零件的索引
        part_index = chromosome[total_process_num+i]
        #机器的索引

        machine_index = count_process[part_index] + sum_process_num_list[part_index]
        
        schedule[i,0] = chromosome[machine_index]
        schedule[i,1] = part_index
        schedule[i,2] = count_process[part_index]
        count_process[part_index] += 1

    return schedule


def cal_C_max(encoder,machines,times):
    schedule = decoder(encoder,machines)
    #计算C_max
    total_process_num = int(machines.shape[0])
    total_machines = 20
    schedule = np.array(schedule,dtype=np.int16)
    #初始化机器开始加工时间、结束时间、工序、开始加工时间以及结束时间的DataFrame
    stat_dict = {"设备编号":[schedule[i,0] for i in range(total_process_num)],
                 "零件编号":[schedule[i,1] for i in range(total_process_num)],
                 "工序编号":[schedule[i,2] for i in range(total_process_num)],
                 "开始时间":[0 for i in range(total_process_num)],
                 "结束时间":[0 for i in range(total_process_num)]}
    stat_df = pd.DataFrame(stat_dict)


    #得到总共56个零件所对应的工序数最开始的索引
    sum_process_num_list = get_sum_list()

    #机器上一道工序的索引  减1是为了避免第一个进行加工的机器 在下次加工时被判断为第一次加工
    previous_works = np.zeros(total_machines,dtype=np.int16) - 1
    #每个工件的上一个工序的在总列表里的索引
    previous_index = np.zeros(56,dtype=np.int16)

    for j in range(total_process_num):
        machine, part, process = schedule[j]
        machine, part, process = int(machine), int(part), int(process)

        #机器是否是第一次加工的标志  -1表示未分配工序，因此是未加工
        machine_flag = (previous_works[machine]==-1)

        #是否是第一个工序的标志
        process_flag = (process==0)
        

        #process_index为该工序在总编码表里的索引,如234/475 从0开始的索引
        process_index = sum_process_num_list[part] + process

        #machine_index为该工序在所有能加工的机器的顺序的索引，比如有4台机器(0,1,3,14)那么第14台机器的索引就是3
        machine_index = np.where(machines[process_index] == machine)

        if machine_flag:  #表明机器是第一次加工
            if process_flag: #表明是第一个工序
               
                stat_df.loc[j,"开始时间"] = 0
                stat_df.loc[j,"结束时间"] = times[process_index][machine_index]
            
            else:  #表明不是第一个工序，但是机器第一次加工
                #上一道工序的索引
                past_index = previous_index[part]
                past_machine_time = stat_df.loc[past_index,"结束时间"]
                stat_df.loc[j,"开始时间"] = past_machine_time
                stat_df.loc[j,"结束时间"] = stat_df.loc[j,"开始时间"] + times[process_index][machine_index]
        else:
            #机器上一次加工在表格里的索引
            past_machine_index = previous_works[machine]
            max_machine_time = stat_df.loc[past_machine_index,"结束时间"]

            if process_flag:
                stat_df.loc[j,"开始时间"] = max_machine_time
                stat_df.loc[j,"结束时间"] = stat_df.loc[j,"开始时间"] + times[process_index][machine_index]

            else:

                #上一道工序的索引
                past_index = previous_index[part]
                past_machine_time = stat_df.loc[past_index,"结束时间"]

                machine_process_time = [max_machine_time,past_machine_time]
                stat_df.loc[j,"开始时间"] = max(machine_process_time)
                stat_df.loc[j,"结束时间"] = stat_df.loc[j,"开始时间"] + times[process_index][machine_index]
        
        previous_works[machine] = j
        previous_index[part] = j

    C_max = stat_df["结束时间"].max()
    return C_max, stat_df


#计算种群的适应度
def cal_fitness(func,population,machines,times):
    C_max = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        C_max[i],_ = func(population[i],machines,times)
    fitness = np.max(C_max) - C_max
    return fitness, np.min(C_max)  

#得到所有工序的集合
def get_all_process():
    parts_process_num = np.array([10,18,7,10,9,11,9,7,8,9,9,8,4,5,4,7,5,7,7,8,10])  #每类零件的工序数
    parts_num = np.array([4,4,8,2,2,2,2,2,4,4,2,2,2,2,2,2,2,2,2,2,2])  #每类零件的数量

    all_process = [] #表示所有工序的列表，如0 0 0..56 56，其中同一个数字出现次数为工序数
    part_index = 0

    for i in range(21):
        for j in range(parts_num[i]):
            for k in range(parts_process_num[i]):
                all_process.append(part_index)
            part_index += 1
    
    all_process = np.array(all_process,dtype=np.int16)
    return all_process


#全局选择生产初始种群
#整体思路是随机生产工序排列表，依次遍历工序表，选择其可以作为加工的机器里用时最短的机器

def global_initialization(N,machines,times):

    all_process = get_all_process()
    total_process_num = int(machines.shape[0])

    #得到总共56个零件所对应的工序数最开始的索引  如第一个零件最开始是0，第二个零件最开始是10
    sum_process_num_list = get_sum_list()
    

    population = np.zeros((N,2*total_process_num),dtype=np.int16)
    
    for i in range(N):
        
        population[i,total_process_num:] = np.random.choice(all_process,total_process_num,replace=False)

        #零件计数步骤矩阵--每个元素表示该零件需要安排到了第几步 如第一个位置的0表示 需要安排第0步
        count_process = np.zeros(56,dtype=np.int16)
        machines_time = np.zeros(20,dtype=np.int16) #代表所有机器当前加工时间的集合

        for j in range(total_process_num):
            #得到零件的索引
            part_index = population[i,total_process_num+j]

            #通过该零件最开始工序的索引加上当前工序的索引，得到该工序在476道工序中的索引
            process_index = sum_process_num_list[part_index] + count_process[part_index]

            #得到所有可用的机器索引
            available_machines_index = machines[process_index]  #是0,1,2,...,20这些数

            if available_machines_index.shape[0] > 1:
                #得到所有可用机器的加工当前工序所需时间
                need_times = np.array(times[process_index],dtype=np.int16) + machines_time[available_machines_index]
                #得到用时最少的机器在所有可用机器里的索引
                least_times_machine_index = np.argmin(need_times)
                #得到最短时间
                least_time = times[process_index][least_times_machine_index]
                #得到当前步骤的最优解
                local_best_machine = available_machines_index[least_times_machine_index]

                #将最优机器填入该个体编码中
                population[i,process_index] = local_best_machine
            
                #更新机器所用时间表
                machines_time[local_best_machine] += least_time
            
            #这是只有一台可用机器的情况，直接计算
            else:
                only_machine = available_machines_index[0]
                population[i,process_index] = only_machine
                machines_time[only_machine] += int(times[process_index][0])

        
            #更新工序计数矩阵
            count_process[part_index] += 1

    return np.array(population,dtype=np.int16)
            


def wrong_crossover(MP,total_process_num,crossover_rate,machines):
    cross_index = np.random.choice(MP.shape[0],MP.shape[0],replace=False)
    cross_index = cross_index.reshape(-1,2)  #两两配对得到进行交叉的索引

    for j in range(cross_index.shape[0]):  #需要进行交叉的最大次数
        c_r = np.random.rand()
        if c_r <= crossover_rate:
            cross_position = np.random.choice(2,2*total_process_num)#得到交换位置的0-1编码  
            #这里1代表需要进行交叉的位置，0代表无需交叉的位置，这样处理是为了方便统计需要交叉的位置个数

            #对前半段进行使用机器的交叉
            individual1, individual2 = cross_index[j]
            for k in range(total_process_num):
                if cross_position[k] == 1:
                    MP[individual1,k] = np.random.choice(machines[k],1)[0]
                    MP[individual2,k] = np.random.choice(machines[k],1)[0]

            #对后半段进行加工顺序的交叉
            #首先将前半段的要交叉的位置全部设置为0，方便后半段计算交叉数量以及交叉位置
            cross_position[:total_process_num] = 0

            num_cross = np.sum(cross_position)  #需要进行交叉的数量
            cross_position_later = (cross_position==1)
            MP[individual1,cross_position_later] = np.random.choice(MP[individual1,cross_position_later],num_cross,replace=False)
            MP[individual2,cross_position_later] = np.random.choice(MP[individual2,cross_position_later],num_cross,replace=False)
        else:
            pass

def delete_code(code,need_delete_code_list):
    for i in range(need_delete_code_list.shape[0]):
        temp_need_delete_index = np.argmax(code == need_delete_code_list[i])  #需要删除工件号在
        code = np.delete(code, temp_need_delete_index)
    return code


def true_crossover(MP,total_process_num,crossover_rate):
    cross_index = np.random.choice(MP.shape[0],MP.shape[0],replace=False)
    cross_index = cross_index.reshape(-1,2)  #两两配对得到进行交叉的索引

    for j in range(cross_index.shape[0]):  #需要进行交叉的最大次数
        c_r = np.random.rand()
        if c_r <= crossover_rate:

            cross_index1, cross_index2 = cross_index[j]
            individual1, individual2 = copy.deepcopy(MP[cross_index1]), copy.deepcopy(MP[cross_index2])

            cross_position = np.random.choice(2,2*total_process_num)#得到交换位置的0-1编码  
            #这里1代表需要进行交叉的位置，0代表无需交叉的位置

            front_cross, back_cross = np.where(cross_position[:total_process_num] == 1),np.where(cross_position[total_process_num:] == 1)[0] + total_process_num
            back_uncross = np.where(cross_position[total_process_num:] == 0)[0] + total_process_num

            #对前半段进行使用机器的交叉
            work_selection1 = copy.deepcopy(individual1[front_cross])
            work_selection2 = copy.deepcopy(individual2[front_cross])
            individual1[front_cross] = work_selection2
            individual2[front_cross] = work_selection1


            delete_code1 = individual1[back_uncross]
            delete_code2 = individual2[back_uncross]

            need_code1 = delete_code(individual2[total_process_num:],delete_code1)
            need_code2 = delete_code(individual1[total_process_num:],delete_code2)
            
            individual1[back_cross] = need_code1
            individual2[back_cross] = need_code2

            MP[cross_index1] = individual1
            MP[cross_index2] = individual2

#GA算法
def FJS_GA(func,machines,times,Max_iter=1000,N=50,crossover_rate=0.75,mutation_rate=0.01,selection_index=0,generation_strategy=0,global_init=False,dynamic_change=False,TLBO=False):
    

    parts_process_num = np.array([10,18,7,10,9,11,9,7,8,9,9,8,4,5,4,7,5,7,7,8,10])  #每类零件的工序数
    parts_num = np.array([4,4,8,2,2,2,2,2,4,4,2,2,2,2,2,2,2,2,2,2,2])  #每类零件的数量
    total_process_num = int(machines.shape[0])
    
    all_process = get_all_process()
    
    #每次迭代的最优值记录列表
    C_max_list = np.zeros(Max_iter)
    C_max = np.zeros(N) #每一轮的C_max值
    
    #种群初始化
    if global_init:  #全局初始化
        population = global_initialization(N,machines,times)
    
    else:  #随机初始化
        population = np.zeros((N,2*total_process_num),dtype=np.int16)  
        #其中，前面total_process_num表示每道工序所选择的机器，后面total_process_num表示每道工序的工作顺序  从0开始索引
        for i in range(N):
            for j in range(total_process_num):
                population[i,j] = np.random.choice(machines[j],1)[0]
            population[i,total_process_num:] = np.random.choice(all_process,total_process_num,replace=False)

    unchange_iter = 0
    #得到总共56个零件所对应的工序数最开始的索引 即第一个零件是从0开始，第二个零件是从10开始
    sum_process_num_list = get_sum_list()

    crossover_rate0 = crossover_rate
    mutation_rate0 = mutation_rate

    for i in tqdm(range(Max_iter)):
        
        fitness, min_C_max = cal_fitness(func,population,machines,times)
        C_max_list[i] = min_C_max

        if i == 0:#初始化最优C_max
            last_best_C_max =  min_C_max

        #统计最优值不变的次数
        if last_best_C_max == min_C_max:
            unchange_iter += 1
        else:
            unchange_iter = 0

        #动态变化--将交叉率和变异率进行动态变化
        if dynamic_change:
            if i == 0:
                last_var = np.var(fitness)

            if i // 5 == 0:  #每隔5次进行一次修改
                var = np.var(fitness)
                var_ratio = var / last_var
                last_var = var
                #根据前后两次方差的比值进行交叉率与变异率的修改
                #当方差变大时，说明种群的聚集程度减小，此时就要减小变异率，增大交叉率，来提高种群的聚集程度
                #当方差变小时，说明种群聚集程度增大，陷入局部最优的可能性增大，此时就要减小交叉率，增大变异率来降低种群的聚集程度
                crossover_rate = crossover_rate0 + 0.15 * np.tanh(var_ratio) - 0.05 * np.tanh(np.sqrt(unchange_iter))
                mutation_rate = mutation_rate0 - 0.005 * np.tanh(var_ratio) + 0.0025 * np.tanh(np.sqrt(unchange_iter))
            
        #选择策略1：适应性比率选择法
        if selection_index==0:
            percent = (fitness+1e-10)/np.sum(fitness+1e-10)
            MP_index = np.random.choice(N,N,p=percent)
            MP = population[MP_index]
        #选择策略2：联赛选择法
        else:
            MP_index = np.argsort(fitness)[-2:][::-1]
            MP = population[MP_index*(N//2)]

        #TLBO交叉 仅对后半段进行交叉
        if TLBO:
            if i >=1000:
                MP_fitness, _ = cal_fitness(func,MP,machines,times)
                local_best_index = np.argmax(MP_fitness)
                
                need_change_list = list(range(MP.shape[0]))
                need_change_list.remove(local_best_index)#将最优样本从交配样本中删除
                for j in need_change_list:
                    local_best_machine = MP[local_best_index,:total_process_num]
                    local_best_process = MP[local_best_index,total_process_num:]
                    #需要进行交叉的个体 原编码
                    need_change_machine = MP[j,:total_process_num]
                    need_change_process = MP[j,total_process_num:] 

                    #每个工件已经排的工序数
                    count_process_list = np.zeros(56) 
                    
                    for k in range(total_process_num):
                        rate = np.random.rand()
                        
                        if rate > 0.5:#如果概率大于0.5则选择最优样本的工序安排
                            MP[j,total_process_num+k] = local_best_process[0]
                            need_deleted_part = local_best_process[0]
                            local_machine_index = sum_process_num_list[need_deleted_part] + count_process_list[need_deleted_part]
                            local_machine_index = int(local_machine_index)
                            MP[j,local_machine_index] = local_best_machine[local_machine_index]

                            local_best_process = np.delete(local_best_process, 0)
                            # 找到需要修改规划的指定工序的第一个索引
                            first_duplicate_index = np.argmax(need_change_process == need_deleted_part)
                            # 使用布尔索引来获取删除第一个重复值后的新数组
                            need_change_process = np.delete(need_change_process, first_duplicate_index)
                        
                        else:
                            MP[j,total_process_num+k] = need_change_process[0]
                            need_deleted_part = need_change_process[0]
                            local_machine_index = sum_process_num_list[need_deleted_part] + count_process_list[need_deleted_part]
                            local_machine_index = int(local_machine_index)
                            MP[j,local_machine_index] = need_change_machine[local_machine_index]

                            need_change_process = np.delete(need_change_process, 0)
                            # 找到最优规划的指定工序的第一个索引
                            first_duplicate_index = np.argmax(local_best_process == need_deleted_part)
                            # 使用布尔索引来获取删除第一个重复值后的新数组
                            local_best_process = np.delete(local_best_process, first_duplicate_index)
                            
                        count_process_list[need_deleted_part] += 1
            else:#普通交叉
                true_crossover(MP,total_process_num,crossover_rate)
                    

        else:#普通交叉
            true_crossover(MP,total_process_num,crossover_rate)
        
        #变异--这里的变异是针对每个个体的每个编码值都有可能变异
        for j in range(MP.shape[0]):
            for k in range(2*total_process_num):
                m_r = np.random.rand()
                if m_r <= mutation_rate:
                    #选择变异的方式 0--交换 1--插入 2--翻转
                    mutation_selection = np.random.randint(0,3,1)[0]

                    #交换
                    if mutation_selection == 0:
                        #针对前半段是在可以加工的机器里重新选择一台
                        if k<total_process_num:
                            MP[j,k] = np.random.choice(machines[k],1)[0]
                        #针对后半段是随机选择染色体上相邻的一对基因交换位置
                        else:
                            if k == 2*total_process_num-1:
                                MP[j,k],MP[j,total_process_num] = MP[j,total_process_num],MP[j,k]
                            else:
                                MP[j,k],MP[j,k+1] = MP[j,k+1],MP[j,k]
                    
                    #插入
                    elif mutation_selection == 1:
                        #仅对后半段进行处理，即随机选择染色体上的一个基因，插入到另一个随机位置
                        insert_index, insert_position = total_process_num + np.random.choice(total_process_num,2)
                        insert_element = MP[j,insert_index]
                        if insert_index < insert_position:
                            np.delete(MP[j],insert_index)
                            np.insert(MP[j],insert_position-1,insert_element)
                        else:
                            np.delete(MP[j],insert_index)
                            np.insert(MP[j],insert_position,insert_element)


                    #翻转
                    else:
                        #仅对后半段进行处理，即随机选取染色体上的两个位置，将两者之间的加工顺序重新排列
                        change_index1, change_index2 = total_process_num + np.sort(np.random.choice(total_process_num,2,replace=False))
                        num_change = change_index2 - change_index1
                        MP[j][change_index1:change_index2] = np.random.choice(MP[j][change_index1:change_index2],num_change,replace=False)
        
        
        #世代策略  -- 0代表精英策略 1代表竞争策略
        if generation_strategy == 0: #这里对精英策略进行一点修改。由于前述代码根据交叉策略产生了与父代一样数量的样本，因此选择其中最优的N-lamb个进入子代
            
            lamb = N//2
            best_father_index = np.argsort(fitness)[-lamb:][::-1]
            son_fitness,_ = cal_fitness(func,MP,machines,times)
            best_son_index = np.argsort(son_fitness)[-(N-lamb):][::-1]
        
            #得到新一代
            population = np.append(population[best_father_index], MP[best_son_index],axis=0)

        else:
            population = np.append(population, MP,axis=0)
            son_fitness,_ = cal_fitness(func,MP,machines,times)
            fitness = np.append(fitness,son_fitness,axis=0)
            best_index = np.argsort(fitness)[-N:][::-1]

            #得到新一代
            population = population[best_index]

    
    #得到最优结果
    for i in range(population.shape[0]):
        C_max[i],_ = func(population[i],machines,times)
    best_index = np.argmin(C_max)
    best_C_max, best_stat_df = func(population[best_index],machines,times)
    
    #将工序表修改成标准形式
    best_stat_df = best_stat_df.sort_values(by=["设备编号","零件编号","工序编号"])
    best_stat_df.set_index(pd.Index([i for i in range(total_process_num)]),inplace=True)

    np.append(C_max_list,best_C_max)
    return best_C_max, best_stat_df, C_max_list
        
                        
