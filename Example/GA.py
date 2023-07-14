#实现遗传算法
import numpy as np
from init import cal_Cmax, cal_C_max_faster
from tqdm import tqdm
import copy

def get_final_result(chromosome,machines):
    job_num = machines.shape[0]
    machine_num = machines.shape[1]
    
    #机器的累积工作数矩阵
    sum_works_num = np.zeros(machine_num)
    #工件的累积工序数
    sum_process_num = np.zeros(job_num)
    
    total_process_num = machine_num * job_num

    machines_plan = np.zeros((machine_num,job_num))
    for i in range(total_process_num):
        part, process = int(chromosome[i]), int(sum_process_num[int(chromosome[i])])

        machine = machines[part,process] - 1
        job_index = int(sum_works_num[machine])
        machines_plan[machine,job_index] = part

        sum_process_num[part] += 1
        sum_works_num[machine] += 1
    
    return machines_plan



def get_all_process(machines):
    all_process = []
    job_num = machines.shape[0]
    machine_num = machines.shape[1]
    for i in range(job_num):
        for j in range(machine_num):
            all_process.append(i)
    return np.array(all_process,dtype=np.int16)


def cal_fitness(func,population,machines,times):
    C_max = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        C_max[i],_,_ = func(population[i],machines,times)
    fitness = np.max(C_max) - C_max
    return fitness, np.min(C_max) 

def cal_fitness_faster(func,population,machines,times): #尝试加快计算速度的一个计算适应度函数，没有效果所以取消了
    C_max = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        C_max[i] = func(population[i],machines,times)
    fitness = np.max(C_max) - C_max
    return fitness, np.min(C_max) 

def TLBO_MP(MP,temp_best_index):
    total_process_num = MP.shape[1]
    TLBO_MP = np.zeros_like(MP)
    need_change_list = list(range(MP.shape[0]))
    need_change_list.remove(temp_best_index)#将最优样本从交配样本中删除
    
    for i in need_change_list:
        temp_best_process = MP[temp_best_index]
        need_change_process = MP[i]
        for j in range(total_process_num):
            rate = np.random.rand()
            
            if rate > 0.5: #rate大于0.5则选择当前最优个体的第一个工序安排
                TLBO_MP[i,j] = temp_best_process[0]
                need_delete_part = temp_best_process[0]
                temp_best_process = np.delete(temp_best_process, 0)
                temp_need_delete_index = np.argmax(need_change_process == need_delete_part)  #需要删除工件号在
                need_change_process = np.delete(need_change_process, temp_need_delete_index)
            
            else:
                TLBO_MP[i,j] = need_change_process[0]
                need_delete_part = need_change_process[0]
                need_change_process = np.delete(need_change_process, 0)
                need_delete_index = np.argmax(temp_best_process == need_delete_part)
                temp_best_process = np.delete(temp_best_process, need_delete_index)
    
    TLBO_MP[temp_best_index] = MP[temp_best_index]
    
    return TLBO_MP


def delete_code(code,need_delete_code_list):
    for i in range(need_delete_code_list.shape[0]):
        temp_need_delete_index = np.argmax(code == need_delete_code_list[i])  #需要删除工件号在
        code = np.delete(code, temp_need_delete_index)
    return code

def wrong_crossover(MP,total_process_num,crossover_rate):
    cross_index = np.random.choice(MP.shape[0],MP.shape[0],replace=False)
    cross_index = cross_index.reshape(-1,2)  #两两配对得到进行交叉的索引

    for j in range(cross_index.shape[0]):  #需要进行交叉的最大次数
        c_r = np.random.rand()
        if c_r <= crossover_rate:
            cross_position = np.random.choice(2,total_process_num)#得到交换位置的0-1编码  
            #这里1代表需要进行交叉的位置，0代表无需交叉的位置，这样处理是为了方便统计需要交叉的位置个数
            individual1, individual2 = cross_index[j]
            num_cross = np.sum(cross_position)  #需要进行交叉的数量
            cross_position_later = (cross_position==1)
            MP[individual1,cross_position_later] = np.random.choice(MP[individual1,cross_position_later],num_cross,replace=False)
            MP[individual2,cross_position_later] = np.random.choice(MP[individual2,cross_position_later],num_cross,replace=False)
        else:
            pass

def true_crossover(MP,total_process_num,crossover_rate):
    cross_index = np.random.choice(MP.shape[0],MP.shape[0],replace=False)
    cross_index = cross_index.reshape(-1,2)  #两两配对得到进行交叉的索引
    for j in range(cross_index.shape[0]):  #需要进行交叉的最大次数
        c_r = np.random.rand()
        if c_r <= crossover_rate:
            cross_position = np.random.choice(2,total_process_num)#得到交换位置的0-1编码  
            #这里1代表需要进行交叉的位置，0代表无需交叉的位置
            need_cross_position = np.where(cross_position == 1)
            unneed_cross_position = np.where(cross_position == 0)

            cross_index1, cross_index2 = cross_index[j]
            individual1, individual2 = copy.deepcopy(MP[cross_index1]), copy.deepcopy(MP[cross_index2])
            #提取出每个交配个体在另一个交配个体上要删除的编码
            delete_code1 = individual1[unneed_cross_position]
            delete_code2 = individual2[unneed_cross_position]

            need_code1 = delete_code(individual2,delete_code1)
            need_code2 = delete_code(individual1,delete_code2)
            
            individual1[need_cross_position] = need_code1
            individual2[need_cross_position] = need_code2

            MP[cross_index1] = individual1
            MP[cross_index2] = individual2

    return MP


  


def JS_GA(func,machines,times,Max_iter=1000,N=50,crossover_rate=0.75,mutation_rate=0.01,selection_index=0,generation_strategy=0,dynamic_change=False,TLBO=False):
    #初始化种群
    job_num = machines.shape[0]
    machine_num = machines.shape[1]
    total_process_num = job_num * machine_num
    C_max_list = np.zeros(Max_iter)
    population = np.zeros((N,total_process_num))
    C_max = np.zeros(N)

    #得到所有工序组成的数组
    all_process = get_all_process(machines)
    

    for i in range(N):
        population[i] = np.random.choice(all_process,total_process_num,replace=False)

    unchange_iter = 0

    crossover_rate0 = crossover_rate
    mutation_rate0 = mutation_rate
    
    for i in tqdm(range(Max_iter)):
        fitness, min_C_max = cal_fitness_faster(func,population,machines,times)
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
                #1.当方差变大时，说明种群的聚集程度减小，此时就要减小变异率，增大交叉率，来提高种群的聚集程度
                #2.当方差变小时，说明种群聚集程度增大，陷入局部最优的可能性增大，此时就要减小交叉率，增大变异率来降低种群的聚集程度
                #3.当最优值不变的次数增加时，说明种群陷入局部最优的可能性增加，此时就要提高变异率，减小交叉率
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
        

        #TLBO交叉
        if TLBO:
            if i >= 500:
                MP_fitness, _ = cal_fitness(func,MP,machines,times)
                temp_best_index = np.argmax(MP_fitness)
                
                MP = TLBO_MP(MP,temp_best_index)
                
            else:#普通交叉
                true_crossover(MP,total_process_num,crossover_rate)
        
        else:#普通交叉
            #wrong_crossover(MP,total_process_num,crossover_rate)
            true_crossover(MP,total_process_num,crossover_rate)

        #变异--这里的变异是针对工序排列顺序进行变异

        for j in range(MP.shape[0]): 
            for k in range(total_process_num):
                m_r = np.random.rand()
                if m_r <= mutation_rate:
                    #选择变异的方式 0--交换 1--插入 2--翻转
                    mutation_selection = np.random.randint(0,3,1)[0]  
                    if mutation_selection == 0:  #交换--随机选择染色体上相邻的一对基因交换位置
                        if k == total_process_num-1:
                            MP[j,k],MP[j,0] = MP[j,0],MP[j,k]
                        else:
                            MP[j,k],MP[j,k+1] = MP[j,k+1],MP[j,k]

                    elif mutation_selection == 1: #插入--随机选择染色体上的一个基因，插入到另一个随机位置
                        insert_index, insert_position = np.random.choice(total_process_num,2)
                        insert_element = MP[j,insert_index]
                        if insert_index < insert_position:
                            np.delete(MP[j],insert_index)
                            np.insert(MP[j],insert_position-1,insert_element)
                        else:
                            np.delete(MP[j],insert_index)
                            np.insert(MP[j],insert_position,insert_element)

                    else:   #翻转--随机选取染色体上的两个位置，将两者之间的加工顺序重新排列
                        change_index1, change_index2 = np.sort(np.random.choice(total_process_num,2,replace=False))
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
        C_max[i],_,_ = func(population[i],machines,times)
    best_index = np.argmin(C_max)
    best_C_max, machine_start_time, machine_end_time = func(population[best_index],machines,times)

    best_sol = get_final_result(population[best_index],machines)
    return best_C_max, best_sol, machine_start_time, machine_end_time, C_max_list
    

   


        

    
        
