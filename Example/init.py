import numpy as np
import pandas as pd

#加载数据
def load_data(data_path):
    times = pd.read_excel(data_path, sheet_name=0)
    machines = pd.read_excel(data_path, sheet_name=1)

    times = np.array(times)
    machines = np.array(machines)

    return times, machines

#将加工顺序编码解码成shape为(total_process_num，3)的矩阵，3个元素分别表示加工机器、零件编号以及工序
def decoder(chromosome,machines):
    part_num = machines.shape[0]
    process_num = machines.shape[1]
    total_process_num = part_num * process_num  #第一个维度为工件数 第二个维度为工序数
    schedule = np.zeros((total_process_num,3),dtype=np.int16)
    sum_process = np.zeros(part_num)

    for i in range(total_process_num):
        part = int(chromosome[i])  #得到工件序号
        process = int(sum_process[part])  #得到工件的工序
        machine = machines[part,process] #根据工件和工序得到加工的机器
        sum_process[part] += 1  #将工序数加1
        schedule[i,0] = machine - 1
        schedule[i,1] = part
        schedule[i,2] = process
    return schedule
    


#计算最大完工时间，返回最大完工时间以及每个机器的开始工作时间以及结束时间矩阵
def cal_Cmax(chromosome, machines, times):
    schedule = decoder(chromosome,machines) #该矩阵内部元素为[机器，工件，工序]

    num_jobs, num_machines = machines.shape[0], machines.shape[1]
    total_process_num = num_jobs * num_machines

    #定义工作开始时间和结束时间矩阵 行代表工作，列代表工序
    job_start_time = np.zeros((num_jobs,num_machines),dtype=np.int16)
    job_end_time = np.zeros((num_jobs,num_machines),dtype=np.int16)

    #定义机器开始时间和完工时间矩阵  行代表机器，列代表该机器的第几件工作
    machine_start_time = np.zeros((num_machines,num_jobs),dtype=np.int16)
    machine_end_time = np.zeros((num_machines,num_jobs),dtype=np.int16)

    #机器的累积工作数矩阵
    sum_works_num = np.zeros(num_machines)

    for i in range(total_process_num):
        machine, part, process = schedule[i]
        job_index = int(sum_works_num[machine])
        if sum_works_num[machine] == 0: #是机器的第一次加工
            if process == 0:  #是第一道工序
                machine_start_time[machine,0] = 0
                machine_end_time[machine,0] = times[part,process]
            
            else:#不是第一道工序
                machine_start_time[machine,0] = job_end_time[part,process-1]
                machine_end_time[machine,0] = machine_start_time[machine,0] + times[part,process]

        else:#不是机器的第一次加工
            if process == 0:#是第一道工序
                machine_start_time[machine,job_index] = machine_end_time[machine,job_index-1]
                machine_end_time[machine,job_index] = machine_start_time[machine,job_index] + times[part,process]

            else:#不是第一道工序
                machine_start_time[machine,job_index] = np.max([job_end_time[part,process-1],machine_end_time[machine,job_index-1]])
                machine_end_time[machine,job_index] = machine_start_time[machine,job_index] + times[part,process]
        
        #更新机器的累积加工数
        sum_works_num[machine] += 1
        job_start_time[part,process] = machine_start_time[machine,job_index]
        job_end_time[part,process] = machine_end_time[machine,job_index]

    
    
    C_max = np.max(machine_end_time)
    return C_max, machine_start_time, machine_end_time


def faster_decoder(chromosome,machines): #将编码，解码为行号为机器序号，列号为加工顺序编号，元素为工件编号的矩阵
    num_jobs, num_machines = machines.shape[0], machines.shape[1]
    schedule = np.zeros((num_machines,num_jobs),dtype=np.int16)
    sum_process_list = np.zeros(num_jobs)
    machine_work_list = np.zeros(num_machines)
    for i in range(chromosome.shape[0]):
        part = int(chromosome[i])
        process = int(sum_process_list[part])
        machine = int(machines[part,process]) - 1
        work_index = int(machine_work_list[machine])
        schedule[machine,work_index] = part
        sum_process_list[part] += 1
        machine_work_list[machine] += 1
    
    return schedule

def change_time(times,machines):#将时间矩阵变成行号为工件号，列号为机器号，元素为第i个工件在j机器加工的时间
    change_times = np.zeros_like(times)
    for i in range(times.shape[0]):
        for j in range(times.shape[1]):
            part = i
            machine = machines[i,j] - 1
            time = times[i,j]
            change_times[part,machine] = time
    return change_times


def cal_C_max_faster(chromosome, machines, times):
    schedule = faster_decoder(chromosome,machines) #该矩阵行为机器序号，列号为加工顺序编号，元素为工件编号

    num_jobs, num_machines = machines.shape[0], machines.shape[1]
    total_process_num = num_jobs * num_machines
    change_times = change_time(times,machines)
    times_matrix = np.zeros((num_machines,num_jobs),dtype=np.int16)
    
    a = 0
    for i in range(num_jobs):
        part = schedule[0,i]#机器1的加工工件号
        a += change_times[part,0]
        times_matrix[0,i] = a

    for i in range(1,num_machines):
        for j in range(num_jobs):
            if j == 0:
                part = schedule[i,j]
                position = np.where(schedule[i-1]==part)[0]
                former_time = times_matrix[i-1,position]
                complete_time = former_time + change_times[part,i]
                times_matrix[i,j] = complete_time
            else:
                part = schedule[i,j]
                position = np.where(schedule[i-1]==part)[0]
                former_time = np.max([times_matrix[i-1,position],times_matrix[i,j-1]])
                complete_time = former_time + change_times[part,i]
                times_matrix[i,j] = complete_time

    return times_matrix[num_machines-1,num_jobs-1]




