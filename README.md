# FJSP
Genetic_algorithm for FJSP
This is a solution to Flexible Jobshop Scheduling Problem(FJSP) using genetic algorithm
And there are detailed Chinese annotations in the code.
Example is the solutions to 3 classic examples--ta01, ta40, ta60
Realcase is the solution to a real case, and sorry to provide the detailed data.
I hope it can help you.

# Genetic Algorithm
## Code Method
MS & OS is applied in my algorithm, which means the machines are coded in the former half chromosome and the orders are the later. And this method can avoid the infeasible solution.
![Code Method](https://github.com/CokerPad/FJSP/blob/main/3.png)
## Global Initialization
To improve the performance of initial solutions, the local best machine is selected when scheduling the MS.
![Global Initialization](https://github.com/CokerPad/FJSP/blob/main/5.png)
## Dynamic Crossover and Mutation Rate
High crossover rate and low mutation rate increase cluster of the population, improving the local search ability; high mutation rate and low crossover rate decrease cluster of the population, improving the global search ability. Meanwhile, the number of unchanged iterations is the important factor.

## TLBO Crossover<sup>[1]
In the later stage of algorithm, TLBO crossover is applied, in which the best individual will crossover with the rest of the population to improve the performance of whole population.
![TLBO Crossover](https://github.com/CokerPad/FJSP/blob/main/6.png)
# Reference
[1] 周鹏鹏,翟志波,戴玉森.基于改进遗传算法的柔性作业车间调度问题研究[J].组合机床与自动化加工技术,2023(03):183-186+192.DOI:10.13462/j.cnki.mmtamt.2023.03.044.
