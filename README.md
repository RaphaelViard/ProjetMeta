# 1. Introduction
This project implements various metaheuristic methods to solve a resource allocation problem.  
It allows testing and comparing the performance of different algorithms on multiple problem instances.  

## 2. Files & parameters

For the classical descent, The file Test3 can be used, with the following parameters we can change:

instance = The instance we want to test  
k = The hamming distance of the neighborhood chosen  
p = A way to generate neighbors : explore by increasing Hamming distance, accept better solutions, stop after p neighborhoods  

For the Simulated Annealing, Test4 with the following parameters we can change:  

instance = The instance we want to test  
p = A way to generate neighbors : explore by increasing Hamming distance, accept better solutions, stop after p neighborhoods  
mu = Cooling coefficient  
nb_iter = The number of cooling iterations of the simulated annealing  
iter_max = The number of neighbors chosen for an iteration of the simulated annealing  

For the version with demands, TestQ5, with the following parameters:  

instance = The instance we want to test  
mu = Cooling coefficient  
T = The temperature  
