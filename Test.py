import read_pb
from src.Heuristics import *
from src.utils import *
import numpy as np
import json


instance = "cap131"

m, n, fcosts, caps, costs = load_instance(instance)


print(f"Facilities: {m}, Customers: {n}")




Verbose = True
x,y = test_opening(n,m,fcosts,costs)
#x,y = random_opening(n, m, fcosts, costs, 0.3)

best_obj = objective_function(x, y, fcosts, costs)
best_x, best_y = x,y
mu = 0.9
T = 10000
"""
for i in range(500):
    x, y = random_recuit(x, y, fcosts, costs, all_neighbors,T, False)
    T = mu*T
    current_obj = objective_function(x, y, fcosts, costs)  
    if current_obj < best_obj:
        if Verbose:
            print(f"New sol opt at step {i}")
        best_obj = current_obj
        best_x, best_y = x, y


"""
for i in range(200):
    x, y = Advanced_recuit(x, y, fcosts, costs, all_neighbors, True)
    current_obj = objective_function(x, y, fcosts, costs)  
    if current_obj < best_obj:
        if Verbose:
            print(f"New sol opt at step {i}")
        best_obj = current_obj
        best_x, best_y = x, y



bool = is_feasible(best_x,best_y)
if bool:
    with open("results.json", "r") as f:
        data = json.load(f)
        print(f"Solution feasible, cost = {round(objective_function(x, y, fcosts, costs),2)}, Optimal cost = {round(data[instance], 2)}")
else:
    print("Solution infeasible")



#print(m)