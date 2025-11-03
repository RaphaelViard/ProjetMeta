import read_pb
#from src.Heuristics import *
#from src.utils import *
import numpy as np
from Functions_Q5 import *
import json


instance = "cap131"

m, n, fcosts, caps, demands, costs = load_instance_cap(instance)


x, y = test_opening_caps(costs, fcosts, caps, demands)

isfsbl = is_feasible_caps(x, y, caps, demands)

print(isfsbl)


best_obj = objective_function(x, y, fcosts, costs)
print(f"Initial obj is : {best_obj}")
best_x, best_y = x,y
mu = 0.9
T = 10000
Verbose = True

for i in range(200):
    x, y = random_recuit_caps(x, y, fcosts, costs, caps, demands, all_neighbors_caps,T, False)
    T = mu*T
    current_obj = objective_function(x, y, fcosts, costs)  
    if current_obj < best_obj:
        if Verbose:
            print(f"New sol opt at step {i}")
        best_obj = current_obj
        best_x, best_y = x, y





bool = is_feasible_caps(best_x,best_y, caps, demands)
if bool:
    with open("results.json", "r") as f:
        data = json.load(f)
        print(f"Solution feasible, cost = {round(objective_function(best_x, best_y, fcosts, costs),2)}, Optimal cost without capacities = {round(data[instance], 2)}")
else:
    print("Solution infeasible")


