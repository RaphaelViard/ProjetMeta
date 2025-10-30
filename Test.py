import read_pb
from src.Heuristics import *
from src.utils import *
import numpy as np
import json


instance = "cap73"

m, n, fcosts, caps, costs = load_instance(instance)


print(f"Facilities: {m}, Customers: {n}")





x,y = test_opening(n,m,fcosts,costs)

bool = is_feasible(x,y)
if bool:
    with open("results.json", "r") as f:
        data = json.load(f)
        print(f"Solution feasible, cost = {objective_function(x, y, fcosts, costs)}, Optimal cost = {round(data[instance], 2)}")
else:
    print("Solution infeasible")



#print(m)