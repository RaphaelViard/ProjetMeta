import read_pb
from src.Heuristics import *
from src.utils import *
import numpy as np
import json
import time

"""
start = time.time()
end = time.time()

.perf_counter()

print(f"Temps d'exécution : {end - start:.3f} secondes")
"""


instance = "cap132"

k = 3
p = 2

m, n, fcosts, caps, costs = load_instance(instance)


print(f"\nFacilities: {m}, Customers: {n}\n")


start = time.time()
x,y = test_opening(n,m,fcosts,costs)
end = time.time()

value = objective_function(x, y, fcosts, costs)
print("Start value : ",value,"")
print(f"Temps d'exécution : {end - start:.5f} s\n")





print("k : ",k)

start = time.time()
desc_x, desc_y, desc_value = descent(x,y,fcosts, costs, k)
end = time.time()
print("Descent value : ",desc_value,"")
print(f"Temps d'exécution : {end - start:.4f} s\n")


m, n, fcosts, caps, costs = load_instance(instance)

x,y = test_opening(n,m,fcosts,costs)

value = objective_function(x, y, fcosts, costs)



print("p : ",p)

start = time.time()
vns_x, vns_y, vns_value = vns(x,y,fcosts, costs, p)
end = time.time()
print("VNS value : ",vns_value,"")
print(f"Temps d'exécution : {end - start:.4f} s\n")