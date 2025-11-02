import read_pb
from src.Heuristics import *
from src.utils import *
import numpy as np
import json


instance = "cap71"

m, n, fcosts, caps, costs = load_instance(instance)


print(f"\nFacilities: {m}, Customers: {n}\n")

x,y = test_opening(n,m,fcosts,costs)
# print(x)
# print("y : ",y)

value = objective_function(x, y, fcosts, costs)
print("Start value : ",value,"\n")

def print_n_y(l):
    for e in l:
        print(e)
    return

k = 1

neighbor3 = neighborhood3(x, y, fcosts, costs, k)
neighbor3_y = [b for (a, b) in neighbor3]
# print_n_y(neighbor3_y)

"""

k = 1

neighbor3 = neighborhood3(x, y, fcosts, costs, k)
neighbor3_y = [b for (a, b) in neighbor3]
# print_n_y(neighbor3_y)

neighbor3random = neighborhood3random(x, y, fcosts, costs, k)
# print(neighbor3random[0][1])
"""



"""
k = 1

neighbor3 = neighborhood3(x, y, fcosts, costs, k)
neighbor3_y = [b for (a, b) in neighbor3]
print_n_y(neighbor3_y)

print("\n\n")

neighbor1 = neighborhood1(x, y, fcosts, costs)
neighbor1_y = [b for (a, b) in neighbor1]
print_n_y(neighbor1_y)

print(np.array_equal(neighbor3_y, neighbor1_y))

"""

"""
k = 2

neighbor3 = neighborhood3(x, y, fcosts, costs, k)
neighbor3_y = [b for (a, b) in neighbor3]
print_n_y(neighbor3_y)

print("\n\n\n\n\n")

neighbor2 = neighborhood2(x, y, fcosts, costs)
neighbor2_y = [b for (a, b) in neighbor2]
print_n_y(neighbor2_y)

print(np.array_equal(neighbor3_y, neighbor2_y))
"""

"""
p = 2

neighbor4 = neighborhood4(x, y, fcosts, costs, p)
neighbor4_y = [b for (a, b) in neighbor4]
# print_n_y(neighbor4_y)
"""


# best_x, best_y, best_value = best_neighbor(x, y, fcosts, costs, k, False)
# print("1st Neightbor value : ",best_value,"\n")


desc_x, desc_y, desc_value = descent(x,y,fcosts, costs, k)
print("Descent value : ",desc_value,"\n")

p = 3

vns_x, vns_y, vns_value = vns(x,y,fcosts, costs, p)
print("VNS value : ",vns_value,"\n")






