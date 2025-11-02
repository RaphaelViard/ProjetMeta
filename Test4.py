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


instance = "capc"

p = 3 # Précédemment, 3 suffisait
mu = 0.85 # Décroissance plus rapide
nb_iter = 350
iter_max = 250


m, n, fcosts, caps, costs = load_instance(instance)

print("\nInstance : ",instance)
print(f"Facilities: {m}, Customers: {n}")

x,y = test_opening(n,m,fcosts,costs)

value = objective_function(x, y, fcosts, costs)
print("Start value : ",value,"\n")

T = value / 3 # Décroissance plus rapide

print("Paramètres :")
print("p : ",p)
print("T : ",T)
print("mu : ",mu)
print("nb_iter : ",nb_iter)
print("iter_max : ",iter_max,"\n")

start = time.time()
x_res, y_res, value_res = simulated_annealing(x, y, fcosts, costs, p, T, mu, nb_iter, iter_max)
end = time.time()
print("Résultat :")
# print("y : ",y_res)
print("value : ",value_res)
print(f"Temps d'exécution : {end - start:.4f} s\n")

ok = is_feasible(x_res,y_res)
print("Solution réalisable : ",ok,"")

