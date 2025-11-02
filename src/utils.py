import numpy as np
import read_pb


def load_instance(f: str):
    m, n, fcosts, caps, costs = read_pb.read_orlib_ufl("/root/Projet_META/ProjetMeta/uncap_data/" + f + ".txt")
    return m,n, np.array(fcosts), np.array(caps), np.array(costs)


def is_feasible(x, y):
    if  not np.all(np.sum(x, axis=1) == 1):
        print("Somme sur x non à 1")
        return False
    if not np.all(x <= y):
        print("Allocation a des usines non ouvertes")
        return False
    if not (np.all(np.isin(x, [0, 1]))):
        print("x non binaire")
        return False
    if not (np.all(np.isin(y, [0, 1]))):
        print("y non binaire")
        return False
    return True



def objective_function(x, y, fcosts, costs):
    return np.sum(costs * x) + np.dot(fcosts, y)
