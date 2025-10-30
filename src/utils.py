import numpy as np
import read_pb


def load_instance(f: str):
    m, n, fcosts, caps, costs = read_pb.read_orlib_ufl("/ProjetMeta/uncap_data/" + f + ".txt")
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
    N,M = np.shape(x)
    Sj = 0
    for j in range(M):
        Si = 0
        for i in range(N):
            Si +=  costs[i][j] * x[i][j]
        Sj += Si + fcosts[j] * y[j]
    return Sj
