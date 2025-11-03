import numpy as np
import read_pb
from src.utils import *

def load_instance_cap(f: str):
    m, n, fcosts, caps, demands, costs = read_pb.read_orlib_try("/ProjetMeta/uncap_data/" + f + ".txt")
    return m,n, np.array(fcosts), np.array(caps), np.array(demands), np.array(costs)


def is_feasible_caps(x, y, caps, demands):
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
    if not np.all(np.dot(demands, x) <= caps + 1e-9):
        print("Capacité d'une usine dépassée")
        return False
    return True

def give_affectations(y, costs):
    n, m = costs.shape
    x = np.zeros((n,m))
    min_idx = np.argmin(np.where(y==1, costs, np.inf), axis=1)
    x[np.arange(n), min_idx] = 1
    return x

def neighborhood1_caps(x, y, fcosts, costs, caps, demands):
    n, m = x.shape
    neighbors = []
    for j in range(m):
        new_y = y.copy()
        new_y[j] = 1-y[j] # opening or closing
        new_x = give_affectations(new_y, costs)
        if is_feasible_caps(new_x, new_y, caps, demands):
            neighbors.append((new_x, new_y))
    return neighbors

def neighborhood2_caps(x, y, fcosts, costs, caps, demands):
    n, m = x.shape
    neighbors = []
    for i in range(m):
        for j in range(i+1,m):
            new_y = y.copy()
            new_y[i] = 1-y[i] # opening or closing
            new_y[j] = 1-y[j] # opening or closing
            new_x = give_affectations(new_y, costs)
            if is_feasible_caps(new_x, new_y, caps, demands):
                neighbors.append((new_x, new_y))
    return neighbors

def neighborhood3_caps(x, y, fcosts, costs, caps, demands):
    n, m = x.shape
    neighbors = []

    for j in range(m):  # pour chaque usine
        if y[j] == 0:
            continue  # déjà fermée

        clients_j = np.where(x[:, j] == 1)[0]
        if len(clients_j) == 0:
            continue  # aucun client à réaffecter

        # on tente de réaffecter tous ces clients vers d'autres usines ouvertes
        new_x = x.copy()
        new_y = y.copy()
        new_x[clients_j, j] = 0
        remaining_caps = caps.copy()
        # calculer la capacité restante pour chaque usine
        for k in range(m):
            remaining_caps[k] -= np.sum(demands[np.where(new_x[:, k] == 1)[0]])

        feasible = True
        for i in clients_j:
            # chercher une autre usine ouverte qui peut prendre ce client
            candidate_usines = [k for k in range(m) if k != j and new_y[k] == 1 and demands[i] <= remaining_caps[k]]
            if len(candidate_usines) == 0:
                feasible = False
                break
            # choisir celle avec le coût de service le plus faible
            best_k = min(candidate_usines, key=lambda k: costs[i, k])
            new_x[i, best_k] = 1
            remaining_caps[best_k] -= demands[i]

        if feasible:
            # si réaffectation réussie, fermer l'usine j
            new_y[j] = 0
            neighbors.append((new_x, new_y))

    return neighbors

def all_neighbors_caps(x, y, fcosts, costs, caps, demands):
    neighbors1 = neighborhood1_caps(x, y, fcosts, costs, caps, demands)
    neighbors2 = neighborhood2_caps(x, y, fcosts, costs, caps, demands)
    neighbors3 = neighborhood3_caps(x, y, fcosts, costs, caps, demands)
    return neighbors1 + neighbors2

def test_opening_caps(costs, fcosts, caps, demands):
    n, m = costs.shape

    x = np.zeros((n, m), dtype=int)
    y = np.zeros(m, dtype=int)
    current_cost = np.full(n, np.inf)
    assigned = np.zeros(n, dtype=bool)
    remaining_cap = caps.copy()

    for j in range(m):
        # gain potentiel par client : combien on économise en passant à cette usine
        potential_gain = current_cost - costs[:, j]

        # on ne garde que les clients non encore totalement servis mais profitables
        candidate_clients = np.where(potential_gain > 0)[0]

        if len(candidate_clients) == 0:
            continue

        # Tri des clients : on préfère ceux avec le meilleur gain par unité de demande
        # (sinon gros clients saturent vite une usine)
        ratio = potential_gain[candidate_clients] / demands[candidate_clients]
        order = candidate_clients[np.argsort(-ratio)]

        served = []
        cap_j = remaining_cap[j]

        # On ajoute les clients tant que la capacité le permet
        for i in order:
            if demands[i] <= cap_j:
                served.append(i)
                cap_j -= demands[i]

        if len(served) == 0:
            continue

        # calcul du gain total réalisé si on ouvre l'usine j
        total_gain = np.sum(potential_gain[served])

        # on ouvre l'usine seulement si gain > coût fixe
        if total_gain > fcosts[j]:
            y[j] = 1
            remaining_cap[j] = cap_j

            # On réaffecte ces clients à cette usine
            x[served, :] = 0
            x[served, j] = 1
            current_cost[served] = costs[served, j]
            assigned[served] = True

    # Quand c'est fini : si encore des clients non servis → fallback
    # On attribue au plus proche possible avec capacité restante
    unassigned = np.where(~assigned)[0]
    for i in unassigned:
        # tri des usines par coût croissant
        order = np.argsort(costs[i])
        for j in order:
            if demands[i] <= remaining_cap[j]:
                y[j] = 1
                x[i, :] = 0
                x[i, j] = 1
                remaining_cap[j] -= demands[i]
                break
        else:
            print(f"Pas possible d'affecter le client {i} ")
    return x, y

"""Recuit dans lequel on choisit aléatoirement le voisin vers lequel on bouge, et T fixé à l'avance"""
def random_recuit_caps(x,y,fcosts, costs, caps, demands, neighbors_function, T, verbose=False):
    n,m = x.shape
    neighbors = neighbors_function(x, y, fcosts, costs, caps, demands)
    new_x, new_y = neighbors[np.random.randint(len(neighbors))]
    old_obj = objective_function(x, y, fcosts, costs)
    new_obj = objective_function(new_x, new_y, fcosts, costs)
    delta = new_obj-old_obj
    if delta<0:
        if verbose:
            print("New solution is better")
        return new_x, new_y
    else:
        p = np.exp(-delta/T)
        if np.random.rand() < p:
            if verbose:
                print("New solution is not better but we keep it")
            return new_x, new_y
    if verbose:
        print("We keep the same solution")
    return x, y