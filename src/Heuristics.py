import numpy as np
from src.utils import *


#Here we have all the heuristics, neighborhoods
""" As simply as possible : we just open 1 location and affect every customer to it """

def basic_feas(n,m):
    x = np.zeros((n,m))
    y = np.zeros(m)
    y[0]=1
    for i in range(x.shape[0]):
        x[i,0] = 1
    return x,y


""" Given a set of open locations, we return the best affectations for customer (useful only if there is not max capacity) """
def give_affectations(y, costs):
    n, m = costs.shape
    x = np.zeros((n,m))
    min_idx = np.argmin(np.where(y==1, costs, np.inf), axis=1)
    x[np.arange(n), min_idx] = 1
    return x



""" The first heuristcs for finding an feasible solution : we try to open every location, by comparing the opening costs with what we gain by changing """
def test_opening(n,m, fcosts, costs):
    x = np.zeros((n,m))
    y = np.zeros(m)
    current_service_costs = np.full(n, np.Inf)
    for j in range(m): #But : comparer cout de construction de l'usine i, avec gains associés  aux changements de clients vers la nouvelle usine
        changes_id = np.where(current_service_costs > costs[:,j]) # les couts meilleurs avec la nouvelle potentielle usine
        gains_clients = np.sum(current_service_costs[changes_id]-costs[changes_id, j]) #la valeur de ce qu'on gagne en cout de service (>0)
        if gains_clients >  fcosts[j]: #on construit bien la nouvelle usine
            y[j]=1
            x[changes_id,:] =  0
            x[changes_id,j] = 1
            current_service_costs[changes_id] = costs[changes_id,j]
    return x,y


"""Give all the neighbors of a couple x,y, as described in the report"""
def neighborhood1(x, y, fcosts, costs):
    n, m = x.shape
    neighbors = []
    for j in range(m):
        new_y = y.copy()
        new_y[j] = 1-y[j]
        new_x = give_affectations(new_y, costs)
        neighbors.append((new_x, new_y))
    return neighbors

def neighborhood2(x, y, fcosts, costs):
    n, m = x.shape
    neighbors = []

    return neighbors


"""Can find a local optimum, just improve the solution with the best neighbor"""
def Simple_search(x,y,fcosts, costs, neighbors_function, verbose=False):
    n, m = x.shape
    
    old_obj = objective_function(x, y, fcosts, costs)
    
    neighbors = neighbors_function(x, y, fcosts, costs)

    improving = []
    worsening = []
    deltas = []

    for i, (new_x, new_y) in enumerate(neighbors):
        new_obj = objective_function(new_x, new_y, fcosts, costs)
        delta = new_obj - old_obj
        deltas.append(delta)
        if delta < 0:
            improving.append(i)
        elif delta > 0:
            worsening.append(i)
    deltas = np.array(deltas)
    if len(improving)==0:
        return x,y
    best_index = improving[np.argmin(deltas[improving])]
    best_x, best_y = neighbors[best_index]

    if verbose:
        print(f"Solution improved with a gap = {deltas[best_index]:.4f}")
    return best_x, best_y
    




"""Recuit dans lequel on choisit aléatoirement le voisin vers lequel on bouge, et T fixé à l'avance"""
def random_recuit(x,y,fcosts, costs, neighbors_function, T, verbose=False):
    n,m = x.shape
    neighbors = neighbors_function(x, y, fcosts, costs)
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




def Advanced_recuit(x, y, fcosts, costs, neighbors_function, verbose=False):
    n, m = x.shape
    
    old_obj = objective_function(x, y, fcosts, costs)
    
    neighbors = neighbors_function(x, y, fcosts, costs)

    improving = []
    worsening = []
    deltas = []

    for i, (new_x, new_y) in enumerate(neighbors):
        new_obj = objective_function(new_x, new_y, fcosts, costs)
        delta = new_obj - old_obj
        deltas.append(delta)
        if delta < 0:
            improving.append(i)
        elif delta > 0:
            worsening.append(i)
    
    deltas = np.array(deltas)
    
    if len(worsening) == 0:
        i = np.random.choice(improving)
        return neighbors[i][0], neighbors[i][1]
    
    if len(improving) == 0: # We are stuck in a local opt
        T = abs(np.mean(deltas[worsening]))
        i = np.random.choice(worsening)
        new_x, new_y = neighbors[i]
        delta = deltas[i]

        p = np.exp(-delta / T)
        if np.random.rand() < p:
            return new_x, new_y
        return x, y
    if np.random.rand() < 0.98: # Dans le cas ou a des sol qui improvent et d'autrs qui n'improvent pas, faire : choisir une qui improve

        i = np.random.choice(improving)
        return neighbors[i][0], neighbors[i][1]
    
    else:

        T = abs(np.mean(deltas[worsening]))
        i = np.random.choice(worsening)
        new_x, new_y = neighbors[i]
        delta = deltas[i]
        p = np.exp(-delta / T)
        if np.random.rand() < p:
            return new_x, new_y
        return x, y