import numpy as np
from src.utils import *
from itertools import combinations


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

def random_opening(n, m, fcosts, costs, p_open=0.5, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    y = np.random.rand(m) < p_open
    if not np.any(y):
        y[np.random.randint(0, m)] = True
        x = np.zeros((n, m))
    x = give_affectations(y, costs)
    return x, y


"""Give all the neighbors of a couple x,y, as described in the report"""
def neighborhood1(x, y, fcosts, costs):
    n, m = x.shape
    neighbors = []
    for j in range(m):
        new_y = y.copy()
        new_y[j] = 1-y[j] # opening or closing
        new_x = give_affectations(new_y, costs)
        neighbors.append((new_x, new_y))
    return neighbors

def neighborhood2(x, y, fcosts, costs):
    n, m = x.shape
    neighbors = []
    for i in range(m):
        for j in range(i+1,m):
            new_y = y.copy()
            new_y[i] = 1-y[i] # opening or closing
            new_y[j] = 1-y[j] # opening or closing
            new_x = give_affectations(new_y, costs)
            neighbors.append((new_x, new_y))
    return neighbors


""" Voisinage avec une Distance de Hamming égale à k """
def neighborhood3(x, y, fcosts, costs, k):
    """
    Génère les voisins en retournant k positions de y (indices strictement croissants).
    Pour chaque voisin, recalcule x via give_affectations(new_y, costs).

    Args:
        x: matrice d'affectation courante (n, m) — non utilisée sauf pour dimensions
        y: vecteur binaire de longueur m (ou matrice n×m si c’était ton design, voir note)
        costs: matrice de coûts (n, m)
        k: nombre de positions à flipper

    Returns:
        neighbors: liste de tuples (new_x, new_y)
    """
    n, m = x.shape
    if not (1 <= k <= m):
        return []

    neighbors = []

    for idxs in combinations(range(m), k):
        new_y = y.copy()
        l = list(idxs)
        new_y[l] = 1 - new_y[l]
        new_x = give_affectations(new_y, costs)
        neighbors.append((new_x, new_y))


    return neighbors

""" k usines aléatoires inversées : donne un seul voisin dans le voisinage """
def neighborhood3random(x, y, fcosts, costs, k):
    n, m = x.shape
    if not (1 <= k <= m):
        return []

    neighbors = []
    new_y = y.copy()
    #print(new_y)
    
    idxs = np.random.choice(m, k, replace=False)
    #print(idxs)
    new_y[list(idxs)] = 1 - new_y[list(idxs)]
    #print(new_y)

    new_x = give_affectations(new_y, costs)
    neighbors.append((new_x, new_y))

    return neighbors

""" Voisinage avec une Distance de Hamming variant de 1 à p """
def neighborhood4(x, y, fcosts, costs, p):
    n, m = x.shape
    if not (1 <= p <= m):
        return []

    neighbors = []
    for k in range(p+1):
        for idxs in combinations(range(m), k):
            new_y = y.copy()

            new_y[list(idxs)] = 1 - new_y[list(idxs)]

            new_x = give_affectations(new_y, costs)
            neighbors.append((new_x, new_y))
    return neighbors



def all_neighbors(x, y, fcosts, costs):
    neighbors1 = neighborhood1(x, y, fcosts, costs)
    neighbors2 = neighborhood2(x, y, fcosts, costs)
    return neighbors1 + neighbors2


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

# Question 3


""" Best Neighbor : find a local optimum among the neighborhood just improve the solution with the best neighbor using the neighborhood based on Hamming distance equal to k """
def best_neighbor(x,y,fcosts, costs, k, verbose=False):
    
    neighbors = neighborhood3(x, y, fcosts, costs, k)

    best_neighbor = (x,y)
    best_value = objective_function(x, y, fcosts, costs)
    #print("best_neighbor()")
    #print(y)
    #print(best_value)

    for new_x, new_y in neighbors:
        value = objective_function(new_x, new_y, fcosts, costs)
        if value < best_value:
            best_value = value
            best_neighbor = (new_x, new_y)

    if verbose:
        print("Best Neighbor : ", best_neighbor)
        print("Neightbor Value : ", best_value)

    return best_neighbor[0], best_neighbor[1], best_value

""" Descente : Meilleur voisin d'itération en itération pour une distance de Hamming égal à k donné """
def descent(x,y,fcosts, costs, k):
    ok = 1

    cur_x, cur_y = x, y
    cur_value = objective_function(x, y, fcosts, costs)

    while ok == 1:
        new_x, new_y, new_value = best_neighbor(cur_x, cur_y, fcosts, costs, k, False) 
        if new_value != cur_value :
            cur_x, cur_y = new_x, new_y
            cur_value = new_value
        else :
            ok = 0
    return cur_x, cur_y, cur_value

""" Variable Neighborhood Descent by changing Hamming distance from 1 to p """
def vns(x,y,fcosts, costs, p):
    k = 1

    cur_x, cur_y = x, y
    cur_value = objective_function(x, y, fcosts, costs)

    while k <= p:
        new_x, new_y, new_value = best_neighbor(cur_x, cur_y, fcosts, costs, k, False)
        if new_value != cur_value :
            cur_x, cur_y = new_x, new_y
            cur_value = new_value
            k = 1
        else :
            k += 1
    return cur_x, cur_y, cur_value

    




# Question 4

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
    if np.random.rand() < 0.6: # Dans le cas ou a des sol qui improvent et d'autrs qui n'improvent pas, faire : choisir une qui improve

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









def random_neighbor(x, y, fcosts, costs, p):
    k = np.random.randint(1, p + 1)

    neighborhood = neighborhood3random(x, y, fcosts, costs, k)
    neighbor = neighborhood[0]

    return neighbor



""" 
x,y généré avec test_opening
T > 0 et 0 < mu < 1 
"""
def simulated_annealing(x, y, fcosts, costs, p, T, mu, nb_iter, iter_max):
    # Meilleur solution visitée
    x_min = x
    y_min = y
    value_min = objective_function(x, y, fcosts, costs)

    # Solution courante
    x_cur = x
    y_cur = y
    value_cur = value_min

    count = 0

    while count < nb_iter:
        for h in range(iter_max):
            # Tire au sort un voisin de la solution courante
            rd_neighbor = random_neighbor(x_cur, y_cur, fcosts, costs, p)
            
            x_neighbor, y_neighbor = rd_neighbor[0], rd_neighbor[1]
            value_neighbor = objective_function(x_neighbor,y_neighbor,fcosts,costs)
            
            delta = value_neighbor - value_cur

            if delta <= 0:
                x_cur = x_neighbor
                y_cur = y_neighbor
                value_cur = value_neighbor

                if value_cur < value_min:
                    x_min = x_cur
                    y_min = y_cur
                    value_min = value_cur
            else:
                q = np.random.uniform(0, 1)
                if q <= np.exp(- delta / T):
                    x_cur = x_neighbor
                    y_cur = y_neighbor
                    value_cur = value_neighbor
        T = mu * T
        count += 1

    return x_min,y_min,value_min



def simulated_annealing2(x, y, fcosts, costs, p, T, mu, seuil, iter_max):
    # Meilleur solution visitée
    x_min = x
    y_min = y
    value_min = objective_function(x, y, fcosts, costs)

    # Solution courante
    x_cur = x
    y_cur = y
    value_cur = value_min

    k = p

    while T > seuil:
        for h in range(iter_max):
            
            # Tire au sort un voisin de la solution courante
            rd_neighbor = random_neighbor(x_cur, y_cur, fcosts, costs, k)
            
            x_neighbor, y_neighbor = rd_neighbor[0], rd_neighbor[1]
            value_neighbor = objective_function(x_neighbor,y_neighbor,fcosts,costs)
            
            delta = value_neighbor - value_cur

            if delta <= 0:
                x_cur = x_neighbor
                y_cur = y_neighbor
                value_cur = value_neighbor

                if value_cur < value_min:
                    x_min = x_cur
                    y_min = y_cur
                    value_min = value_cur
            else:
                q = np.random.uniform(0, 1)
                if q <= np.exp(- delta / T):
                    x_cur = x_neighbor
                    y_cur = y_neighbor
                    value_cur = value_neighbor
        T = mu * T

    return x_min,y_min,value_min



    

