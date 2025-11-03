import numpy as np
from src.utils import *
from itertools import combinations


# Toutes les heuristiques, voisinages et métaheuristique sont présents.

# Question 1

""" 
À partir d'un ensemble d'usines ouvertes 'y', on renvoie les meilleures affectations 'costs' pour les clients 
Retour : Clients affectés 'x'
"""
def give_affectations(y, costs):
    n, m = costs.shape
    x = np.zeros((n,m))
    min_idx = np.argmin(np.where(y==1, costs, np.inf), axis=1)
    x[np.arange(n), min_idx] = 1
    return x

""" 
Heuristique pour trouver une 1ère solution réalisable 
On essaye d'ouvrir chaque site en comparant les coûts d'ouverture avec le gain dû aux changement. 
Retour : Variables 'x' et 'y'
"""
def test_opening(n,m, fcosts, costs):
    x = np.zeros((n,m))
    y = np.zeros(m)
    current_service_costs = np.full(n, np.Inf)
    for j in range(m): # But : comparer cout de construction de l'usine i, avec gains associés  aux changements de clients vers la nouvelle usine
        changes_id = np.where(current_service_costs > costs[:,j]) # les couts meilleurs avec la nouvelle potentielle usine
        gains_clients = np.sum(current_service_costs[changes_id]-costs[changes_id, j]) #la valeur de ce qu'on gagne en cout de service (>0)
        if gains_clients >  fcosts[j]: # on construit bien la nouvelle usine
            y[j]=1
            x[changes_id,:] =  0
            x[changes_id,j] = 1
            current_service_costs[changes_id] = costs[changes_id,j]
    return x,y

# Question 2

"""
Voisinage 1 : Utilise la distance de Hamming égale à 1
Retour : Voisinage sous forme d'une liste
"""
def neighborhood1(x, y, fcosts, costs):
    n, m = x.shape
    neighbors = []
    for j in range(m):
        new_y = y.copy()
        new_y[j] = 1-y[j] # opening or closing
        new_x = give_affectations(new_y, costs)
        neighbors.append((new_x, new_y))
    return neighbors

"""
Voisinage 2 : Utilise la distance de Hamming égale à 2
Retour : Voisinage sous forme d'une liste
"""
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

"""
Voisinage 3 : Utilise la distance de Hamming égale à k
Retour : Voisinage sous forme d'une liste
"""
def neighborhood3(x, y, fcosts, costs, k):
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

"""
Voisinage 3 bis : Construit aléatoirement un voisin avec une distance de Hamming égale à 'k'
Retour : Voisinage sous forme d'une liste contenant un seul voisin

Fonction utilisé dans le recuit simulé
"""
def neighborhood3random(x, y, fcosts, costs, k):
    n, m = x.shape
    if not (1 <= k <= m):
        return []

    neighbors = []
    new_y = y.copy()
    
    idxs = np.random.choice(m, k, replace=False)
    new_y[list(idxs)] = 1 - new_y[list(idxs)]

    new_x = give_affectations(new_y, costs)
    neighbors.append((new_x, new_y))

    return neighbors

"""
Voisinage 4 : Utilise la distance de Hamming égale à 1 jusqu'à 'p'
Retour : Voisinage sous forme d'une liste
"""
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


# Question 3


""" 
On cherche le meilleur voisin à partir d'une solution initiale 'x', 'y' en utilisant la distance de Hamming égale à 'k'
Retour : Variables 'x_min', 'y_min' et la mailleur valeur rencontrée 'value_min' 
"""
def best_neighbor(x,y,fcosts, costs, k, verbose=False):
    
    neighbors = neighborhood3(x, y, fcosts, costs, k)

    best_neighbor = (x,y)
    best_value = objective_function(x, y, fcosts, costs)

    for new_x, new_y in neighbors:
        value = objective_function(new_x, new_y, fcosts, costs)
        if value < best_value:
            best_value = value
            best_neighbor = (new_x, new_y)

    if verbose:
        print("Best Neighbor : ", best_neighbor)
        print("Neightbor Value : ", best_value)

    return best_neighbor[0], best_neighbor[1], best_value

""" 
Descente : Meilleur voisin d'itération en itération pour une distance de Hamming égal à k donné 
Retour : Variables 'cur_x', 'cur_y' et la meilleur valeur rencontrée 'cur_value' 
"""
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

""" 
VNS : Variable Neighborhood Descent 
En faisant varier la distance de Hamming de 1 à p 
Retour : Variables 'cur_x', 'cur_y' et la meilleur valeur rencontrée 'cur_value' 
"""
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

"""
Génère un voisin aléatoire à partir d'une solution initiale 'x', 'y' avec un distance de Hamming de 1 jusqu'à p
Retour : Voisin 'neighbor'
"""
def random_neighbor(x, y, fcosts, costs, p):
    k = np.random.randint(1, p + 1)

    neighborhood = neighborhood3random(x, y, fcosts, costs, k)
    neighbor = neighborhood[0]

    return neighbor



""" 
Pré-requis :
x,y généré avec test_opening
T > 0 et 0 < mu < 1 

Recuit Simulé 
Retour : Variables 'x_min', 'y_min' et la meilleur valeur rencontrée 'value_min' 
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


    

