import numpy as np

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
def give_affectations(costs, y):
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