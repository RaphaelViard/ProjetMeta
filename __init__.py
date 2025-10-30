import numpy as np
import read_pb



m, n, fcosts, caps, costs = read_pb.read_orlib_ufl("/ProjetMeta/uncap_data/cap71.txt")
print(f"Facilities: {m}, Customers: {n}")
#print("Fixed costs:", fcosts)
#print("Service cost matrix shape:", costs.shape)
#print("First customer cost vector:", costs[0])
fcosts = np.array(fcosts)
caps = np.array(caps)
costs = np.array(costs)

#m : Nb emplacements
#n : Nb clients
#fcosts : couts de construction usine (fj)
#costs[i,j] : couts affectaction client i a usine j (cij)
#caps : capacités (cj)

#Tabou search, recuit, algo génétique
#Definir les voisinages
#taille des couts douvertures par rapport aux couts d'affection
#tres bonne heuristique gloutonne/mauvaise recherche
#ou l'inverse
#si je ferme une usine x : comparer cout fermeture usine VS cout des autres d'aller vers la plus proche
#heuristiques de base : pour chaque client : regarder cout de le relier a la meilleur usine deja existante, ou de créer usine proche et de le relier a cette derniere
#si je rajoute une usine, qui aura envie de changer d'usine, la réduction de cout associée
#Sans capacité : juste savoir quelle usine on va ouvrir.

#Les heuristiques du prof : on part de tt fermé puis on ouvre, ou l'inverse
#Créer les voisinanges : j'ouve 1 usine, j'en ferme une, je fais les 2
#Alterner ouv/ferm dans le voisinage.




def is_feasible(x, y):
    n,m = x.shape
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


x = np.zeros((n,m))
y = np.zeros(m)

def basic_feas(n,m): #on ouvre l'usine 1
    x = np.zeros((n,m))
    y = np.zeros(m)

    y[0]=1
    for i in range(x.shape[0]):
        x[i,0] = 1
    return x,y

def test_opening(n,m, fcosts, costs):
    x = np.zeros((n,m))
    y = np.zeros(m)
    current_service_costs = np.full(n, np.Inf)
    for j in range(m): #But : comparer cout de construction de l'usine i, avec gains associés  aux changements de clients vers la nouvelle usine
        changes_id = np.where(current_service_costs > costs[:,j]) # les couts meilleurs avec la nouvelle potentielle usine
        gains_clients = np.sum(current_service_costs[changes_id]-costs[changes_id, j]) #la valeur de ce qu'on gagne en cout de service (>0)
        print(len(changes_id[0]))
        if gains_clients >  fcosts[j]: #on construit bien la nouvelle usine
            y[j]=1
            x[changes_id,:] =  0
            x[changes_id,j] = 1
            current_service_costs[changes_id] = costs[changes_id,j]
    return x,y


def voisinage

x,y = test_opening(n,m,fcosts,costs)
#x,y = basic_feas(n,m)

print(is_feasible(x,y))
print(objective_function(x, y, fcosts, costs))




#faire l'heuristique : on regarde en ouvrant 1 usine si le cout de construction est < a la somme des diff des couts des gens qui changent.

