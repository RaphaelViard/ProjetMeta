import numpy as np
import read_pb


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










#faire l'heuristique : on regarde en ouvrant 1 usine si le cout de construction est < a la somme des diff des couts des gens qui changent.

