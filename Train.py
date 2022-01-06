import numpy as np
from scipy import optimize
import pickle

#Ce programme nécessite une version récente de NumPy sinon il y des erreurs lors du calcul matriciel

list_coup_joue_alea = []


def tanh(x):
    return 1 - (2 / ( (np.exp(2*x) ) + 1))

def sigmoide(x):
    return 1/(1+np.exp(-x))

def sigmoidePrime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
    
def sum(x):
    n=len(x)
    s=0
    for i in range(n):
        s=s+x[i]
        
    return s

# ------------- Reseau -------------

class Reseau:
    
    def __init__(self):
        "Cree un reseau de neurones 36-36-36"

        self.nb_I = 36  # Nombre de neurones sur la couche d'entrée
        self.nb_H1 = 36 # Nombre de neurones sur la couche cachée
        self.nb_O = 36  # Nombre de neurones sur la couche de sortie

        self.P1 = np.random.randn(self.nb_H1,self.nb_I)/2 # Premiere matrice de poids

        self.P2 = np.random.randn(self.nb_O,self.nb_H1)/2 # Seconde matrice de poids
        
    def propagation(self,entrees):
        "Execute le reseau avec une matrice 36x1 en argument"
        
        self.I = entrees.reshape(self.nb_I,1)

        self.z2 = self.P1@self.I
        self.H1 = sigmoide(self.z2)


        self.z3 = self.P2@self.H1
        self.O = sigmoide(self.z3)

        return self.O

    def fonction_objectif(self,X,y):
        "Cacul l'ecart entre le resultat obtenu et celui voulu"

        yHat = self.propagation(X)

        a = 0.5*(y.ravel() - yHat.ravel())**2

        cost = sum(a) 

        return cost

    def getParams(self):
        "Recupere tout les poids dans une seule liste"

        params = np.concatenate((self.P1.ravel(), self.P2.ravel()))
        return params

    def setParams(self, params):
        "Modifie les matrices de poids"

        P1_start = 0
        P1_end = self.nb_H1 * self.nb_I
        self.P1 = np.reshape(params[P1_start:P1_end], (self.nb_H1 , self.nb_I))

        P2_end = P1_end + self.nb_H1*self.nb_O
        self.P2 = np.reshape(params[P1_end:P2_end], (self.nb_O, self.nb_H1))

    def fonction_objectifPrime(self,X,y):
        "Calcul du gradient de la fonction objectif"

        y = y.reshape(self.nb_O,1)

        yHat = self.propagation(X)

        delta3 = np.multiply(-(y-yHat),sigmoidePrime(self.z3))

        dCdP2 = delta3 @ self.H1.T

        delta2 = np.multiply( ( self.P2.T @ delta3), sigmoidePrime(self.z2))

        dCdP1 = delta2 @ self.I.T 



        return dCdP1, dCdP2


    def gradient(self, X, y):
        "Renvoie le gradient"
        dCdP1, dCdP2 = self.fonction_objectifPrime(X,y)
        return np.concatenate((dCdP1.ravel(), dCdP2.ravel()))


 # ------------ Jeu ----------------

class BatailleNavale:

    def __init__(self,taille):
        "Cree une partie de bataille navale"

        self.s = taille
        self.plateau = np.zeros((self.s,self.s))
        self.plateauvisible = np.zeros((self.s,self.s))
        bateaux = [3,4,4]

        for bat in range(3):
            ver = True


            while ver:

                ver = False

                sens = np.random.randint(2)
                xrand = np.random.randint(self.s-bateaux[bat]*(1-sens))
                yrand = np.random.randint(self.s-bateaux[bat]*sens)
                coor = (xrand,yrand)

                for pbat in range(bateaux[bat]):

                    if self.plateau[ pbat*(1-sens)+coor[0] ][ pbat*sens + coor[1] ]:
                        ver = True

            for pbat in range(bateaux[bat]):

                self.plateau[ pbat*(1-sens)+coor[0] ][ pbat*sens + coor[1] ] = 1

    


    def torpille(self,coor):


        if self.plateauvisible[coor[1]][coor[0]]:
            return False
        else:    
            if self.plateau[coor[1]][coor[0]]:
                self.plateauvisible[coor[1]][coor[0]] = 1

            else:
                self.plateauvisible[coor[1]][coor[0]] = -1
            return True


# ------------ Entraineur ----------------


class Trainer:

    def __init__(self, N):

        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.fonction_objectif(X,y)
        grad = self.N.gradient(X,y)
        return cost, grad
        
    def train(self, X, y):
        self.X = X
        self.y = y


        
        params0 = self.N.getParams()

        options = {'maxiter': 1, 'disp' : True}
        res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(res.x)
        self.optimizationResults = res





 # --------- Tri -----------------

def partition(L,g,d):
    i=g-1
    for j in range(g,d):
        if L[j][1]<=L[d][1]:
            i=i+1
            L[j],L[i]=L[i],L[j]
    L[i+1],L[d]=L[d],L[i+1]
    return i+1

def trirec(L,g,d) :
    if g<d :
        a=partition(L,g,d)
        trirec(L,g,a-1)
        trirec(L,a,d)
def trirapide(L):
    L=trirec(L,0,len(L)-1)
    return L

def findmax(tableau):
    imax = 0
    for i,el in enumerate(tableau):
        if el > tableau[imax]:
            imax = i
    return imax






# ------------ Entrainement du reseau  ----------------


NN = Reseau()

# Lignes en commentaire permettant de restaurer un réseau pour reprendre l'entraînement
# with open('donnees3','rb') as save_reseau:
#     mon_depickler = pickle.Unpickler(save_reseau)
#     NN = mon_depickler.load()

T = Trainer(NN)

nbtr = 36


def exemples():
    global nbtr
    nbtour = 0
    nbcoup = 0
    while nbcoup < nbtr:
        nbtour += 1
        if nbtour > 100:
            nbtour = 0
            nbtr += -1


        jeuBN = BatailleNavale(6)
        historique = [[jeuBN.plateauvisible],jeuBN.plateau]
        list_coup_joue = []

        while sum(list_coup_joue) != 11:

                O = NN.propagation(jeuBN.plateauvisible.ravel())
                l = [i for i in enumerate(O.ravel())]
                trirapide(l)
                a=1
                nb_coord = l[-a][0]
                coord = (nb_coord%6,nb_coord//6)  #divmod
                while not jeuBN.torpille(coord):
                    a += 1
                    nb_coord = l[-a][0]
                    coord = (l[-a][0]%6,l[-a][0]//6)  #divmod
                #print(coord,l)
                list_coup_joue.append(jeuBN.plateau[ coord[1] ][ coord[0] ])
                #print(list_coup_joue[-1])
                historique[0].append(jeuBN.plateauvisible.copy())
        nbcoup = len(list_coup_joue)


    a = np.random.randint(1,len(list_coup_joue)-1)
    print(list_coup_joue[a],len(list_coup_joue),nbtr,nbtour)

    T.train(historique[0][a].ravel(),historique[1])
    print(historique[0][a])




for k in range(300):
    for _ in range(100):
        exemples()

    with open('reseau 2 h36 '+str(k+1)+'00','wb') as save_reseau: # en cas d'erreur indiquer le chemin complet à la place de 'reseau 2 h36' (exemple : 'C:/Projetinfo/reseau 2 h36')
        mon_pickler = pickle.Pickler(save_reseau)
        mon_pickler.dump(NN)

