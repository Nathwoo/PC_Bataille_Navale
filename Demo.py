from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MultipleLocator

    ## creation fenetre principale ##
fen = Tk()
fen.title('bataille navale')
fen['bg']='cornflowerblue'

def joue(event):
    if sum(list_coup_joue) == 11:
        canevas.create_text(123,280,text="fini en "+str(len(list_coup_joue))+" tours",fill="yellow",font=("Arial,24"))
    else:
        if event.y/41 < 6:
            joueurjoue(event)
        elif event.y/41 < 8.5:
            ordijoue()
        else:
            color_bas(jeuBN.plateau) 


def joueurjoue(event):
    global list_coup_joue
    coor1 = int(event.x/41),int(event.y/41)
    if jeuBN.torpille(coor1):
        list_coup_joue.append(jeuBN.plateau[ coor1[1] ][ coor1[0] ])
        color_haut(jeuBN.plateauvisible)



fen.bind("<ButtonPress-1>", joue)

   ## creation de la fenetre de grille de jeu ##
nc = 6 #nbr de colonne
nl = 2*nc #nbr de lignes
larg = nc*41+4 #largeur de la fenetre
haut = larg*2+100 #hauteur de la fenetre

canevas = Canvas(fen, width=larg, height=haut, background='black') # création du canvas


      ## fonction creation grille de jeu ##
def createG(x0,y0,x1,y1,L):
    for i in range(nc):
        for j in range(nc):

            dx,dy = 41*i,41*j  #deplacement des coordonnées
            case = canevas.create_rectangle(x0 +dx ,y0+ dy,x1+ dx,y1 + dy, fill='white')
            L.append(case)
            

      ## création grille joueur ##
x0,y0 = 4,4 #coordonnees des abscisses premier carré
x1,y1 = 43,43 #coordonnees des ordonnées premier carré
J=[]  #liste des tag du joueur
createG(x0,y0,x1,y1,J)


      ## création grille IA ##
x0,y0 = 4,larg+100 #coordonnees des abscisses premier carré
x1,y1 = 43,larg+139 #coordonnees des ordonnées premier carré
E = []  #liste de stag de l'IA
createG(x0,y0,x1,y1,E)




        ## fonctions de base ##

def tanh(x):
    return 1 - (2 / ( (np.exp(2*x) ) + 1))

def sigmoide(x):
    return 1/(1+np.exp(-x))


    
def sum(x):
    n=len(x)
    s=0
    for i in range(n):
        s=s+x[i]
        
    return s

        ## Reseau ##

class Reseau:
    
    def __init__(self):

        self.nb_I = nc**2
        self.nb_H1 = 36
        self.nb_O = nc**2

        self.P1 = np.random.randn(self.nb_H1,self.nb_I)
        self.P2 = np.random.randn(self.nb_O,self.nb_H1)
        
    def propagation(self,entrees):
        
        self.I = entrees

        self.H1 = sigmoide(np.dot(self.P1,self.I))

        self.O = sigmoide(np.dot(self.P2,self.H1))

        return self.O

            ## Jeu ##

class BatailleNavale:

    def __init__(self,taille):


        self.s = taille
        self.plateau = np.zeros((self.s,self.s))
        self.plateauvisible = np.zeros((self.s,self.s))
        bateaux = [4,4,3]

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

    
            ## tir sur une case ##

    def torpille(self,coor):


        if self.plateauvisible[coor[1]][coor[0]]:
            return False
        else:    
            if self.plateau[coor[1]][coor[0]]:
                self.plateauvisible[coor[1]][coor[0]] = 1

            else:
                self.plateauvisible[coor[1]][coor[0]] = -1
            return True


            ## Tri ##

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


            ## Colorier grille du bas ##
            
def color_bas(x):
    for i in range (len(x)):
        for j in range(len(x)):
            if x[j][i]==1:
                for w in canevas.find_withtag(E[j+nc*i]):  
                    canevas.itemconfigure(w, fill='red2')
            elif x[j][i]==-1:
                for w in canevas.find_withtag(E[j+nc*i]):  
                    canevas.itemconfigure(w, fill='dodgerblue')
                    
                    
                    
                ## Colorier grille du haut ##       
                         
def color_haut(x):
    for i in range (len(x)):
        for j in range(len(x)):
            if x[j][i]==1:
                for w in canevas.find_withtag(J[j+nc*i]):  
                    canevas.itemconfigure(w, fill='red2')
            elif x[j][i]==-1:
                for w in canevas.find_withtag(J[j+nc*i]):  
                    canevas.itemconfigure(w, fill='dodgerblue')


def findmax(tableau):
    imax = 0
    for i,el in enumerate(tableau):
        if el > tableau[imax]:
            imax = i
    return imax

jeuBN = BatailleNavale(nc)

print(jeuBN.plateau)

print(jeuBN.plateauvisible)


NN = Reseau()
with open('reseau 2 h36 3500','rb') as save_reseau: # en cas d'erreur indiquer le chemin complet à la place de 'reseau 2 h36 3500' (exemple : 'C:/Projetinfo/reseau 2 h36 3500')
    mon_depickler = pickle.Unpickler(save_reseau)
    NN = mon_depickler.load()

l = [i for i in enumerate(NN.propagation(jeuBN.plateauvisible.ravel()))]


trirapide(l)
list_coup_joue = []

def ordijoue():
    global list_coup_joue

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
    color_haut(jeuBN.plateauvisible)


            ## afficher les fenetres ##
canevas.pack()

fen.mainloop()