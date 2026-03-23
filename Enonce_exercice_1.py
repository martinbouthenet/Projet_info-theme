import numpy as np
import matplotlib.pyplot as plt 

#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Import du jeu de données d'entrainement
X_train, T_train = readdataset2d("nuage_train_exercice_1") # dataset 1
# X_train, T_train = readdataset2d("forme_exercice_1") # dataset 2
N, D = X_train.shape
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1") # dataset 1
# X_test, T_test = readdataset2d("forme_test_exercice_1") # dataset 2
N, D = X_test.shape
plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

#%%Exercice 1:
#Question 1    

def sigma(x):
    return 1/(1+np.exp(-x))

def predit_classe(Y):
    return np.round(Y)

def taux_precision(C, T):
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

def cross_entropy(Y,T):
    N = len(Y)
    J = 0
    for i in range(N):
        if T[i] == 1:
            if np.log(Y[i]) == 0.0:
                continue
            else :
                J -= np.log(Y[i])
        else :
            if np.log(1-Y[i]) == 0.0:
                continue
            else :
                J -= np.log(1-Y[i])
    return J

def affichage(X, T, C):
    fig,ax = plt.subplots(1,2, figsize = (15,7))
    # Affichage des vrais classes
    ax[0].scatter(X[:,0], X[:,1], c=T, s = 40)
    ax[0].set_title("Classes réelles données par T")

    # Affichage des prédictions
    ax[1].scatter(X[:,0], X[:,1], c=C, s = 40)
    ax[1].set_title("Classes prédites données par C")
    plt.show()
    
    
def predit_proba(X, W, b):
    Z = [X]
    for i in range(len(b)-1):                
        #on a len(W)-1 Z à calculer puis il faut calculer Y avec le dernier element de W
        Z.append(sigma(Z[-1].dot(W[i] + b[i])))
    Y = sigma(Z[-1].dot(W[-1]) + b[-1])
    Z.remove(X)
    return Z, Y

def initialise(dimensions):
    W = []
    b = []
    for i in range(len(dimensions)-1):
        W.append(np.random.uniform(-2, 2, size=(dimensions[i], dimensions[i+1])))
        b.append(np.random.uniform(-0.5, 0.5, size=dimensions[i+1]))
    Z, Y = predit_proba(X_train, W, b)
    C = predit_classe(Y)
    return W, b, Z, Y, C
    
def updateWb(W, b, X, Z, Y, T, lr):
    delta = [Y-T]
    for i in range(len(W)-1):
        W[-i-1] -= lr*(Z[-i-1].transpose()).dot(delta[-1])
        b[-i-1] -= lr*sum(delta[-1])
        delta.append(np.dot(delta[-1], W[-i-1].transpose())*Z[-i-1]*(1-Z[-i-1]))
    W[0] -= lr*np.transpose(X).dot(delta[-1])
    b[0] -= lr*sum(delta[-1])
    
def reseau(W, b, X, Z, Y, T, lr=0.1, nb_iter=100, int_affiche=10):
    suite_erreur = [cross_entropy(Y,T)]
    for i in range(nb_iter):
        updateWb(W, b, X, Z, Y, T, lr)
        Z[:], Y[:] = predit_proba(X, W, b) #Sans le [:], le tableau ne serait pas modifié
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
    return suite_erreur
        

#%%Question 2:
    
dimensions_1 = [2, 3, 3, 3, 1]
        
dimensions_2 = [2, 7, 7, 7, 1]

dimensions_3 = [2, 15, 15, 1]

dimensions_4 = [2, 3, 15, 15, 1]

dimensions_5 = [2, 15, 15, 3, 1]

dimensions_6 = [2, 40, 1]

dimensions_7 = [2, 20, 20, 1]

dimensions_8 = [2, 5, 4, 4, 4, 4, 1]

W, b, Z, Y, C_init = initialise(dimensions_5)
W_init = W.copy()
b_init = b.copy()
suite_erreur = reseau(W, b, X_train, Z, Y, T_train)
C_train_final = predit_classe(Y)

print('\nSituation initiale')
print("Poids initiaux : ", W_init ,b_init)
print("Taux de précision initial= ", taux_precision(C_init, T_train))
print("Erreur d'entropie initiale :", suite_erreur[0])


print('\nRésultats')
print("Poids optimises :", W, b)
print("Erreur d'entropie finale :", suite_erreur[-1])
print("Taux de précision final = ", taux_precision(C_train_final, T_train))
affichage(X_train, T_train, C_train_final)




