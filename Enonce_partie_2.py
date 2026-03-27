import numpy as np
import matplotlib.pyplot as plt

# %% Introduction : Lecture des jeux de données fournis
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1, 1))
    return np.array(X), T
# %% Fonction convertit de la question 2
def nb_classes(T):
    """
    Fonction auxiliaire permettant de compter le nombre de classes du nuage de points
    au lieu de le faire directement dans la fonction convertit
    """
    K = 0
    s_classe = set()
    for i in range(len(T)):
        if T[i, 0] not in s_classe:
            s_classe.add(T[i, 0])
            K += 1
    return K

def convertit(T, K):
    """
    Transforme et renvoie la matrice T donnée en argument pour qu'elle soit de taille N,K
    et que pour tout (i,j), T[i, j] soit égal à 1 si i est de classe j et 0 sinon
    """
    N = T.shape[0]
    T_convertit = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            if T[i, 0] == j:
                T_convertit[i, j] = 1
            else:
                T_convertit[i, j] = 0
    return T_convertit
# %% QUESTION 3
def softmax(A):
    """Fonction qui prend en argument une matrice A de taille (n,p) et 
    renvoie une matrice B tel que chaque ligne i de B soit le softmax de la ligne i de A"""
    n, p = A.shape
    B = np.zeros((n, p))
    for i in range(n):
        somme = 0
        for z in range(len(A[i, :])):
            somme += np.exp(A[i, z])
        for j in range(p):
            B[i, j] = np.exp(A[i, j]) / somme
    return B

# %% Question 4
def predit_proba(X, W, b):
    """On utilise la relation donné par l'énoncé 
    pour obtenir la matrice de prédiction Y"""
    return softmax(np.dot(X, W) + b)

# %% Question 5
def predit_classe(Y):
    """Pour obtenir la matrice C on utilise le principe énoncé en question 5"""
    n, p = Y.shape
    C = np.zeros((n, p))
    l_max = Y.max(axis=1)
    for i in range(n):
        for j in range(p):
            if Y[i, j] == l_max[i]:
                C[i, j] = 1
            else:
                C[i, j] = 0
    return C

# %% Question 6
def regression_logistique(W, b, X, Y, T, lr=0.1, nb_iter=100, int_affiche=10):
    """On applique l'algorithme de la regression logistique avec notre
    fonction predit_proba, updateWb et cross_entropy"""
    suite_erreur = []
    n = 0
    while n < nb_iter:
        Y[:] = predit_proba(X, W, b)
        updateWb(W, b, X, Y, T, lr)
        n += 1
        if n % int_affiche == 0:
            suite_erreur.append(cross_entropy(Y, T))
    return suite_erreur
    
def initialise(D, K, X):
    W = np.random.uniform(-2, 2, size=(D, K)) #On initialise la matrice W de tzille (D,K)
    b = np.random.uniform(size=K)
    Y = predit_proba(X, W, b)
    print(Y.shape)
    C_train = predit_classe(Y)
    return W, b, Y, C_train
    
def cross_entropy(Y, T):
    """On calcule l'erreur d'entropie avec la formule donné par l'énoncé"""
    J = 0
    N = T.shape[0]
    K = T.shape[1]
    for n in range(N):
        for k in range(K):
            J += T[n, k] * np.log(Y[n, k])
    return -J
    
def updateWb(W, b, X, Y, T, lr):
    Nabla_W = np.dot(np.transpose(X), (Y-T))
    W[:] = W - lr * Nabla_W
    Nabla_b = (Y-T).sum(axis=0)#On prend en compte la nouvelle dimension de b 
    b -= lr * Nabla_b

def taux_precision(Y, T):
    nb_pt = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i,j] == 1 and T[i,j] == 1:
                nb_pt += 1
    return nb_pt / len(Y)

# Exercice 3
def sigma(x):
    return 1/(1+np.exp(-x))

def softmax(A):
    n, p = A.shape
    B = np.zeros((n, p))
    for i in range(n):
        somme = 0
        for z in range(len(A[i, :])):
            somme += np.exp(A[i, z])
        for j in range(p):
            B[i, j] = np.exp(A[i, j]) / somme
    return B


def predit_classe(Y):
    n, p = Y.shape
    C = np.zeros((n, p))
    l_max = Y.max(axis=1)
    for i in range(n):
        for j in range(p):
            if Y[i, j] == l_max[i]:
                C[i, j] = 1
            else:
                C[i, j] = 0
    return C

def taux_precision(C, T):
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

def cross_entropy(Y, T):
    J = 0
    N = T.shape[0]
    K = T.shape[1]
    for n in range(N):
        for k in range(K):
            J += T[n, k] * np.log(Y[n, k])
    return -J

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
    for i in range (len(b)-1):
        Z.append(sigma(Z[-1].dot(W[i])+b[i]))
    Y = softmax(np.dot(Z[-1],W[-1])+b[-1])
    return Z, Y

def initialise(dimensions) :
    global N
    W = []
    b = []
    for p in range(len(dimensions)-1) :
        W.append(np.random.uniform(-2,2,size=(dimensions[p],dimensions[p+1])))
        b.append(np.random.uniform(-0.5,0.5,size=(dimensions[p+1])))
    Z,Y = predit_proba(X_train, W, b)
    C = predit_classe(Y)
    return W,b,Y,Z,C

def updateWb(W, b, X, Z, Y, T, lr):
    delta_a_lenvers=[Y-T]
    for i in range (len(W)-1):
        W[-1-i] -= lr*(Z[-1-i].transpose()).dot(delta_a_lenvers[-1])
        b[-1-i] -= lr*sum(delta_a_lenvers[-1])
        delta_a_lenvers.append(np.dot(delta_a_lenvers[-1], W[-1-i].transpose())*Z[-1-i]*(1-Z[-1-i]))
    W[0] -= lr*(np.transpose(X)).dot(delta_a_lenvers[-1])
    b[0] -= lr*sum(delta_a_lenvers[-1])
    
def reseau(W, b, X, Z, Y, T, lr=0.1, nb_iter=100, int_affiche=10) :
    suite_erreur = [cross_entropy(Y,T)]
    for i in range(nb_iter):
        updateWb(W ,b ,X ,Z ,Y ,T ,lr)
        Z[:], Y[:] = predit_proba(X, W, b) #Sans le [:], le tableau ne serait pas modifié
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
    return suite_erreur

# EXERCICE 2   
# %% Import du jeu de données : probleme à 4 classes
X_train, T_train = readdataset2d("probleme_4_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
plt.show()

K = nb_classes(T_train)
T = convertit(T_train, K)
W, b, Y_train, C_train_init = initialise(D, K,X_train)
W_init = W.copy()
b_init = b.copy()
suite_erreur = regression_logistique(W, b, X_train, Y_train, T, lr=0.01, int_affiche=10)
C_train_final = predit_classe(Y_train)
print('\nSituation initiale')
print("Poids initiaux : ", W_init, b_init)
print("Taux de précision initial= ", taux_precision(C_train_init, T))
print("Erreur d'entropie initiale :", suite_erreur[0])


print('\nRésultats')
print("Poids optimises :", W, b)
print("Erreur d'entropie finale :", suite_erreur[-1])
print("Taux de précision final = ", taux_precision(C_train_final, T))
print(N,K)

# %% Import du jeu de données : probleme à 5 classes
X_train, T_train = readdataset2d("probleme_5_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
plt.show()

K = nb_classes(T_train)
T = convertit(T_train, K)
W, b, Y_train, C_train_init = initialise(D, K,X_train)
W_init = W.copy()
b_init = b.copy()
suite_erreur = regression_logistique(W, b, X_train, Y_train, T, lr=0.01, int_affiche=10)
C_train_final = predit_classe(Y_train)

print('\nSituation initiale')
print("Poids initiaux : ", W_init, b_init)
print("Taux de précision initial= ", taux_precision(C_train_init, T))
print("Erreur d'entropie initiale :", suite_erreur[0])


print('\nRésultats')
print("Poids optimises :", W, b)
print("Erreur d'entropie finale :", suite_erreur[-1])
print("Taux de précision final = ", taux_precision(C_train_final, T))

x = np.linspace(-6, 6, 13)#On définit x une liste de -6 à 6 avec 13 éléments 
y = np.linspace(-6, 6, 13)
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
for i in range(K) :
    for j in range(i+1,K) :
        w = W[:,i] - W[:,j] 
        B = b[i] - b[j]
        f = - (w[0]*x + B) / (w[1])
        plt.plot(x,f)
plt.ylim(-6, 6) #On limite l'axe des y de -6 à 6
plt.show()
        
# Exercice 3 
# %% Import du jeu de données : probleme à 6 classes
X_train, T_train = readdataset2d("probleme_5_classes_dur")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
plt.show()

dimensions5 = [2,15,15,3,5]

W,b,Y,Z,C_init = initialise(dimensions5)
W_init = W.copy()
b_init = b.copy()

suite_erreur = reseau_dense(W, b, X_train, Z, Y, T_train,lr=0.001,nb_iter=10000)
C_train_final = predit_classe(Y)

print('\nSituation initiale')
print("Poids initiaux : ", W_init,b_init)
print("Taux de précision initial= ", taux_precision(C_init, T_train))
print("Erreur d'entropie initiale :", suite_erreur[0])


print('\nRésultats')
print("Poids optimises :", W, b)
print("Erreur d'entropie finale :", suite_erreur[-1])
print("Taux de précision final = ", taux_precision(C_train_final, T_train))
affichage(X_train, T_train, C_train_final)

X_train, T_train = readdataset2d("probleme_5_classes_dur")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
plt.show()
dimension_5=[2, 15, 15, 3, 5]

W,b,Y,Z,C_init = initialise(dimension_5)
W_init = W.copy()
b_init = b.copy()

T = convertit(T_train)
suite_erreur = reseau(W, b, X_train, Z, Y, T,lr=0.001,nb_iter=1000)
C_train_final = predit_classe(Y)
print(C_train_final.shape,T_train.shape,X_train.shape)

print('\nSituation initiale')
print("Poids initiaux : ", W_init,b_init)
print("Taux de précision initial= ", taux_precision(C_init, T_train))
print("Erreur d'entropie initiale :", suite_erreur[0])


print('\nRésultats')
print("Poids optimises :", W, b)
print("Erreur d'entropie finale :", suite_erreur[-1])
print("Taux de précision final = ", taux_precision(C_train_final, T_train))

