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
    fonction auxiliaire permettant de compter le nombre de classes du nuage de points
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
    return softmax(np.dot(X, W) + b)

# %% Question 5
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

# %% Question 6
def regression_logistique(W, b, X, Y, T, lr=0.1, nb_iter=100, int_affiche=10):
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
    W = np.random.uniform(-2, 2, size=(D, K))
    b = np.random.uniform(size=K)
    Y = predit_proba(X, W, b)
    print(Y.shape)
    C_train = predit_classe(Y)
    return W, b, Y, C_train


def cross_entropy(Y, T):
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
    Nabla_b = (Y-T).sum(axis=0)
    b -= lr * Nabla_b

def taux_precision(C, T):
    nb_pt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i,j] == T[i,j]:
                nb_pt += 1
    return nb_pt / len(C)


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

# %% Import du jeu de données : probleme à 6 classes
X_train, T_train = readdataset2d("probleme_5_classes_dur")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:, 0], X_train[:, 1], c=T_train, s=30)
plt.show()
