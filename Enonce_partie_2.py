import numpy as np
import matplotlib.pyplot as plt 

#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            print(x)
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Fonction convertit de la question 2
def convertit(T):
    pass



#%% Import du jeu de données : probleme à 4 classes
X_train, T_train = readdataset2d("probleme_4_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()


#%% Import du jeu de données : probleme à 5 classes
X_train, T_train = readdataset2d("probleme_5_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()

#%% Import du jeu de données : probleme à 6 classes
X_train, T_train = readdataset2d("probleme_5_classes_dur")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
plt.show()

