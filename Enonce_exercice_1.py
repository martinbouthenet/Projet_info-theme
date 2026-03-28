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
#X_train, T_train = readdataset2d("forme_train_exercice_1") # dataset 2
N, D = X_train.shape
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1") # dataset 1
#X_test, T_test = readdataset2d("forme_test_exercice_1") # dataset 2
N, D = X_test.shape
plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

#%% Définition du réseau de neurones
def sigma(x):
    return 1/(1+np.exp(-x))

def predit_classe(Y):
    '''
    Utilisée pour transformer une matrice ligne de coefficients
    compris entre 0 et 1 en une matrice ligne de même taille contenant 
    des 0 et des 1 en arrondissant la matrice de départ
    '''
    return np.round(Y)

def taux_precision(C, T):
    '''
    prends en argument deux matrices de même taille et renvoi en 
    pourcentage, le nombre de coefficients égaux
    '''
    assert len(C) == len(T), 'les matrices C et T sont de tailles différentes'
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

def cross_entropy(Y,T):

    '''
    utilisée pour renvoyer la matrice d'erreur d'entropie à partir
    des matrices des prédictions et des classes réelles prises en entrée
    '''
    assert len(Y) == len(T), 'les matrices Y et T sont de tailles différentes'
    N = len(Y)
    J = 0
    for i in range(N):
        if T[i] == 1:   #Si T[i]=1, 1-T[i]=0 donc il faut enlever seulement ln(Y[i])
            
            if Y[i] == 0.0:   #On passe si Y[i]=0 car on ne pourra pas prendre le ln
                continue
            else :
                J -= np.log(Y[i])
        else :    # si T[i] n'est pas 1 c'est qu'il vaut 0 car T ne contient que des 0 et des 1, il faut alors enlever ln(1-Y[i])
            if Y[i] == 1.0: # de meme on passe si 1-Y[i]=0
                continue
            else :
                J -= np.log(1-Y[i])
    return J

def affichage(X, T, C):
    '''
    Prends en paramètres les points, leur classe et leur prédiction et
    affiche à coté les points avec leur réele classe et les points avec
    leur classe prédite
    '''
    fig,ax = plt.subplots(1,2, figsize = (15,7))
    # Affichage des vrais classes
    ax[0].scatter(X[:,0], X[:,1], c=T, s = 40)
    ax[0].set_title("Classes réelles données par T")

    # Affichage des prédictions
    ax[1].scatter(X[:,0], X[:,1], c=C, s = 40)
    ax[1].set_title("Classes prédites données par C")
    plt.show()

def predit_proba(X, W, b):
    '''
    Prends en paramètres X la liste des points, W la liste de tout les
    paramètres Wi et b la liste de tout les paramètres bi.
    Renvoie Z la liste de toutes les données intermédiares Zi et 
    Y la prédiction de la classe des points X avec les paramètres W et b
    '''
    Z = [X]     #On met X au début de Z pour pouvoir itérer sur Z et commencer par X
    for i in range (len(b)-1):
        Z.append(sigma(Z[-1].dot(W[i])+b[i]))
    Y = sigma(Z[-1].dot(W[-1]) + b[-1])
    Z.remove(X)     #On suprimme le X qu'on avait rajouté pour le programme mais qui n'a pas lieu d'être là
    return Z, Y

def initialise(dimensions):
    '''
    Avec la liste dimensions, on défini des liste W et b aléatoires 
    avec des Wi et bi de la bonne taille et avec des scalaires entre
    -2 et 2 pour les Wi et entre -0.5 et 0.5 pour les bi
    On aurait pu prendre en argument X pour ne pas avoir à faire la 
    distinction entre X_train et X_test au sein de la fonction
    '''
    # Initialise les poids W et b
    W=[]
    b=[]
    for i in range (len(dimensions)-1):
        W.append(np.random.uniform(-2, 2, size=(dimensions[i],dimensions[i+1])))
        b.append(np.random.uniform(-0.5, 0.5, size=(dimensions[i+1])))


    # Initialisation les prediction Y et C on pourra passer de X_train à 
    #X_test en changeant la ligne commentée
    
    #Z, Y = predit_proba(X_train, W, b)
    Z, Y = predit_proba(X_test, W, b)
    C = predit_classe(Y)
    return W, b, Z, Y, C

# def updateWb(W, b, X, Z, Y, T, lr):
#     '''
#     Prends en paramètre W et b et les fait évoluer à l'aide de la
#     formule d'entrainement grace aux classes, prédictions, données
#     intermédiaires et d'un learning rate'
#     '''
#     delta_a_lenvers=[Y-T]    #On initialise une liste des deltas
#     # On choisit ci-après de définir i de manière croissante puis de parcourir les liste de la fin vers le début
#     for i in range (len(W)-1):    #Pour tout W sauf le premier terme qui est différent
#         #On fait évoluer les W et les b:
#         W[-1-i] -= lr*(Z[-1-i].transpose()).dot(delta_a_lenvers[-1])    
#         b[-1-i] -= lr*sum(delta_a_lenvers[-1])
#         #delta change de valeur donc on ajoute sa nouvelle valeur dans la liste des deltas
#         delta_a_lenvers.append(np.dot(delta_a_lenvers[-1], W[-1-i].transpose())*Z[-1-i]*(1-Z[-1-i]))
#     W[0] -= lr*(np.transpose(X)).dot(delta_a_lenvers[-1])
#     b[0] -= lr*sum(delta_a_lenvers[-1])

def updateWb(W, b, X, Z, Y, T, lr):
    '''
    Prends en paramètre W et b et les fait évoluer à l'aide de la
    formule d'entrainement grace aux classes, prédictions, données
    intermédiaires et d'un learning rate'
    '''
    delta = Y-T    #On initialise delta
    # On choisit ci-après de définir i de manière croissante puis de parcourir les liste de la fin vers le début
    for i in range (len(W)-1):    #Pour tout W sauf le premier terme qui est différent
        #On fait évoluer les W et les b:
        W[-1-i] -= lr*(Z[-1-i].transpose()).dot(delta)    
        b[-1-i] -= lr*sum(delta)
        #On change la valeure de delta pour modifier les autres W et b
        delta = np.dot(delta, W[-1-i].transpose())*Z[-1-i]*(1-Z[-1-i])
    
    # On finit par modifier les premières valeurs qui ne marchent pas
    # exactement de la même manière car celle de W évolue avec la 
    # première classe qui est X et non plus une donnée intermédiaire Z
    
    W[0] -= lr*(np.transpose(X)).dot(delta)
    b[0] -= lr*sum(delta)
    
def reseau(W, b, X, Z, Y, T, lr=0.1, nb_iter=100, int_affiche=10):
    '''
    Cette fonction fais fonctionner toute la prédiction en liant les autres fonctions du programme.
    Elle met nb_iter fois à jour les poids W et b donnés en paramètres, ainsi que les prédictions.
    En plus des mises à jour, cette fonction calcule l'erreur d'entropie à chaque itération et l'ajoute à la liste suite_erreur, qu'elle renvoie ensuite.
    Toutes les int_affiche itérations, la fonction affiche l'erreur d'entropie calculée.
    '''
    suite_erreur = [cross_entropy(Y,T)]
    for i in range(nb_iter):
        updateWb(W ,b ,X ,Z ,Y ,T ,lr)
        Z[:], Y[:] = predit_proba(X, W, b) #Sans le [:], le tableau ne serait pas modifié
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
    return suite_erreur

#%% Question 2

dimension_1=[2, 3, 3, 3, 1]
dimension_2=[2, 7, 7, 7, 1]
dimension_3=[2, 15, 15, 1]
dimension_4=[2, 3, 15, 15, 1]
dimension_5=[2, 15, 15, 3, 1]
dimension_6=[2, 40, 1]
dimension_7=[2, 20, 20, 1]
dimension_8=[2,5,4,4,4,4,1]
dimensions=[dimension_1,dimension_2,dimension_3,dimension_4,dimension_5,dimension_6,dimension_7,dimension_8]

#On teste ici toutes les dimensions en même temps en faissant varier à la main les paramètres
#On peut passer des données train à test en changeant les lignes commentées
for i in range (8):
    W, b, Z_train, Y_train, C_train_init = initialise(dimensions[i])
    W_init = W.copy()
    b_init = b

    #suite_erreur = reseau(W, b, X_train, Z_train, Y_train, T_train, lr=0.0005, nb_iter = 10000, int_affiche=100)
    suite_erreur = reseau(W, b, X_test, Z_train, Y_train, T_test, lr=0.0006, nb_iter = 10000, int_affiche=100)
    C_train_final = predit_classe(Y_train)

    print('\nSituation initiale')
    print("Poids initiaux : ", W_init,b_init,)
    #print("Taux de précision initial= ", taux_precision(C_train_init, T_train))
    print("Taux de précision initial= ", taux_precision(C_train_init, T_test))
    print("Erreur d'entropie initiale :", suite_erreur[0])


    print('\nRésultats')
    print("Poids optimises :", W, b)
    print("Erreur d'entropie finale :", suite_erreur[-1])
    #print("Taux de précision final = ", taux_precision(C_train_final, T_train))
    print("Taux de précision final = ", taux_precision(C_train_final, T_test))
    #affichage(X_train, T_train, C_train_final)
    affichage(X_test, T_test, C_train_final)
