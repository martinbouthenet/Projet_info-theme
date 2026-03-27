from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt

# Open the image from the working directory
image = Image.open('gr_cathedrale.png')

# Convert the image into a np.array
X_large = np.asarray(image)

# Print the information of the data
print("Format de l'image originale' : ", X_large.shape)
print("Nombre de nuances de gris : ", X_large.max())

# Affiche l'image seule
imshow(X_large,cmap=get_cmap('gray'))

# Definition d'une fonction qui affiche 2 imges cote a cote
def displayTwoImages(img1, img2):
  _, axes = plt.subplots(ncols=2)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))

# Definition d'une fonction qui affiche 3 images cote a cote
def displayThreeImages(img1, img2, img3):
  _, axes = plt.subplots(ncols=3)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
  axes[2].imshow(img3, cmap=plt.get_cmap('gray'))

def pooling_moy(X,ratio_x, ratio_y):
    Dx, Dy = X.shape
    Dx_pooled = Dx // ratio_x
    Dy_pooled = Dy // ratio_y
    X_pooled = np.zeros((Dx_pooled,Dy_pooled))
    for i in range(Dx_pooled):
        for j in range(Dy_pooled):
            X_pooled[i,j] = np.mean(X[i*ratio_x:(i+1)*ratio_x,j*ratio_y:(j+1)*ratio_y])
    return X_pooled

# Reduction de la taille de l'image X
X = pooling_moy(X_large, 40, 36)
print("Format de l'image réduite' : ", X.shape)
print("Nombre de nuances de gris de l'image réduite : ", X.max())
displayTwoImages(X_large, X)

#%% Exercice 1 : Convolution
def convolution1D(X,F):
    N = len(X)
    H = len(F)
    Z = []
    for i in range(N-H+1): #pour tout entier i de 0 à N-H 
       somme = 0
       for j in range(H):  #on somme pour h de 0 à H-1
          somme += X[i+H-(j+1)]*F[j]
       Z.append(somme)
    return Z

#%% Test des fonctions
# Definitions des donnees
X_1 = [80,0,0,0,0,0,80]
X_2 = [60,20,10,0,10,20,60]
X_3 = [10,20,30,40,60,70,80]
# Definition des filtres
F_1 = [1,2,1]
F_1_norm = [0.25,0.5,0.25]
F_2 = [-1,2,-1]
F_3 = [0,1,2]
F_3_inv = [2,1,0]

# Ces lignes permettent de tester les fonctions de convolutions
# # Convolution avec F_1
print("Convolution avec F_1 = [1,2,1] et F_1_norm = [0.25,0.5,0.25] :")
print("Convolution X_1*F_1 : ", convolution1D(X_1, F_1)) #[80, 0, 0, 0, 80]
print("Convolution X_1*F_1_norm : ", convolution1D(X_1, F_1_norm)) # [20.0, 0.0, 0.0, 0.0, 20.0]
print("Convolution X_2*F_1 : ", convolution1D(X_2, F_1)) #[110, 40, 20, 40, 110]
print("Convolution X_2*F_1_norm : ", convolution1D(X_2, F_1_norm)) #[27.5, 10.0, 5.0, 10.0, 27.5]
print("Convolution X_3*F_1 : ", convolution1D(X_3, F_1)) #[80, 120, 170, 230, 280]
print("Convolution X_3*F_1_norm : ", convolution1D(X_3, F_1_norm),'\n') #[20.0, 30.0, 42.5, 57.5, 70.0]

# # Convolution avec F_2
# print("Convolution avec F_2 = [-1,2,-1]") #[-1,2,-1]
# print("Convolution X_1*F_2 : ", convolution1D(X_1, F_2)) #[-80, 0, 0, 0, -80]
# print("Convolution X_2*F_2 : ", convolution1D(X_2, F_2)) #[-30, 0, -20, 0, -30]
# print("Convolution X_3*F_2 : ", convolution1D(X_3, F_2),'\n') #[0, 0, -10, 10, 0] 

# # Convolution avec F_3
# print("Convolution avec F_3 = [0,1,2]")
# print("Convolution X_1*F_3 : ", convolution1D(X_1, F_3)) #[160, 0, 0, 0, 0]
# print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
# print("Convolution X_3*F_3 : ", convolution1D(X_3, F_3)) #[40, 70, 100, 140, 190]
# print("Convolution X_3*F_3_inv : ", convolution1D(X_3, F_3_inv),'\n') #[80, 110, 160, 200, 230]


#%% Exercice 2 : Convolution 2D
def convolution_2D(X, F):
    Dx, Dy = X.shape
    Hx, Hy = F.shape
    Z = np.zeros((Dx-Hx+1, Dy-Hy+1))
    for i in range(Dx-Hx+1):  #pour tout entier i de 0 à Dx-Hx
           for j in range(Dy-Hy+1): #pour tout entier j de 0 à Dy-Hy
                somme = 0
                for x in range(Hx):  
                    for y in range(Hy):
                        #on somme avec x compris entre 0 et Hx-1 et y compris entre 0 et Hy-1
                        somme += X[i+Hx-(x+1),j+Hy-(y+1)]*F[x,y]  #on applique la formule
                Z[i,j] = somme      
    return Z


def applique_filtre(X, F):
    """
    Applique le filtre F donné en paramètres à l'image X donné en paramètres grâce à la fonction convolution_2D 
    puis affiche l'image de base et l'image modifiée grâce à la fonction displayTwoImages  
    """
    Z = convolution_2D(X, F)
    displayTwoImages(X, Z)

#%% Filtres à tester sur l'image X qui est obtenue par pooling l'image originale X_large

s = 5
filtre_1 = np.ones((s,s))/100

filtre_2 = np.array([[0.0625, 0.125, 0.0625],
                     [0.125, 0.25, 0.125],
                     [0.0625, 0.125, 0.0625]])

filtre_3 = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

filtre_4 = np.array([[2, 0, -2],
                     [4, 0, -4],
                     [2, 0, -2]])

filtre_5 = np.array([[0, 0, 0],
                    [-1, 1, 0],
                    [0, 0, 0]])

# Faire varier la valeur centrale entre 0 et -200
filtre_5 = np.array([[0, 1, 0],
                     [1, -200, 1],
                     [0, 1, 0]])

filtre_6 = np.array([[1, 1, 1],
                     [1, -200, 1],
                     [1, 1, 1]])


# Faire varier la valeur centrale entre 0 et 200
filtre_7 = np.array([[0, -1, 0],
                     [-1, 10, -1],
                     [0, -1, 0]])

filtre_8 = np.array([[-1, -1, -1],
                     [-1, 10, -1],
                     [-1, -1, -1]])


filtre_9 = np.array([[0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [-1, -1, 10, -1, -1],
                     [0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0]])

applique_filtre(X, filtre_1)
applique_filtre(X, filtre_2)
applique_filtre(X, filtre_3)
applique_filtre(X, filtre_4)
applique_filtre(X, filtre_5)
applique_filtre(X, filtre_6)
applique_filtre(X, filtre_7)
applique_filtre(X, filtre_8)
applique_filtre(X, filtre_9)

