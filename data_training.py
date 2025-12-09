"""
Entraînement d'un modèle de classification des émotions avec Keras

Ce script :
1. Charge les données collectées (fichiers .npy) pour différentes émotions.
2. Encode les labels en vecteurs one-hot.
3. Mélange aléatoirement les données.
4. Construit un réseau de neurones dense.
5. Entraîne le modèle et le sauvegarde.

Prérequis : fichiers .npy nommés par émotion (ex: happy.npy, sad.npy, etc.)
"""

import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model

# Flags et variables d'initialisation
is_init = False  # indique si le premier dataset a été chargé
size = -1  # taille du premier dataset

label = []  # liste des noms d'émotions trouvées
dictionary = {}  # mappage émotion -> index numérique
c = 0  # compteur pour créer les indices des émotions

# Boucle de chargement des fichiers de données (un par émotion)
# Cherche tous les fichiers .npy sauf "labels.npy"
#Il est essentiel pour transformer des fichiers séparés (angry.npy, sad.npy, etc.) en un grand tableau cohérent que Keras peut ingérer.
#Le réseau de neurones ne peut traiter que des nombres. Cette section crée la clé de traduction qui sera utilisée juste après pour convertir les noms d'émotion ('angry') en index numériques (0).
for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
		emotion_name = i.split('.')[0]  # ex: "happy" depuis "happy.npy"
		
		if not(is_init):
			# Premier dataset trouvé : initialiser X et y
			is_init = True
			X = np.load(i)
			size = X.shape[0]
			# Créer les labels avec le nom de l'émotion (ex: ["happy", "happy", ...])
			y = np.array([emotion_name]*size).reshape(-1,1)
		else:
			# Datasets suivants : concaténer à X et y existants
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([emotion_name]*size).reshape(-1,1)))

		# Ajouter l'émotion à la liste et créer son index dans le dictionnaire
		label.append(emotion_name)
		dictionary[emotion_name] = c
		c = c + 1


# Convertir les labels textes en indices numériques
# ex: "happy" -> 0, "sad" -> 1, etc.
for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convertir les indices en vecteurs one-hot    ,Le réseau de neurones que vous avez défini utilise une fonction de coût (categorical_crossentropy) et une couche de sortie (softmax) qui nécessitent un format d'étiquette très spécifique : le One-Hot Encoding.
# ex: 0 -> [1,0,0,...], 1 -> [0,1,0,...] , La fonction categorical_crossentropy mesure l'erreur en comparant la probabilité de sortie du réseau (ex: [0.1, 0.8, 0.05, 0.05]) avec le vecteur One-Hot cible (ex: [0, 1, 0, 0]). Cela permet un calcul d'erreur précis.
y = to_categorical(y)

# Copier les données pour le mélange aléatoire
X_new = X.copy()
y_new = y.copy()
counter = 0

'''Après la concaténation dans la section précédente, vos données sont triées par émotion (ex: les 100 échantillons de "colère" d'abord, puis les 100 de "tristesse", etc.). Si vous entraînez un modèle sur des données triées :

Pendant les 100 premières étapes, il n'apprend que la "colère".

Il oublie ce qu'il a appris en voyant la "tristesse".
'''
# Générer une permutation aléatoire des indices
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

# Mélanger les données en fonction de la permutation
for i in cnt:
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


# Construction du modèle de réseau de neurones
# Couche d'entrée : prend en compte le nombre de features
ip = Input(shape=(X.shape[1],))

# Couches cachées avec activation ReLU
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

# Couche de sortie avec softmax (classification multi-classe)
op = Dense(y.shape[1], activation="softmax")(m)

# Créer le modèle en liant entrée et sortie
model = Model(inputs=ip, outputs=op)

# Compiler le modèle avec optimiseur RMSprop et loss categorical crossentropy
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Entraîner le modèle sur 50 épochs
model.fit(X, y, epochs=50)


# Sauvegarder le modèle entraîné et les labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
