
"""
Collecte de données de landmarks (visage et mains) avec MediaPipe

Ce script ouvre la caméra, extrait les landmarks du visage et des mains,
et enregistre des vecteurs de features relatifs dans un fichier NumPy.

Usage:
1. Lancez le script.
2. Entrez un nom pour le fichier de sortie (ex: "happy", "sad").
3. Le script collecte 100 échantillons puis sauvegarde `{name}.npy`.

Les vecteurs contiennent des coordonnées x et y relatives par rapport
à certains points de référence (par exemple landmark 1 pour le visage,
et landmark 8 pour les mains) afin de normaliser la position.

MediaPipe pour la détection des landmarks (points clés), 
NumPy pour la manipulation des tableaux (données d'entraînement), et
 OpenCV (cv2) pour la gestion de la caméra et de la fenêtre d'affichage.
"""

import mediapipe as mp
import numpy as np
import cv2

# Ouvre la première caméra disponible (index 0)
cap = cv2.VideoCapture(0)

# Nom du fichier de sortie fourni par l'utilisateur
name = input("Enter the name of the data : ")

# Initialisation des modules MediaPipe
holistic = mp.solutions.holistic  #détecter et de suivre simultanément les points clés (landmarks) du visage et des mains en temps réel.
hands = mp.solutions.hands  #Je prépare les informations sur les connexions spécifiques aux mains pour le dessin.
holis = holistic.Holistic() #Je charge et active le modèle d'IA qui fera le travail de détection.
drawing = mp.solutions.drawing_utils #Je prépare l'outil qui va afficher les résultats de la détection à l'écran.

# Liste pour stocker tous les vecteurs de caractéristiques
X = []
data_size = 0  # compteur d'échantillons collectés

# Boucle principale de capture
while True:
	# Liste temporaire pour stocker les coordonnées d'un seul échantillon
	lst = []

	# Lecture d'une frame depuis la caméra
	_, frm = cap.read()

	# Miroir de l'image pour correspondre à l'orientation webcam habituelle
	frm = cv2.flip(frm, 1)

	# Traitement MediaPipe (convertir en RGB avant) , envoie l'image brute de la caméra au modèle d'intelligence artificielle de MediaPipe pour la détection des points clés.
	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

	# Si des landmarks du visage sont détectés, extraire les coordonnées
	if res.face_landmarks:
		# Pour chaque landmark du visage, on enregistre la position relative
		# par rapport au landmark 1 (comme référence) — x puis y
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		# Main gauche: si présente, ajouter coordonnées relatives, sinon 42 zéros
		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			# 21 landmarks * 2 (x,y) = 42 valeurs de remplissage
			for i in range(42):
				lst.append(0.0)

		# Main droite: si présente, ajouter coordonnées relatives, sinon 42 zéros
		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		# Ajouter l'échantillon à la liste principale et incrémenter le compteur
		X.append(lst)
		data_size = data_size + 1


	# Dessiner les landmarks sur la frame pour retour visuel
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	# Afficher le nombre d'échantillons collectés sur la fenêtre
	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	# Afficher la fenêtre de la caméra
	cv2.imshow("window", frm)

	# Quitter si l'utilisateur appuie sur Échap (27) ou si on a 100 échantillons
	if cv2.waitKey(1) == 27 or data_size > 99:
		cv2.destroyAllWindows()
		cap.release()
		break


# Sauvegarder les données collectées dans un fichier NumPy
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)
