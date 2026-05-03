# 🔬 TP6 – Détection et Suivi du Mouvement dans une Séquence d'Images

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.13-green?logo=opencv" />
  <img src="https://img.shields.io/badge/NumPy-2.4-orange" />
  <img src="https://img.shields.io/badge/Matplotlib-3.10-red" />
  <img src="https://img.shields.io/badge/Statut-Terminé ✅-brightgreen" />
</p>

> **Matière :** Vision Artificielle | **Filière :** GBM2  
> **Enseignante :** Dr. Imene OUALI  
> **Institut :** Institut National des Technologies et des Sciences du Kef – Université de Jendouba  
> **Année universitaire :** 2025-2026

---

## 📋 Description

Ce TP implémente un **pipeline complet de détection et de suivi du mouvement** dans une séquence d'images biomédicales simulant le déplacement d'une cellule en microscopie.

```
Images → Différence → Seuil → Morphologie → Détection → Tracking → Analyse
```

---

## 🗂️ Contenu du dépôt

```
📦 D-tection-et-suivi-du-mouvement/
│
├── 📄 TP6_detection_suivi_mouvement.py   ← Code principal complet (10 parties)
├── 📄 README.md                          ← Ce fichier
│
└── 📁 captures/                          ← Générées automatiquement à l'exécution
    ├── partie1_lecture.png
    ├── partie2_difference.png
    ├── partie3_seuillage.png
    ├── partie4_morphologie.png
    ├── partie5_detection.png
    ├── partie8_trajectoire.png
    ├── partie8_trajectoire2D.png
    └── partie9_vitesse.png
```

---

## ⚙️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/boualin902-lab/D-tection-et-suivi-du-mouvement.git
cd D-tection-et-suivi-du-mouvement
```

### 2. Installer les dépendances
```bash
pip install opencv-python numpy matplotlib
```

| Bibliothèque | Version testée |
|---|---|
| Python | 3.14 |
| opencv-python | 4.13.0 |
| numpy | 2.4.4 |
| matplotlib | 3.10.8 |

---

## 🚀 Utilisation

```bash
python TP6_detection_suivi_mouvement.py
```

Le script génère automatiquement une séquence synthétique de 15 images, puis exécute les 10 parties du TP.

---

## 🔬 Pipeline détaillé – Les 10 parties

### Partie 1 – Lecture de la séquence
Chargement de 15 images en niveaux de gris (256×256 px) simulant une cellule en déplacement diagonal sur fond bruité.

### Partie 2 – Différence absolue entre deux images
```python
diff = cv2.absdiff(img2, img1)
```
Les **zones claires** de l'image résultante indiquent les pixels qui ont changé, donc les zones de mouvement.

### Partie 3 – Seuillage
```python
_, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
```
Transformation en masque binaire. Seuils testés : 10, 20, 30, 40. **Seuil retenu : 25** (meilleur équilibre signal/bruit).

### Partie 4 – Nettoyage morphologique
```python
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # supprime le bruit
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)  # comble les trous
```

### Partie 5 – Détection de l'objet
```python
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
```

### Partie 6 – Centre par moments
```python
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])
```

### Partie 7 – Tracking sur toute la séquence
Répétition du pipeline frame par frame → stockage des 14 positions détectées.

### Partie 8 – Trajectoire
Tracé de la trajectoire sur la dernière frame + graphe 2D des positions.

### Partie 9 – Vitesse et direction
```python
d = np.sqrt((x2-x1)**2 + (y2-y1)**2)  # distance euclidienne en pixels/frame
```

### Partie 10 – Interprétation biomédicale
Applications, limites, comparaison avec méthodes avancées.

---

## 📊 Résultats obtenus

| Paramètre | Valeur |
|---|---|
| Nombre d'images | 15 |
| Taille des images | 256 × 256 px |
| Seuil optimal | 25 |
| Positions détectées | **14 / 14 (100%)** |
| Vitesse moyenne | **14.40 px/frame** |
| Vitesse max | 15.26 px/frame |
| Vitesse min | 13.04 px/frame |
| Écart-type | 0.54 px/frame |
| Type de trajectoire | Diagonale rectiligne |

---

## 💡 Réponses aux questions clés

| Question | Réponse |
|---|---|
| Zones claires dans la différence ? | Pixels ayant changé = zones de mouvement |
| Seuil trop faible ? | Trop de bruit → faux positifs |
| Seuil trop élevé ? | Mouvement raté → faux négatifs |
| Intérêt de la morphologie ? | Supprime bruit (OPEN) + comble trous (CLOSE) |
| Pourquoi le plus grand contour ? | Correspond à l'objet principal dans un cas simple |
| Rôle du centre ? | Position représentative pour calculer trajectoire/vitesse |
| Détection vs Tracking ? | Détection = présence ; Tracking = identité dans le temps |
| Mouvement toujours réel ? | Non : éclairage, bruit, caméra créent des faux positifs |

---

## 🏥 Applications biomédicales

| Application | Description |
|---|---|
| 🧬 Suivi de cellule T | Observer comment une cellule immunitaire traque sa cible |
| 🩹 Wound healing assay | Mesurer la vitesse de cicatrisation cellulaire |
| 💊 Nanoparticules | Suivre des agents médicamenteux dans un flux sanguin |
| 🩸 Hématologie | Analyser le mouvement de globules rouges/blancs |

---

## ⚠️ Limites de la méthode

- Fond doit être **statique**
- Sensible aux **variations de luminosité**
- Ne gère qu'**un seul objet** à la fois
- Pas de **mémoire** entre frames
- Perd l'objet s'il **s'arrête complètement**

> Solutions avancées : MOG2 (soustraction de fond adaptative), Optical Flow (Lucas-Kanade), Deep Learning (SORT, DeepSORT)

---

## 📚 Référence dataset réel (optionnel)

Pour travailler avec de vraies images biomédicales :
```bash
# Cellules HeLa – Cell Tracking Challenge (37 MB, gratuit)
wget https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
```
Source : [celltrackingchallenge.net](https://celltrackingchallenge.net/2d-datasets/)

---

## 👩‍💻 Auteur

**boualin902-lab**  
Étudiant(e) GBM2 – Institut National des Technologies et des Sciences du Kef  
Université de Jendouba, Tunisie

---

*TP réalisé dans le cadre du cours de Vision Artificielle – Semestre 2, 2025-2026*
