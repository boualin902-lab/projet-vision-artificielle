"""
TP 6 : Détection et suivi du mouvement dans une séquence d'images
Filière : GBM2 | Vision Artificielle
Google Colab - Code complet
"""

# ============================================================
# ETAPE 0 : Installation et téléchargement des données
# ============================================================

# Télécharger une séquence biomédicale (cellules HeLa - Cell Tracking Challenge)
# Exécuter dans Colab :

"""
!pip install opencv-python-headless -q

# Téléchargement direct de la séquence de cellules HeLa
!wget -q "https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip" -O HeLa.zip
!unzip -q HeLa.zip -d hela_data

# Vérifier les images disponibles
import os
folder = "hela_data/DIC-C2DH-HeLa/01"
print(os.listdir(folder)[:5])
"""

# ============================================================
# ALTERNATIVE : Générer une séquence synthétique biomédicale
# (si le téléchargement ne marche pas ou trop lent)
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generer_sequence_synthetique(n_frames=15, taille=256, dossier="sequence"):
    """
    Génère une séquence d'images simulant une cellule en mouvement
    sur fond bruité (image pseudo-biomédicale).
    """
    os.makedirs(dossier, exist_ok=True)
    
    cx, cy = 40, 40       # Position initiale de la cellule
    rayon  = 18           # Rayon de la cellule
    
    for i in range(n_frames):
        # Fond bruité (simule un fond microscopique)
        img = np.random.randint(20, 60, (taille, taille), dtype=np.uint8)
        
        # Texture fond (légère variation)
        bruit = np.random.normal(0, 5, (taille, taille)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + bruit, 0, 255).astype(np.uint8)
        
        # Dessiner la cellule (cercle blanc flou = réaliste)
        cx_i = int(cx + i * 12)   # Déplacement horizontal
        cy_i = int(cy + i * 8)    # Déplacement diagonal
        cv2.circle(img, (cx_i, cy_i), rayon, 200, -1)
        
        # Noyau de la cellule
        cv2.circle(img, (cx_i, cy_i), rayon // 3, 240, -1)
        
        # Flou gaussien pour effet microscopique
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        
        filename = os.path.join(dossier, f"frame_{i+1:02d}.png")
        cv2.imwrite(filename, img)
    
    print(f"Séquence générée : {n_frames} images dans '{dossier}/'")

# Générer la séquence
generer_sequence_synthetique(n_frames=15)
folder = "sequence"


# ============================================================
# PARTIE 1 : Lecture et affichage de la séquence
# ============================================================

images = []
files = sorted(os.listdir(folder))

for file in files:
    if file.endswith(".png"):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

print("Nombre d'images :", len(images))
print("Taille image :", images[0].shape)

plt.figure(figsize=(12, 4))
for i in range(min(3, len(images))):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(f"Frame {i+1}")
    plt.axis("off")
plt.suptitle("Partie 1 : Lecture de la séquence", fontweight="bold")
plt.tight_layout()
plt.savefig("partie1_lecture.png", dpi=150)
plt.show()


# ============================================================
# PARTIE 2 : Différence entre deux images
# ============================================================

img1 = images[0]
img2 = images[1]
diff = cv2.absdiff(img2, img1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img1, cmap="gray")
plt.title("Image 1 (Frame 1)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap="gray")
plt.title("Image 2 (Frame 2)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(diff, cmap="gray")
plt.title("Différence absolue")
plt.axis("off")

plt.suptitle("Partie 2 : Différence d'images", fontweight="bold")
plt.tight_layout()
plt.savefig("partie2_difference.png", dpi=150)
plt.show()

# Réponse à la question :
print("""
INTERPRETATION PARTIE 2 :
Les zones claires dans l'image différence correspondent aux pixels
qui ont changé entre les deux frames. Elles indiquent les zones
où la cellule (ou l'objet) s'est déplacée.
""")


# ============================================================
# PARTIE 3 : Seuillage
# ============================================================

seuils = [10, 20, 30, 40]

plt.figure(figsize=(15, 4))
for idx, seuil in enumerate(seuils):
    _, masque = cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)
    pixels_blancs = np.sum(masque == 255)
    
    plt.subplot(1, 4, idx+1)
    plt.imshow(masque, cmap="gray")
    plt.title(f"Seuil = {seuil}\n({pixels_blancs} px blancs)")
    plt.axis("off")

plt.suptitle("Partie 3 : Test de différents seuils", fontweight="bold")
plt.tight_layout()
plt.savefig("partie3_seuillage.png", dpi=150)
plt.show()

# Seuil retenu
threshold_value = 25
_, motion_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

print("""
REPONSES PARTIE 3 :
- Seuil trop faible (ex: 10) : capte trop de bruit (faux positifs),
  zones non significatives détectées comme mouvement.
- Seuil trop élevé (ex: 40) : ignore des vraies zones de mouvement
  faible, le mouvement peut être raté (faux négatifs).
- Meilleur seuil ici : 20-25 (bon équilibre signal/bruit).
""")


# ============================================================
# PARTIE 4 : Nettoyage par morphologie
# ============================================================

kernel = np.ones((3, 3), np.uint8)
clean = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)   # Supprime bruit
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)         # Bouche trous

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(motion_mask, cmap="gray")
plt.title("Avant nettoyage")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clean, cmap="gray")
plt.title("Après nettoyage morphologique")
plt.axis("off")

plt.suptitle("Partie 4 : Nettoyage morphologique", fontweight="bold")
plt.tight_layout()
plt.savefig("partie4_morphologie.png", dpi=150)
plt.show()

print("""
REPONSE PARTIE 4 :
- MORPH_OPEN : érosion puis dilatation → supprime les petits bruits isolés
- MORPH_CLOSE : dilatation puis érosion → bouche les petits trous dans l'objet
- Résultat : masque plus propre, contours plus précis, moins de faux positifs.
""")


# ============================================================
# PARTIE 5 : Détection de l'objet en mouvement
# ============================================================

img_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"Objet détecté à : x={x}, y={y}, largeur={w}, hauteur={h}")
    print(f"Surface contour : {cv2.contourArea(cnt):.1f} px²")

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Partie 5 : Objet détecté")
plt.axis("off")
plt.savefig("partie5_detection.png", dpi=150)
plt.show()

print("""
REPONSE PARTIE 5 :
On garde le plus grand contour car dans un cas simple (1 cellule),
il correspond à l'objet principal. Les petits contours sont du bruit
résiduel après nettoyage.
""")


# ============================================================
# PARTIE 6 : Calcul du centre de l'objet
# ============================================================

if len(contours) > 0:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    print(f"Centre de l'objet : ({cx}, {cy})")

print("""
REPONSE PARTIE 6 :
Le centre (centroïde) est le point représentatif de l'objet.
Dans le tracking, il sert à représenter la position de l'objet
à chaque frame et à calculer la trajectoire et la vitesse.
""")


# ============================================================
# PARTIE 7 : Tracking sur toute la séquence
# ============================================================

positions = []

for i in range(len(images) - 1):
    img1_t = images[i]
    img2_t = images[i + 1]
    
    diff_t = cv2.absdiff(img2_t, img1_t)
    _, motion_mask_t = cv2.threshold(diff_t, 25, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    clean_t = cv2.morphologyEx(motion_mask_t, cv2.MORPH_OPEN, kernel)
    clean_t = cv2.morphologyEx(clean_t, cv2.MORPH_CLOSE, kernel)
    
    contours_t, _ = cv2.findContours(clean_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_t) > 0:
        cnt_t = max(contours_t, key=cv2.contourArea)
        M_t = cv2.moments(cnt_t)
        if M_t["m00"] != 0:
            cx_t = int(M_t["m10"] / M_t["m00"])
            cy_t = int(M_t["m01"] / M_t["m00"])
            positions.append((cx_t, cy_t))

print(f"Positions détectées ({len(positions)} points) :")
for i, pos in enumerate(positions):
    print(f"  Frame {i+1}→{i+2} : {pos}")


# ============================================================
# PARTIE 8 : Tracé de la trajectoire
# ============================================================

last_img = cv2.cvtColor(images[-1], cv2.COLOR_GRAY2BGR)

for i in range(1, len(positions)):
    cv2.line(last_img, positions[i-1], positions[i], (255, 100, 0), 2)
    cv2.circle(last_img, positions[i], 4, (0, 255, 0), -1)

# Marquer le point de départ
if len(positions) > 0:
    cv2.circle(last_img, positions[0], 6, (0, 0, 255), -1)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB))
plt.title("Partie 8 : Trajectoire de l'objet\n(Rouge=départ, Vert=points, Bleu=chemin)")
plt.axis("off")
plt.savefig("partie8_trajectoire.png", dpi=150)
plt.show()

# Tracé de la trajectoire en 2D
if len(positions) >= 2:
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'b-o', markersize=6, linewidth=2)
    plt.plot(xs[0], ys[0], 'rs', markersize=10, label="Départ")
    plt.plot(xs[-1], ys[-1], 'g^', markersize=10, label="Arrivée")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Trajectoire 2D de la cellule")
    plt.legend()
    plt.gca().invert_yaxis()  # Inverser Y (origine image = haut gauche)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("partie8_trajectoire2D.png", dpi=150)
    plt.show()


# ============================================================
# PARTIE 9 : Calcul de la vitesse
# ============================================================

distances = []
for i in range(1, len(positions)):
    x1, y1 = positions[i-1]
    x2, y2 = positions[i]
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distances.append(d)

print("Déplacements successifs (pixels/frame) :")
for i, d in enumerate(distances):
    print(f"  Frame {i+1}→{i+2} : {d:.2f} px")

if len(distances) > 0:
    print(f"\nVitesse moyenne : {np.mean(distances):.2f} pixels/frame")
    print(f"Vitesse max     : {np.max(distances):.2f} pixels/frame")
    print(f"Vitesse min     : {np.min(distances):.2f} pixels/frame")
    print(f"Écart-type      : {np.std(distances):.2f} (mesure régularité)")

# Graphe de vitesse
if len(distances) > 0:
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(distances)+1), distances, color='steelblue', alpha=0.8)
    plt.axhline(np.mean(distances), color='red', linestyle='--', label=f'Moyenne = {np.mean(distances):.2f}')
    plt.xlabel("Intervalle de frames")
    plt.ylabel("Distance (pixels)")
    plt.title("Partie 9 : Vitesse à chaque frame")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("partie9_vitesse.png", dpi=150)
    plt.show()

print("""
REPONSES PARTIE 9 :
1. Le mouvement est-il régulier ? 
   → Oui dans notre simulation (déplacement constant par frame).
   En réalité biologique, le mouvement cellulaire est souvent irrégulier.

2. La vitesse est-elle constante ?
   → Dans la simulation : oui (déplacement identique à chaque step).
   On le voit dans les distances (toutes similaires).

3. Comment le voir ?
   → Si les distances varient beaucoup (grand écart-type), 
   le mouvement est irrégulier. Si elles sont proches, il est uniforme.
""")


# ============================================================
# PARTIE 10 : Interprétation biomédicale
# ============================================================

print("""
=============================================================
PARTIE 10 : INTERPRETATION BIOMEDICALE
=============================================================

1. POURQUOI LE SUIVI DU MOUVEMENT EST-IL UTILE EN BIOMÉDICAL ?
   - Analyser le comportement cellulaire (migration, division)
   - Mesurer la motilité cellulaire (ex: cellules cancéreuses)
   - Suivre des particules dans le sang (globules, agents thérapeutiques)
   - Évaluer l'efficacité de médicaments sur la mobilité cellulaire

2. EXEMPLES D'APPLICATION :
   a) Suivi de cellule :
      Observer comment une cellule T (immunité) traque une cible.
   
   b) Migration cellulaire :
      Mesurer la vitesse et direction de migration cellulaire
      lors d'une cicatrisation (wound healing assay).
   
   c) Suivi de particules :
      Nano-particules médicamenteuses dans un flux sanguin simulé.

3. LIMITES DE LA DIFFÉRENCE D'IMAGES :
   - Sensible aux variations de luminosité (faux mouvement)
   - Ne fonctionne pas si la caméra bouge (fond non stationnaire)
   - Perd l'objet si deux cellules se croisent ou se touchent
   - Ne distingue pas plusieurs objets en mouvement
   - Bruit d'acquisition peut être détecté comme mouvement
=============================================================
""")


# ============================================================
# QUESTIONS FINALES DU RAPPORT
# ============================================================

print("""
=============================================================
REPONSES AUX QUESTIONS FINALES
=============================================================

1. RÔLE DU SEUIL :
   Transformer l'image différence (niveaux de gris) en image
   binaire (noir/blanc). Il sépare les vraies zones de mouvement
   du bruit de fond. Un bon seuil isole uniquement le mouvement
   significatif.

2. DIFFÉRENCE ENTRE DÉTECTION ET TRACKING :
   - Détection : identifier la PRÉSENCE d'un objet dans une image.
   - Tracking : suivre la POSITION d'un objet dans le TEMPS,
     en associant les détections successives au même objet.

3. LE MOUVEMENT DÉTECTÉ EST-IL TOUJOURS UN VRAI MOUVEMENT ?
   Non. Des faux positifs sont possibles :
   - Variation de luminosité (ombre, reflet)
   - Bruit du capteur
   - Mouvement de la caméra
   Ces cas nécessitent des méthodes plus robustes.

4. LIMITES DE LA MÉTHODE :
   - Fond doit être statique
   - Sensible au bruit et aux changements de lumière
   - Un seul objet par séquence (simple différence)
   - Pas de mémoire de l'objet entre frames
   - Perd l'objet si l'objet s'arrête complètement
=============================================================
""")

print("TP terminé ! Captures sauvegardées dans le répertoire courant.")
