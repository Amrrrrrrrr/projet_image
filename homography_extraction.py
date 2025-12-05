import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def homography_apply(H, x, y):
    """Applique une homographie SANS coordonnées homogènes."""
    denom = H[2,0] * x + H[2,1] * y + H[2,2]
    x2 = (H[0,0] * x + H[0,1] * y + H[0,2]) / denom
    y2 = (H[1,0] * x + H[1,1] * y + H[1,2]) / denom
    return x2, y2
    
def homography_estimate(x1, y1, x2, y2):

    A = np.zeros((8, 8))
    B = np.zeros((8,))

    for i in range(4):
        X1, Y1 = x1[i], y1[i] 
        X2, Y2 = x2[i], y2[i]  


        A[2*i, :] = [
            X1, Y1, 1,     
            0,  0,  0,     
            -X2*X1, -X2*Y1 
        ]
        B[2*i] = X2

  
        A[2*i + 1, :] = [
            0,  0,  0,     
            X1, Y1, 1,    
            -Y2*X1, -Y2*Y1 
        ]
        B[2*i + 1] = Y2

    X = np.linalg.solve(A, B)

    H = np.array([
        [X[0], X[1], X[2]],
        [X[3], X[4], X[5]],
        [X[6], X[7], 1.0]
    ])

    return H

#test
x1 = [0, 1, 1, 0]
y1 = [0, 0, 1, 1]


x2 = [0, 2, 2, 0]
y2 = [0, 0, 1, 1]

H = homography_estimate(x1, y1, x2, y2)
print("H =\n", H)
def homography_extraction(I1, x_src, y_src, w, h):
    
    # Définir les 4 coins de l'image destination (Rectangle parfait)
    #  On va jusqu'à w-1 et h-1 car les indices commencent à 0
    x_dst = np.array([0, w-1, w-1, 0])
    y_dst = np.array([0, 0, h-1, h-1])
    
    # 2. Identification de l'homographie : Destination -> Source 
    # C'est pour savoir "où aller chercher la couleur" pour chaque pixel du résultat.
    H_inv = homography_estimate(x_dst, y_dst, x_src, y_src)
    
    # On génère tous les couples (u, v) pour l'image de taille w x h
    u_range = np.arange(w)
    v_range = np.arange(h)
    u_grid, v_grid = np.meshgrid(u_range, v_range) # Grilles 2D
    
    # Aplatissement pour utiliser votre fonction homography_apply vectorisée
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Application de l'homographie pour trouver les antécédents
    # Pour chaque pixel (u,v) du rectangle, on trouve (sx, sy) dans l'image source
    sx_flat, sy_flat = homography_apply(H_inv, u_flat, v_flat)
    
    # 5. Interpolation (Plus Proche Voisin)
    # On arrondit pour tomber sur le pixel le plus proche
    sx_rounded = np.round(sx_flat).astype(int)
    sy_rounded = np.round(sy_flat).astype(int)
    
    # 6. Gestion des bords (Masque de validité)
    height_src, width_src = I1.shape[:2]
    # On ne garde que les coordonnées qui tombent DANS l'image source
    mask = (sx_rounded >= 0) & (sx_rounded < width_src) & \
           (sy_rounded >= 0) & (sy_rounded < height_src)
    
    # 7. Remplissage de l'image finale
    # On prépare une image vide (noire)
    if I1.ndim == 3: # Si image couleur (RGB)
        I2_flat = np.zeros((w * h, 3), dtype=I1.dtype)
        # On remplit seulement les pixels valides en allant lire dans I1
        # Attention à l'ordre d'indexation numpy : I1[ligne, colonne] donc I1[y, x]
        valid_indices = (sy_rounded[mask], sx_rounded[mask])
        I2_flat[mask] = I1[valid_indices]
        I2 = I2_flat.reshape((h, w, 3))
    else: # Si image niveaux de gris
        I2_flat = np.zeros((w * h), dtype=I1.dtype)
        valid_indices = (sy_rounded[mask], sx_rounded[mask])
        I2_flat[mask] = I1[valid_indices]
        I2 = I2_flat.reshape((h, w))
        
    return I2

if __name__ == "__main__":
    # 1. Création d'une image de test (Damier) si pas d'image externe
    # Cela permet de tester le code sans fichier externe
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Fond gris
    img[:] = [50, 50, 50]
    # Un carré blanc au milieu "tourné" (simulé par un losange)
    # C'est juste pour avoir quelque chose à cliquer
    cv = [200, 200]
    for y in range(400):
        for x in range(400):
            if abs(x-200) + abs(y-200) < 100: # Forme de losange
                img[y, x] = [255, 255, 255]
                # Ajout de détails pour voir l'orientation
                if x > 200 and y < 200: img[y,x] = [255, 0, 0] # Coin rouge (HD)

    print("--- Validation Primitive Extraction ---")
    print("1. Une fenêtre va s'ouvrir.")
    print("2. Cliquez sur 4 points formant un quadrilatère.")
    print("   ORDRE IMPORTANT : Haut-Gauche -> Haut-Droit -> Bas-Droit -> Bas-Gauche")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Cliquez 4 points (Z-order: HG, HD, BD, BG)")
    
    # Récupération des clics [cite: 111]
    pts = plt.ginput(4, timeout=-1)
    plt.close()
    
    if len(pts) == 4:
        # Conversion en tableaux numpy
        x_s = np.array([p[0] for p in pts])
        y_s = np.array([p[1] for p in pts])
        
        # Dimensions choisies pour la sortie [cite: 112]
        W_out, H_out = 200, 300 
        
        # Appel de la fonction [cite: 113]
        res = homography_extraction(img, x_s, y_s, W_out, H_out)
        
        # Affichage du résultat
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img)
        ax1.plot(np.append(x_s, x_s[0]), np.append(y_s, y_s[0]), 'r-x', linewidth=2)
        ax1.set_title("Image Source (Sélection)")
        
        ax2.imshow(res)
        ax2.set_title(f"Extraction Rectangulaire ({W_out}x{H_out})")
        
        plt.show()
    else:
        print("Erreur : Vous n'avez pas cliqué 4 points.")