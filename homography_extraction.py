import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy as np

def compute_output_size(x_s, y_s):
    pts = np.stack([x_s, y_s], axis=1)  

    w_top    = np.linalg.norm(pts[1] - pts[0])  
    w_bottom = np.linalg.norm(pts[2] - pts[3])  
    h_left   = np.linalg.norm(pts[3] - pts[0])  
    h_right  = np.linalg.norm(pts[2] - pts[1])  

    W_out = int(round((w_top + w_bottom) / 2.0))
    H_out = int(round((h_left + h_right) / 2.0))

    # sécurité
    W_out = max(W_out, 1)
    H_out = max(H_out, 1)
    return W_out, H_out

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

    x_dst = np.array([0, w-1, w-1, 0])
    y_dst = np.array([0,0,h-1, h-1])

    H_inv = homography_estimate(x_dst, y_dst, x_src, y_src)

    height_src, width_src = I1.shape[0], I1.shape[1]

    shape_out = list(I1.shape)
    shape_out[0] = h  
    shape_out[1] = w   
    I2 = np.zeros(shape_out, dtype=I1.dtype)

    for v in range(h):    
        for u in range(w): 
            sx, sy = homography_apply(H_inv, u, v)
            # Plus proche voisin
            sx_i = int(round(sx))
            sy_i = int(round(sy))
            # Vérification que le point est dans l'image source
            if 0 <= sx_i < width_src and 0 <= sy_i < height_src:
                I2[v, u] = I1[sy_i, sx_i]

    return I2

# --- Charger votre image ---
image_path = "graffiti1.jpg" 
img = mpimg.imread(image_path)


# --- Instructions utilisateur ---
print("--- Extraction via homographie ---")
print("1. Une fenêtre va s'ouvrir.")
print("2. Cliquez sur 4 points formant un quadrilatère.")
print("   ORDRE : Haut-Gauche -> Haut-Droit -> Bas-Droit -> Bas-Gauche")

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("Cliquez 4 points (HG, HD, BD, BG)")
pts = plt.ginput(4, timeout=-1)
plt.close()

if len(pts) == 4:
    x_s = np.array([p[0] for p in pts])
    y_s = np.array([p[1] for p in pts])

    # Taille de sortie souhaitée
    W_out, H_out = compute_output_size(x_s, y_s)
    res = homography_extraction(img, x_s, y_s, W_out, H_out)

    res_u8 = (np.clip(res, 0, 1) * 255).astype(np.uint8) if res.dtype != np.uint8 else res
    plt.imsave("extraction.jpg", res_u8)
    print("Sauvegardée : extraction.jpg")

    # Affichage du résultat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.plot(np.append(x_s, x_s[0]), np.append(y_s, y_s[0]), 'r-x', linewidth=2)
    ax1.set_title("Image Source (Sélection)")
    ax2.imshow(res)
    ax2.set_title(f"Extraction Rectangulaire ({W_out}x{H_out})")
    plt.show()
else:
    print("Erreur : vous n'avez pas cliqué 4 points.")
