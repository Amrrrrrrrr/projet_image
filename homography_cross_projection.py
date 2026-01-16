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


def homography_cross_projection(I, x1, y1, x2, y2):
    x1 = np.array(x1, dtype=float); y1 = np.array(y1, dtype=float)
    x2 = np.array(x2, dtype=float); y2 = np.array(y2, dtype=float)

    # --- NORMALISATION DES 4 COINS : TL, TR, BR, BL + sens cohérent
    P1 = np.stack([x1, y1], axis=1)  # (4,2)
    s1 = P1[:,0] + P1[:,1]
    d1 = P1[:,0] - P1[:,1]
    TL1 = P1[np.argmin(s1)]
    BR1 = P1[np.argmax(s1)]
    TR1 = P1[np.argmax(d1)]
    BL1 = P1[np.argmin(d1)]
    P1 = np.array([TL1, TR1, BR1, BL1], dtype=float)

    P2 = np.stack([x2, y2], axis=1)
    s2 = P2[:,0] + P2[:,1]
    d2 = P2[:,0] - P2[:,1]
    TL2 = P2[np.argmin(s2)]
    BR2 = P2[np.argmax(s2)]
    TR2 = P2[np.argmax(d2)]
    BL2 = P2[np.argmin(d2)]
    P2 = np.array([TL2, TR2, BR2, BL2], dtype=float)

    # Forcer le même sens (si un quad est inversé -> flip)
    def signed_area(P):
        return 0.5 * np.sum(P[:,0]*np.roll(P[:,1], -1) - P[:,1]*np.roll(P[:,0], -1))

    if signed_area(P1) * signed_area(P2) < 0:
        # inverse l'ordre (garde TL en premier)
        P2 = P2[[0, 3, 2, 1]]

    x1, y1 = P1[:,0], P1[:,1]
    x2, y2 = P2[:,0], P2[:,1]
    # --- FIN NORMALISATION

    # Carré canonique dans l'espace temporaire
    xC = np.array([0, 1, 1, 0], dtype=float)
    yC = np.array([0, 0, 1, 1], dtype=float)

    H_q1_to_C = homography_estimate(x1, y1, xC, yC)
    H_q2_to_C = homography_estimate(x2, y2, xC, yC)
    H_C_to_q1 = homography_estimate(xC, yC, x1, y1)
    H_C_to_q2 = homography_estimate(xC, yC, x2, y2)

    H_src, W_src = I.shape[0], I.shape[1]
    I_out = I.copy()

    for y in range(H_src):
        for x in range(W_src):
            u1, v1 = homography_apply(H_q1_to_C, x, y)
            if 0 <= u1 <= 1 and 0 <= v1 <= 1:
                x2f, y2f = homography_apply(H_C_to_q2, u1, v1)
                xs = int(round(x2f)); ys = int(round(y2f))
                if 0 <= xs < W_src and 0 <= ys < H_src:
                    I_out[y, x] = I[ys, xs]
                continue

            u2, v2 = homography_apply(H_q2_to_C, x, y)
            if 0 <= u2 <= 1 and 0 <= v2 <= 1:
                x1f, y1f = homography_apply(H_C_to_q1, u2, v2)
                xs = int(round(x1f)); ys = int(round(y1f))
                if 0 <= xs < W_src and 0 <= ys < H_src:
                    I_out[y, x] = I[ys, xs]
                continue

    return I_out


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == "__main__":
    # 1) Charger l'image
    image_path = "singe.jpg.webp"   # <-- change le nom de fichier ici
    I = mpimg.imread(image_path)

    # Si l'image est en float [0,1], on la passe en uint8 [0,255] (optionnel)
    if I.dtype == np.float32 or I.dtype == np.float64:
        I = (I * 255).astype(np.uint8)

    # 2) Sélection du premier quadrilatère
    print("Sélection du QUADRILATÈRE 1")
    print("Clique 4 points dans l'ordre : Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche")

    plt.figure(figsize=(8, 8))
    plt.imshow(I)
    plt.title("Quad 1 : clique 4 points (HG, HD, BD, BG)")
    pts1 = plt.ginput(4, timeout=-1)
    plt.close()

    if len(pts1) != 4:
        print("Erreur : tu n'as pas cliqué 4 points pour le quad 1.")
        exit(1)

    x1 = [p[0] for p in pts1]
    y1 = [p[1] for p in pts1]

    # 3) Sélection du deuxième quadrilatère
    print("Sélection du QUADRILATÈRE 2")
    print("Clique 4 points dans l'ordre : Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche")

    plt.figure(figsize=(8, 8))
    plt.imshow(I)
    plt.title("Quad 2 : clique 4 points (HG, HD, BD, BG)")
    pts2 = plt.ginput(4, timeout=-1)
    plt.close()

    if len(pts2) != 4:
        print("Erreur : tu n'as pas cliqué 4 points pour le quad 2.")
        exit(1)

    x2 = [p[0] for p in pts2]
    y2 = [p[1] for p in pts2]

    # 4) Appel de ta fonction de projection croisée
    I_out = homography_cross_projection(I, x1, y1, x2, y2)

    # 5) Affichage du résultat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(I)
    ax1.set_title("Image originale")
    ax1.plot(x1 + [x1[0]], y1 + [y1[0]], 'r-x')
    ax1.plot(x2 + [x2[0]], y2 + [y2[0]], 'b-x')
    ax1.legend(["Quad 1", "Quad 2"])

    ax2.imshow(I_out)
    ax2.set_title("Après homography_cross_projection")

    plt.show()
