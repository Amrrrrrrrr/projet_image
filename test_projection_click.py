import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from homography_apply import homography_apply
from homography_estimate import homography_estimate
from homography_projection import homography_projection

I1 = imread("I1.jpeg")   
I2 = imread("I3.jpg")   

if I1.ndim == 3:
    I1 = I1.mean(axis=2)

if I2.ndim == 3:
    I2 = I2.mean(axis=2)

print("=== Test Projection ===")
print("Clique 4 points dans l'ordre : Haut-Gauche -> Haut-Droit -> Bas-Droit -> Bas-Gauche")

plt.figure(figsize=(8, 8))
plt.imshow(I2, cmap='gray')
plt.title("Cliquez 4 points (ordre : HG, HD, BD, BG)")
pts = plt.ginput(4, timeout=-1)
plt.close()

if len(pts) != 4:
    print("Erreur : vous devez cliquer EXACTEMENT 4 points.")
    exit()

# Extraire x[] et y[]
x = np.array([p[0] for p in pts], dtype=float)
y = np.array([p[1] for p in pts], dtype=float)

print("Points cliques :")
print("x =", x)
print("y =", y)


I2_proj = homography_projection(I1, I2, x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


ax1.imshow(I2, cmap='gray')
ax1.plot(np.append(x, x[0]), np.append(y, y[0]), 'r-x', linewidth=2)
ax1.set_title("Image Destination (Quadrilatere selectionne)")

ax2.imshow(I2_proj, cmap='gray')
ax2.set_title("Apres Projection")

plt.show()


plt.imsave("projection_result.png", I2_proj, cmap='gray')
print("Image projetee sauvegardee sous : projection_result.png")
