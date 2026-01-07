import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi


def otsu_threshold(values, nbins=256):
    """Calcule un seuil Otsu sur un tableau 1D de floats."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    p = hist / (hist.sum() + 1e-12)

    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float(centers[k])


def rank_colored_squares(image_path, min_area=800, show=True, debug=False):
    """
    Classe les carrés colorés par surface décroissante.
    Surface = nb de pixels du composant.
    Retourne: list of dicts {id, area, rel, bbox}
    """

    I = mpimg.imread(image_path)
    if I.ndim == 2:  # grayscale -> on force 3 canaux
        I = np.stack([I, I, I], axis=-1)
    if I.shape[2] == 4:  # RGBA -> RGB
        I = I[:, :, :3]

    # normaliser en float [0,1]
    if I.dtype != np.float32 and I.dtype != np.float64:
        I = I.astype(np.float32)
        if I.max() > 1.5:
            I /= 255.0

    H, W, _ = I.shape

    # 1) Couleur du fond (mur) : médiane des BORDS (beaucoup plus stable)
    border = np.vstack([I[0, :, :], I[-1, :, :], I[:, 0, :], I[:, -1, :]])
    bg = np.median(border, axis=0)

    # 2) distance couleur au fond
    dist = np.sqrt(((I - bg) ** 2).sum(axis=2))

    # 3) seuil automatique Otsu (+ petit facteur pour être permissif)
    t = otsu_threshold(dist) * 0.90   # si encore 1 carré manque: mets 0.85
    mask = dist > t

    # 4) nettoyage morphologique (doux)
    # (on enlève l'opening qui peut "manger" des carrés proches du mur)
    mask = ndi.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndi.binary_fill_holes(mask)

    if debug:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(dist, cmap="gray")
        plt.title("dist au mur")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("mask carrés")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # 5) composantes connexes
    lbl, n = ndi.label(mask)
    if n == 0:
        raise RuntimeError("Aucun objet détecté. Vérifie l'image ou baisse le seuil (facteur 0.90 -> 0.80).")

    # aire de chaque composante
    areas = ndi.sum(mask, lbl, index=np.arange(1, n + 1))

    comps = []
    for k, area in enumerate(areas, start=1):
        if area < min_area:
            continue
        ys, xs = np.where(lbl == k)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        comps.append({"id": k, "area": float(area), "bbox": (x0, y0, x1, y1)})

    if not comps:
        raise RuntimeError("Objets détectés trop petits. Diminue min_area (ex: 200).")

    # 6) tri décroissant + surfaces relatives
    comps.sort(key=lambda d: d["area"], reverse=True)
    amax = comps[0]["area"]
    for c in comps:
        c["rel"] = c["area"] / amax

    # affichage console
    print("\nClassement (surface relative à la plus grande) :")
    for i, c in enumerate(comps, start=1):
        print(f"{i:02d}. composante#{c['id']}  area={int(c['area'])} px   rel={c['rel']:.3f}")

    # 7) affichage image annotée
    if show:
        plt.figure(figsize=(10, 6))
        plt.imshow(I)
        plt.title("Carrés détectés + classement")
        ax = plt.gca()

        for rank, c in enumerate(comps, start=1):
            x0, y0, x1, y1 = c["bbox"]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2))
            ax.text(x0, max(0, y0 - 5), f"{rank} ({c['rel']:.2f})", fontsize=12)

        plt.axis("off")
        plt.show()

    return comps


if __name__ == "__main__":
    # Mets ici TON image extraite (extraction.jpg / extraction.png)
    rank_colored_squares("extraction.jpg", min_area=800, show=True, debug=False)
