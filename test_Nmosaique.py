import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def compute_output_size(x_s, y_s):
    pts = np.stack([x_s, y_s], axis=1)
    w_top    = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left   = np.linalg.norm(pts[3] - pts[0])
    h_right  = np.linalg.norm(pts[2] - pts[1])
    W_out = int(round((w_top + w_bottom) / 2.0))
    H_out = int(round((h_left + h_right) / 2.0))
    if W_out < 1: W_out = 1
    if H_out < 1: H_out = 1
    return W_out, H_out


def homography_apply(H, x, y):
    denom = H[2,0] * x + H[2,1] * y + H[2,2]
    denom = denom + 1e-12
    x2 = (H[0,0] * x + H[0,1] * y + H[0,2]) / denom
    y2 = (H[1,0] * x + H[1,1] * y + H[1,2]) / denom
    return x2, y2


def homography_estimate(x1, y1, x2, y2):
    A = np.zeros((8, 8))
    B = np.zeros((8,))

    for i in range(4):
        X1, Y1 = float(x1[i]), float(y1[i])
        X2, Y2 = float(x2[i]), float(y2[i])

        A[2*i, :] = [X1, Y1, 1, 0, 0, 0, -X2*X1, -X2*Y1]
        B[2*i]    = X2

        A[2*i+1,:]= [0, 0, 0, X1, Y1, 1, -Y2*X1, -Y2*Y1]
        B[2*i+1]  = Y2

    X = np.linalg.solve(A, B)

    H = np.array([
        [X[0], X[1], X[2]],
        [X[3], X[4], X[5]],
        [X[6], X[7], 1.0]
    ])
    return H


def homography_extraction(I0, x_src, y_src, w, h):
    x_dst = np.array([0, w-1, w-1, 0])
    y_dst = np.array([0, 0,   h-1, h-1])

    H_rect_to_0 = homography_estimate(x_dst, y_dst, x_src, y_src)

    Hinv = H_rect_to_0 
    Hs, Ws = I0.shape[0], I0.shape[1]

    if I0.ndim == 2:
        I_out = np.zeros((h, w))
    else:
        I_out = np.zeros((h, w, I0.shape[2]))

    for v in range(h):
        for u in range(w):
            sx, sy = homography_apply(Hinv, u, v)
            sx_i = int(round(float(sx)))
            sy_i = int(round(float(sy)))
            if 0 <= sx_i < Ws and 0 <= sy_i < Hs:
                I_out[v, u] = I0[sy_i, sx_i]

    return I_out, H_rect_to_0


def ItoMIB(I):
    I = np.asarray(I)
    h, w = I.shape[0], I.shape[1]
    M = np.ones((h, w))     
    B = (0, 0, w, h)
    return [I, M, B]

def warp_bbox(w, h, H):
    corners_x = np.array([0, w-1, w-1, 0])
    corners_y = np.array([0, 0,   h-1, h-1])

    xw, yw = homography_apply(H, corners_x, corners_y)

    xmin = int(np.floor(xw.min()))
    ymin = int(np.floor(yw.min()))
    xmax = int(np.ceil (xw.max())) + 1
    ymax = int(np.ceil (yw.max())) + 1
    return (xmin, ymin, xmax, ymax)


def MIB_transform(MIBin, H, outB):
    I, M, _ = MIBin
    I = np.asarray(I)
    M = np.asarray(M)
    H = np.asarray(H)

    hs, ws = I.shape[0], I.shape[1]
    xmin, ymin, xmax, ymax = map(int, outB)
    Wout, Hout = xmax - xmin, ymax - ymin

    xd = np.arange(xmin, xmax)
    yd = np.arange(ymin, ymax)
    Xd, Yd = np.meshgrid(xd, yd)

    Hinv = np.linalg.inv(H)
    xs, ys = homography_apply(Hinv, Xd, Yd)

    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)

    inside = (xi >= 0) & (xi < ws) & (yi >= 0) & (yi < hs)

    if I.ndim == 2:
        Iout = np.zeros((Hout, Wout))
    else:
        Iout = np.zeros((Hout, Wout, I.shape[2]))

    Mout = np.zeros((Hout, Wout))  

    Msrc = M[yi.clip(0, hs-1), xi.clip(0, ws-1)]
    src_valid = inside & (Msrc > 0)

    if I.ndim == 2:
        Iout[src_valid] = I[yi[src_valid], xi[src_valid]]
    else:
        Iout[src_valid, :] = I[yi[src_valid], xi[src_valid], :]

    Mout[src_valid] = 1.0
    return [Iout, Mout, (xmin, ymin, xmax, ymax)]


def mib_fusion(MIB1, MIB2, mode="mean"):
    I1, M1, B1 = MIB1
    I2, M2, B2 = MIB2

    x1min, y1min, x1max, y1max = B1
    x2min, y2min, x2max, y2max = B2

    xmin = min(x1min, x2min)
    ymin = min(y1min, y2min)
    xmax = max(x1max, x2max)
    ymax = max(y1max, y2max)

    Wout = int(xmax - xmin)
    Hout = int(ymax - ymin)

    I1 = np.asarray(I1); M1 = np.asarray(M1)
    I2 = np.asarray(I2); M2 = np.asarray(M2)

    is_gray = (I1.ndim == 2)

    off1x, off1y = int(x1min - xmin), int(y1min - ymin)
    h1, w1 = I1.shape[0], I1.shape[1]
    rr1, cc1 = slice(off1y, off1y + h1), slice(off1x, off1x + w1)

    off2x, off2y = int(x2min - xmin), int(y2min - ymin)
    h2, w2 = I2.shape[0], I2.shape[1]
    rr2, cc2 = slice(off2y, off2y + h2), slice(off2x, off2x + w2)

    m1 = (M1 > 0)
    m2 = (M2 > 0)

    if mode == "mean":
        if is_gray:
            num = np.zeros((Hout, Wout))  
        else:
            num = np.zeros((Hout, Wout, I1.shape[2]))
        den = np.zeros((Hout, Wout))     
        if is_gray:
            num[rr1, cc1] += I1 * (m1 * 1.0)
        else:
            num[rr1, cc1, :] += I1 * (m1[:, :, None] * 1.0)
        den[rr1, cc1] += (m1 * 1.0)

        if is_gray:
            num[rr2, cc2] += I2 * (m2 * 1.0)
        else:
            num[rr2, cc2, :] += I2 * (m2[:, :, None] * 1.0)
        den[rr2, cc2] += (m2 * 1.0)

        valid = den != 0

        if is_gray:
            Iout = np.zeros((Hout, Wout))
            Iout[valid] = num[valid] / den[valid]
        else:
            Iout = np.zeros((Hout, Wout, num.shape[2]))
            Iout[valid, :] = num[valid, :] / den[valid][:, None]

        Mout = valid * 1.0
        return [Iout, Mout, (xmin, ymin, xmax, ymax)]


    elif mode == "I1":
        if is_gray:
            Iout = np.zeros((Hout, Wout))
        else:
            Iout = np.zeros((Hout, Wout, I1.shape[2]))
        Mout = np.zeros((Hout, Wout))

        if is_gray:
            Iout[rr2, cc2][m2] = I2[m2]
        else:
            Iout[rr2, cc2][m2, :] = I2[m2, :]
        Mout[rr2, cc2][m2] = 1.0

        if is_gray:
            Iout[rr1, cc1][m1] = I1[m1]
        else:
            Iout[rr1, cc1][m1, :] = I1[m1, :]
        Mout[rr1, cc1][m1] = 1.0

        return [Iout, Mout, (xmin, ymin, xmax, ymax)]

    else:
        raise ValueError("mode doit être 'mean' ou 'I1'")



def bruiter_image(I, sigma):
    
    I = np.asarray(I)
    bruit = sigma * np.random.randn(*I.shape)
    J = I + bruit
    J = np.clip(J, 0, 1)
    return J


def extraire_imagettes_depuis_image_origine(img, N):
    
    images = []
    H_list = []
    for k in range(N):
        xk, yk = choisir_quad(img, f"EXTRACTION {k+1}/{N} (zones recouvrantes)")
        Wk, Hk = compute_output_size(xk, yk)
        Ik, Hk_rect_to_0 = homography_extraction(img, xk, yk, Wk, Hk)
        images.append(Ik)
        H_list.append(Hk_rect_to_0)
    return images, H_list


def homographies_vers_reference(Hrect_to_0_list, ref=0):
    Href = Hrect_to_0_list[ref]
    Href_inv = np.linalg.inv(Href)
    H_to_ref = []
    for Hi in Hrect_to_0_list:
        H_to_ref.append(Href_inv @ Hi)
    return H_to_ref

def choisir_quad(img, title):
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(title + "\nClique 4 points: HG, HD, BD, BG")
    pts = plt.ginput(4, timeout=-1)
    plt.close()
    if len(pts) != 4:
        raise RuntimeError("Il faut 4 points.")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return x, y


def bbox_globale(images, H_to_ref):
    """
    Calcule BG = union de toutes les bbox des images warpées dans le repère ref.
    """
    BG = None
    for I, H in zip(images, H_to_ref):
        h, w = I.shape[0], I.shape[1]
        B = warp_bbox(w, h, H) 
        if BG is None:
            BG = B
        else:
            BG = (min(BG[0], B[0]), min(BG[1], B[1]),
                  max(BG[2], B[2]), max(BG[3], B[3]))
    return BG


def MIB_fusion_N_parallele(MIB_list):
    
    I0, M0, B0 = MIB_list[0]
    Hout, Wout = I0.shape[0], I0.shape[1]

    num = np.zeros(I0.shape)
    den = np.zeros((Hout, Wout))

    for I, M, B in MIB_list:
        m = (np.asarray(M) > 0) * 1.0
        if np.asarray(I).ndim == 2:
            num += I * m
        else:
            num += I * m[:, :, None]
        den += m

    valid = den != 0

    Iout = np.zeros(I0.shape)
    if I0.ndim == 2:
        Iout[valid] = num[valid] / den[valid]
    else:
        Iout[valid, :] = num[valid, :] / den[valid][:, None]

    Mout = valid * 1.0
    return [Iout, Mout, B0]


def MIB_fusion_N_sequentielle(MIB_list, mode="mean"):
    
    acc = MIB_list[0]
    for k in range(1, len(MIB_list)):
        acc = mib_fusion(acc, MIB_list[k], mode=mode)
    return acc

if __name__ == "__main__":
    img = mpimg.imread("singe.jpg.webp")
    img = mpimg.imread("singe.jpg.webp").astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0


    N = 4         
    sigma = 0.5  
    ref = 0          

    images, Hrect_to_0_list = extraire_imagettes_depuis_image_origine(img, N)

    if sigma > 0:
        images_bruitees = []
        for I in images:
            images_bruitees.append(bruiter_image(I, sigma))
        images = images_bruitees

    H_to_ref = homographies_vers_reference(Hrect_to_0_list, ref=ref)

    BG = bbox_globale(images, H_to_ref)
    print("BG =", BG)

    MIB_big_list = []
    for I, H in zip(images, H_to_ref):
        MIB = ItoMIB(I)
        MIB_big = MIB_transform(MIB, H, outB=BG)
        MIB_big_list.append(MIB_big)

    MIB_par = MIB_fusion_N_parallele(MIB_big_list)
    I_par = MIB_par[0]


    MIB_seq_1 = MIB_fusion_N_sequentielle(MIB_big_list, mode="mean")
    MIB_seq_2 = MIB_fusion_N_sequentielle(list(reversed(MIB_big_list)), mode="mean")
    I_seq_1 = MIB_seq_1[0]
    I_seq_2 = MIB_seq_2[0]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(I_par)
    plt.title("Fusion PARALLELE (ordre indépendant)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(I_seq_1)
    plt.title("Fusion SEQ (ordre normal)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(I_seq_2)
    plt.title("Fusion SEQ (ordre inversé)")
    plt.axis("off")

    plt.show()

    plt.imsave("mosaic_parallele.png", I_par)
    plt.imsave("mosaic_seq_normal.png", I_seq_1)
    plt.imsave("mosaic_seq_inverse.png", I_seq_2)
    print("✅ Sauvegardé : mosaic_parallele.png / mosaic_seq_normal.png / mosaic_seq_inverse.png")
