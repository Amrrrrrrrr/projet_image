import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from homography_apply import homography_apply
from homography_estimate import homography_estimate

def homography_projection(I1, I2, x, y):

    H1, W1 = I1.shape[:2]

    xs = np.array([0,W1-1, W1-1, 0], dtype=float)
    ys = np.array([0, 0,  H1-1, H1-1], dtype=float)

  
    xd = np.array(x, dtype=float)
    yd = np.array(y, dtype=float)


    H_sd = homography_estimate(xs, ys, xd, yd)


    H_ds = homography_estimate(xd, yd, xs, ys)


    I2_proj = I2.copy()


    xmin = int(np.floor(xd.min()))
    xmax = int(np.ceil(xd.max()))
    ymin = int(np.floor(yd.min()))
    ymax = int(np.ceil(yd.max()))

    H2, W2 = I2.shape[:2]

    xmin = max(0, xmin)
    xmax = min(W2-1, xmax)
    ymin = max(0, ymin)
    ymax = min(H2-1, ymax)


    for j in range(ymin, ymax+1):
        for i in range(xmin, xmax+1):

            xs_src, ys_src = homography_apply(H_ds, i, j)


            xs_r = int(round(xs_src))
            ys_r = int(round(ys_src))


            if 0 <= xs_r < W1 and 0 <= ys_r < H1:
                I2_proj[j, i] = I1[ys_r, xs_r]

    return I2_proj

