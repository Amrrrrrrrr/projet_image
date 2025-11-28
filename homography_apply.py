def homography_apply(H, x, y):
    """Applique une homographie SANS coordonnées homogènes."""
    denom = H[2,0] * x + H[2,1] * y + H[2,2]
    x2 = (H[0,0] * x + H[0,1] * y + H[0,2]) / denom
    y2 = (H[1,0] * x + H[1,1] * y + H[1,2]) / denom
    return x2, y2