import numpy as np

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