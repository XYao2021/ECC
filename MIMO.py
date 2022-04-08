import random
import numpy as np

M = 3
N = 4

a = complex(2, 5)
b = random.gauss(0, 1/2)

H = [[0 for _ in range(N)] for _ in range(M)]
# print(H)
for i in range(M):
    for j in range(N):
        H[i][j] = complex(random.gauss(0, 1/2), random.gauss(0, 1/2))
print(np.array(H).shape)
