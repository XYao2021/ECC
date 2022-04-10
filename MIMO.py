import numpy as np
from Functions import *

M = 3
N = 2

H = H_generator(M, N)
print(H)

a = np.array([-1, -1, -1, 1, 1, 1])
print(np.split(a, M))

