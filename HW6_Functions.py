import math
import numpy as np

def q(x, yi, delta):
    qij = 1 / (1 + math.exp((-2 * x * yi) / delta))
    return round(qij, 4)

# def r(data):
#     rji = 1
#     for i in range(len(data)):
#         r_com = 1-2*data[i]
#         rji = rji*r_com
#     rji = round(1/2 + (1/2)*rji, 3)
#     if rji > 0:
#         alpha = 1
#     else:
#         alpha = -1
#     beta = abs(rji)
#     return alpha, beta

def LLR_qij(a, b):
    return round(math.log(a/b), 4)

def phi(x):
    return math.log((math.exp(x)+1)/(math.exp(x)-1))

def LLR_rji(data):
    alpha = []
    beta = []
    for item in data:
        if item > 0:
            alpha.append(1)
            beta.append(abs(item))
        else:
            alpha.append(-1)
            beta.append(abs(item))
    he = 0
    for thing in beta:
        he += phi(thing)
    return round(np.product(alpha)*phi(he), 4)

def LLR_q_update(L_Xi, r_data):
    return round(L_Xi + sum(r_data), 4)


c = phi(1)
# print(c)

