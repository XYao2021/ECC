import math
import numpy as np
import itertools
from numpy .linalg import norm
import random

def interleaver(idx_list, data):
    shuffle_data = data.copy()
    for i in range(len(data)):
        shuffle_data[idx_list[i]] = data[i]
    return shuffle_data

def de_interleaver(idx_list, data):
    shuffle_data = data.copy()
    for i in range(len(data)):
        shuffle_data[i] = data[idx_list[i]]
    return shuffle_data

def H_generator(M, N):
    H = [[0 for _ in range(N)] for _ in range(M)]
    # print(H)
    for i in range(M):
        for j in range(N):
            H[i][j] = complex(random.gauss(0, 1 / 2), random.gauss(0, 1 / 2))
    return np.array(H)

def max_star(data):  # Max-log-MAP
    if len(data) < 2:
        return data[0]
    elif len(data) == 2:
        max_value = max(data)
        return round(max_value + math.log(1 + math.exp(-abs(data[0] - data[1]))), 4)
    elif len(data) > 2:
        d = data[0:2]
        for i in range(2, len(data)):
            d_max = round(max(d) + math.log(1 + math.exp(-abs(d[0] - d[1]))), 4)
            d = [d_max, data[i]]
            # print(i, d)
        return round(max(d) + math.log(1 + math.exp(-abs(d[0] - d[1]))), 4)
#
# def max_star(data):  # Max
#     return max(data)
def QPSK_Mapping(data):
    # print(data)
    S = []
    for k in range(int(len(data) / 2)):
        # print(data[2*k:2*k+2])
        if data[2*k:2*k+2] == [-1, -1]:
            S.append(complex(1, 1))
        if data[2*k:2*k+2] == [-1, 1]:
            S.append(complex(-1, 1))
        if data[2*k:2*k+2] == [1, 1]:
            S.append(complex(-1, -1))
        if data[2*k:2*k+2] == [1, -1]:
            S.append(complex(1, -1))
    # print(S, '\n')
    return S

def MIMO_detector(H, Y, LA, delta):  # QPSK: 00 -> 1; 01 -> j; 11 -> -1; 10 -> -j
    LD = []
    LD_extrinsic = []
    # print(len(LA))
    for i in range(len(LA)):
        LA_com = LA.copy()
        LA_com.remove(LA[i])
        max0, max1 = [], []
        # print(len(LA_com))
        for item in itertools.product([0, 1], repeat=len(LA_com)):
            item_map = []
            for bit in item:
                if bit == 0:
                    item_map.append(-1)
                elif bit == 1:
                    item_map.append(1)
            he = 0
            # print(item_map)
            for m in range(len(LA_com)):
                he += item_map[m]*LA_com[m]
            item_0 = list(item_map).copy()
            item_1 = list(item_map).copy()
            item_0.insert(i, -1)
            item_1.insert(i, 1)
            # print(item_0, item_1)
            S0 = QPSK_Mapping(item_0)
            S1 = QPSK_Mapping(item_1)
            # print(i, S0, S1)
            # S0 = np.split(np.array(item_0), len(H))
            # print(i, norm((Y - H @ S0), 2) ** 2)
            max0.append((-1 / (2 * delta)) * (norm((Y - H @ S0), 2) ** 2) + ((1 / 2) * he))
            # S1 = np.split(np.array(item_1), len(H))
            max1.append((-1 / (2 * delta)) * (norm((Y - H @ S1), 2) ** 2) + ((1 / 2) * he))
        # print(max_star(max1), max_star(max0), '\n')
        LD.append(LA[i] + max_star(max1) - max_star(max0))
        LD_extrinsic.append(max_star(max1) - max_star(max0))
    return LD, LD_extrinsic  # LD_extrinsic goes to Turbo Decoder 1

