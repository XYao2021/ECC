import numpy as np
import math
from MIMO_Functions import *
from turbo_functions import *

N = 2  # number RX of antennas
M = 2  # number RX of antennas
R = 1/2
Mc = 2  # number of bits in each constellation symbol
Eb_dB = 2  # snr_Eb = snr_Es + 10 * math.log(N/(R*M*Mc), 10)
Es = 4
G = [7, 5]
info_length = 5000
divide_length = info_length/5

Eb_linear = 10**(Eb_dB/10)
sigma_square = (Es * N) / (2 * Eb_linear * R * M * Mc)
# print(sigma_square)
g_0, g_1 = to_binary(G)
n = int(max(len(g_0), len(g_1))) - 1
snr_d = 10**(Eb_dB/10)
u = list(np.random.randint(0, 2, info_length))
states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)

Y = np.array([[1+1.5j, 2-0.2j], [-2+1j, -0.6+1.4j], [0.6-0.2j, 1.3-0.5j], [-1+0.8j, -1-0.4j]])
H = np.array([[[-0.9 + 2j, 0.2 + 2j], [-0.3-1j, 2.5+0.5j]], [[1j, -0.1+1j], [0.5+1j, 0.5j]], [[-0.9+0.7j, 1-0.2j], [0.5+0.5j, 0.3+0.2j]], [[-0.6-0.5j, -0.8+1j], [0.6-2j, -0.7+0.2j]]])
Turbo_interleaver = [2, 1, 7, 5, 3, 6, 8, 4]
Channel_interleaver = [3, 8, 14, 1, 5, 4, 10, 9, 11, 16, 15, 12, 13, 6, 7, 2]
# LA = [0, 0, 0, 0]
LA = [1.2, -0.5, -1.5, 2]

LD, LD_extrinsic = MIMO_detector(H, Y, LA, sigma_square)
# print('LD: ', LD)
# print('LD_extrinsic: ', LD_extrinsic)

r = [2, -0.5, 1, 0.9, 1.2, -0.8, -1, -0.3, -0.5, 1.6, -0.8, 2, -0.9, -1.3, 0.6, 1.2, -1.6, 0.5, -1.4, -1.6, 0.3, 1.6, -0.2, 2.5, -3.2, 2, -1.4, 0.7, 2.2, -1.2, 2, -1.3, 1.6, -0.4, -1.6, 1.8, -1.8, -2.5, 1.1, -2]
K, N = int(len(r) / (1 / R)), len(r)
La = [0 for i in range(K + n)]
# shuffle_index = [12, 5, 9, 2, 10, 7, 1, 14, 6, 11, 3, 8, 4, 13]  #case 1
shuffle_index = [5, 7, 16, 4, 1, 19, 10, 15, 3, 20, 12, 8, 14, 2, 17, 11, 18, 6, 13, 9]



