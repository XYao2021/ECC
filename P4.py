import itertools
import random
import numpy as np
# from functions import max_star, Gamma_star, sys_state_generator, non_sys_state_generator, trellis_map_generator, awgn, code_generator, to_binary, awgn_normal, BCJR_decoder
from functions import *

"initialization"
# G = [27, 31]
# G = [7, 5]
G = [3, 2]
snr_dB = -6.02
R = 1/4
info_length = 2000
divide_length = info_length/5
code_rate = 1/3

"setup"
g_0, g_1 = to_binary(G)
n = int(max(len(g_0), len(g_1))) - 1
SNR = snr_dB*R  #signal SNR
snr_d = 10**(snr_dB/10)
# snr_d = 0.2
u = list(np.random.randint(0, 2, info_length))
states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)

# codeword = np.array(code_generator(u, n, states, nsd, out))
# r = awgn(codeword, SNR, L=1)
# r = awgn_normal(codeword, 0, 1)
# r = [+ 1.5339, +0.6390, -0.6747, -3.0183, +1.5096, +0.7664, -0.4019, +0.3185, +2.7121, -0.7304, +1.4169, -2.0341, +0.8971, -0.3951, +1.6254, -1.1768, +2.6954, -1.0575]
# r = [+0.8, -0.6, +1.2, -0.5, 2, -1, -1.5, 2.0, 1.3, -0.7]
# r = [0.6, -0.3, 0.2, 0.7, 0.2, 0.6, -0.4, -0.5, 0.3, -0.1, -0.8, -0.4, -0.7, 0.5, 0.4, 0.3, 0.8, 1, -1.2, -1.4]
r = [+0.8, +0.1, -1.2, +1.0, -0.5, +1.2, -1.8, +1.1, +0.2, +1.6, -1.6, -1.1]

K, N = int(len(r)/(1/code_rate)), len(r)
# K, N = int(len(r)/2), len(r)
La = [0 for i in range(K+n)]
# La = [2, -1, 0, 0.5, -0.7]
# La = [0, 1, -0.5, 0.8, 0.4, -0.8, 0.6, 1, 1.2, -1.4]
# La = [1, -0.5, -1.5, 0, 0.8, -1.2, 2, -1.8]
# La.extend([0 for i in range(n)])
Lc = 4*snr_d

trellis_map = trellis_map_generator(n, K, nsd, states)
unterminated_trellis_map = trellis_map_generator_unterminated(n, K, nsd, states)

# BCJR = BCJR_decoder(K, r, n, states, nsd, out, in_nsd, trellis_map, La, Lc, u, divide_length)
# print('the terminated LLR result: ', BCJR[0])
# BCJR1 = BCJR_decoder(K, r, n, states, nsd, out, in_nsd, unterminated_trellis_map, La, Lc, u, divide_length)
# print('the unterminated LLR result: ', BCJR1[0])
# BCJR1 = BCJR_decoder(K, r, n, states, nsd, out, in_nsd, unterminated_trellis_map, La, Lc, u, divide_length)
# print('this is LLR result: ', BCJR1[0])
r_org = []
P1, P2 = [], []
info_bits = []
for i in range(K):
    r_org += r[3*i:3*i+2]
    P1.append(r[(3*i)+1])
    P2.append(r[(3*i)+2])
    info_bits.append(r[3*i])
# print(r_org)
info_bits_interleaved = interleaved_example(info_bits)
r_interleaved = []
for j in range(len(P2)):
    r_interleaved += [info_bits_interleaved[j], P2[j]]
# print(r_interleaved)
#
BCJR1 = BCJR_decoder(K, r_org, n, states, nsd, out, in_nsd, trellis_map, La, Lc, u, divide_length)
# print('this is LLR 1 result: ', BCJR1[0])
extrinsic_1 = extrinsic(BCJR1[0], La, r_org, Lc)
# print('this is extrinsic_1 value: ', extrinsic_1, '\n')
extrinsic_1_interleaved = interleaved_example(extrinsic_1)
# print(extrinsic_1_interleaved)

BCJR2 = BCJR_decoder(K, r_interleaved, n, states, nsd, out, in_nsd, trellis_map, extrinsic_1_interleaved, Lc, u, divide_length)
# print('this is LLR 2 result: ', BCJR2[0])
extrinsic_2 = extrinsic(BCJR2[0], extrinsic_1_interleaved, r_interleaved, Lc)
# print('this is extrinsic_2 value: ', extrinsic_2, '\n')
extrinsic_2_interleaved = interleaved_example(extrinsic_2)
print('this is LLR 1 result: ', BCJR1[0])
print('this is LLR 2 result: ', BCJR2[0], '\n')
print('this is final extrinsic_1 value: ', extrinsic_1)
print('this is final extrinsic_2 value: ', extrinsic_2_interleaved, '\n')

