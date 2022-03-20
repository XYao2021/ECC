import itertools
import random
import numpy as np
# from functions import max_star, Gamma_star, sys_state_generator, non_sys_state_generator, trellis_map_generator, awgn, code_generator, to_binary, awgn_normal, BCJR_decoder
from functions import *

"initialization"
G = [7, 5]
# G = [37, 21]
# snr_dB = -1
snr_dB = -1
R = 1/4
info_length = 5000
divide_length = info_length/5
code_rate = 1/2

"setup"
g_0, g_1 = to_binary(G)
# print(g_0, g_1)
n = int(max(len(g_0), len(g_1))) - 1
SNR = snr_dB*R  #signal SNR
snr_d = 10**(snr_dB/10)
# snr_d = 1/4
u = list(np.random.randint(0, 2, info_length))
states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)
# codeword = np.array(code_generator(u, n, states, nsd, out))
# r = awgn(codeword, SNR, L=1)
# r = awgn_normal(codeword, 0, 1)
# r = [+ 1.5339, +0.6390, -0.6747, -3.0183, +1.5096, +0.7664, -0.4019, +0.3185, +2.7121, -0.7304, +1.4169, -2.0341, +0.8971, -0.3951, +1.6254, -1.1768, +2.6954, -1.0575]
# r = [+0.8, +0.1, +1.0, -0.5, -1.8, +1.1, +1.6, -1.6]
# r = [+0.8, +0.1, -1.2, +1.0, -0.5, +1.2, -1.8, +1.1, +0.2, +1.6, -1.6, -1.1]
r = [1.2, -0.8, -0.2, 1.5, -0.3, -0.6, 1.1, 2, 2.5, 0.7, -0.2, 0.1, -1.6, -1.3, -0.4, 0.9, -2, -0.25, 0.5, 0.4, 0.15, -1.5, 3, -0.9, 0.4, 2.2, -1.8, 1.4]  # case 1
# r = [2, -0.5, 1, 0.9, 1.2, -0.8, -1, -0.3, -0.5, 1.6, -0.8, 2, -0.9, -1.3, 0.6, 1.2, -1.6, 0.5, -1.4, -1.6, 0.3, 1.6, -0.2, 2.5, -3.2, 2, -1.4, 0.7, 2.2, -1.2, 2, -1.3, 1.6, -0.4, -1.6, 1.8, -1.8, -2.5, 1.1, -2]

K, N = int(len(r)/(1/code_rate)), len(r)
# K, N = int(len(r)/2), len(r)
La = [0 for i in range(K+n)]
# La = [1, -0.5, -1.5, 0, 0.8, -1.2, 2, -1.8]
# La.extend([0 for i in range(n)])
Lc = 4*snr_d
trellis_map = trellis_map_generator(n, K, nsd, states)
trellis_map_unterminated = trellis_map_generator_unterminated(n, K, nsd, states)

P1, P2 = [], []
info_bits = []
for i in range(K):
    if N % 3 == 0:
        P1.append(r[(3*i)+1])
        P2.append(r[(3*i)+2])
        info_bits.append(r[3*i])
    else:
        if i % 2 == 0:
            P1.append(r[(2 * i) + 1])
            P2.append(0)
            info_bits.append(r[2 * i])
        else:
            P1.append(0)
            P2.append(r[(2 * i) + 1])
            info_bits.append(r[2 * i])

# print(P1, len(P1))
# print(P2, len(P2))
# print(info_bits, len(info_bits))
shuffle_index = [12, 5, 9, 2, 10, 7, 1, 14, 6, 11, 3, 8, 4, 13]  #case 1
# shuffle_index = [5, 7, 16, 4, 1, 19, 10, 15, 3, 20, 12, 8, 14, 2, 17, 11, 18, 6, 13, 9]
# print(shuffle_index, len(shuffle_index))
info_bits_interleaved = [0 for i in range(len(shuffle_index))]
for k in range(len(shuffle_index)):
    info_bits_interleaved[k] = info_bits[shuffle_index[k] - 1]
r_interleaved = []
r_org = []
for j in range(len(P2)):
    r_org += [info_bits[j], P1[j]]
    r_interleaved += [info_bits_interleaved[j], P2[j]]
print('r_org: ', r_org)
print('r_interleaved', r_interleaved, '\n')

iter_num = 10
i = 0
BER = []
while i in range(iter_num):
    BCJR1 = BCJR_decoder(K, r_org, n, states, nsd, out, in_nsd, trellis_map, La, Lc, u, divide_length)
    extrinsic_1 = extrinsic(BCJR1[0], La, r_org, Lc)
    extrinsic_1_interleaved = [0 for i in range(len(extrinsic_1))]
    for j in range(len(shuffle_index)):
        extrinsic_1_interleaved[j] = extrinsic_1[shuffle_index[j] - 1]

    BCJR2 = BCJR_decoder(K, r_interleaved, n, states, nsd, out, in_nsd, trellis_map_unterminated, extrinsic_1_interleaved, Lc, u,
                         divide_length)
    extrinsic_2 = extrinsic(BCJR2[0], extrinsic_1_interleaved, r_interleaved, Lc)
    extrinsic_2_interleaved = [0 for i in range(len(extrinsic_2))]
    for j in range(len(shuffle_index)):
        extrinsic_2_interleaved[shuffle_index[j] - 1] = extrinsic_2[j]
    La = extrinsic_2_interleaved
    print(i, 'this is LLR 1 result: ', BCJR1[0])
    print(i, 'this is LLR 2 result: ', BCJR2[0])
    print(i, 'this is final extrinsic_1 value: ', extrinsic_1)
    print(i, 'this is final extrinsic_2 value: ', extrinsic_2)
    print(i, 'this is deinterleaved extrinsic_2 value: ', extrinsic_2_interleaved)
    print(i, 'new_La: ', La, '\n')
    L = []
    for k in range(len(extrinsic_2_interleaved)):
        if extrinsic_2_interleaved[k] < 0:
            L.append(0)
        elif extrinsic_2_interleaved[k] > 0:
            L.append(1)
    count = 0
    for m in range(len(L)):
        if L[m] != u[m]:
            count += 1
    ber = count/len(u)
    print(i, 'temp BER: ', ber)
    i += 1

# BCJR1 = BCJR_decoder(K, r_org, n, states, nsd, out, in_nsd, trellis_map, La, Lc, u, divide_length)
# extrinsic_1 = extrinsic(BCJR1[0], La, r_org, Lc)
# extrinsic_1_interleaved = interleaved_example(extrinsic_1)
#
# BCJR2 = BCJR_decoder(K, r_interleaved, n, states, nsd, out, in_nsd, trellis_map, extrinsic_1_interleaved, Lc, u, divide_length)
# extrinsic_2 = extrinsic(BCJR2[0], extrinsic_1_interleaved, r_interleaved, Lc)
# extrinsic_2_interleaved = interleaved_example(extrinsic_2)

# print('this is LLR 1 result: ', BCJR1[0])
# print('this is LLR 2 result: ', BCJR2[0], '\n')
# print('this is final extrinsic_1 value: ', extrinsic_1)
# print('this is final extrinsic_2 value: ', extrinsic_2_interleaved, '\n')
