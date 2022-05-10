import numpy as np
import math
from MIMO_Functions import *
from turbo_functions import *
import matplotlib.pyplot as plt
from tabulate import tabulate

N = 2  # number RX of antennas
M = 2  # number RX of antennas
R = 1/2
Mc = 2  # number of bits in each constellation symbol
Eb_dB = 2  # snr_Eb = snr_Es + 10 * math.log(N/(R*M*Mc), 10)
Es = 4
G = [7, 5]
info_length = 1280
divide_length = info_length/10

# Eb_linear = 10**(Eb_dB/10)
# sigma_square = (Es * N) / (2 * Eb_linear * R * M * Mc)
# print(sigma_square)
g_0, g_1 = to_binary(G)
n = int(max(len(g_0), len(g_1))) - 1
# u = list(np.random.randint(0, 2, info_length))
# # print(u)
# # u = [1, 1, 0, 1]
# states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)
#
# Turbo_interleaver = random.sample(list(np.arange(1, info_length+1, dtype=int)), info_length)
# # Turbo_interleaver = [2, 1, 4, 3]
# Channel_interleaver = random.sample(list(np.arange(1, (2*info_length)+1, dtype=int)), 2*info_length)
# # Channel_interleaver = [2, 1, 4, 3, 6, 5, 8, 7]
#
# V = puncture_turbo_code_generator(u, Turbo_interleaver, n, states, nsd, out, in_nsd)
# v = interleaver(Channel_interleaver, V)
# # print(v)
# S = np.split(np.array(QPSK_Mapping(v)), info_length/M)
# print(S)
BER = []
Eb = []
info = []
table = [['Eb_dB', 'Number of Codewords', 'BER']]
num_codeword = 10
while Eb_dB <= 3.1:
    Eb_linear = 10 ** (Eb_dB / 10)
    sigma_square = (Es * N) / (2 * Eb_linear * R * M * Mc)
    print(round(Eb_dB, 1), num_codeword)
    Count = 0
    for num_code in range(num_codeword):
        u = list(np.random.randint(0, 2, info_length))
        # print(u)
        # u = [1, 1, 0, 1]
        states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)

        Turbo_interleaver = random.sample(list(np.arange(1, info_length + 1, dtype=int)), info_length)
        # Turbo_interleaver = [2, 1, 4, 3]
        Channel_interleaver = random.sample(list(np.arange(1, (2 * info_length) + 1, dtype=int)), 2 * info_length)
        # Channel_interleaver = [2, 1, 4, 3, 6, 5, 8, 7]

        V = puncture_turbo_code_generator(u, Turbo_interleaver, n, states, nsd, out, in_nsd)
        v = interleaver(Channel_interleaver, V)
        # print(v)
        S = np.split(np.array(QPSK_Mapping(v)), info_length / M)
        H = []
        for s in range(len(S)):
            H.append(H_generator(M, N))
        H = np.array(H)
        Y = []
        for i in range(len(S)):
            # nn = np.array([np.random.normal(0, sigma_square, 2) for _ in range(N)])
            # noises = []
            # for noise in nn:
            #     noises.append(complex(noise[0], noise[1]))
            # Y.append(H[i] @ S[i] + np.array(noises))
            Y.append(H[i]@S[i]+np.array([complex(random.gauss(0, sigma_square/2), random.gauss(0, sigma_square/2)) for _ in range(N)]))
            # Y.append(H[i] @ S[i] + np.array(
            #     [complex(np.random.normal(0, sigma_square, 1), np.random.normal(0, sigma_square, 1)) for _ in range(N)]))
            # Y.append(H[i] @ S[i])
        Y = np.array(Y)

        LA = [[0 for _ in range(2*N)] for _ in range(len(Y))]
        # # LA = [1.2, -0.5, -1.5, 2]
        iter_num = 4
        for iter in range(iter_num):
            LD_extrinsic = []
            for i in range(len(Y)):
                LD, LD_e = MIMO_detector(H[i], Y[i], LA[i], sigma_square)
                LD_extrinsic += LD_e
            # print(iter, 'LD_extrinsic: ', LD_extrinsic, len(Channel_interleaver))
            LD_extrinsic_de = de_interleaver(Channel_interleaver, LD_extrinsic)
            # print(iter, 'MIMO detector output: ', LD_extrinsic_de, '\n')
            K, N1 = int(len(LD_extrinsic_de) / (1 / R)), len(LD_extrinsic_de)
            La = [0 for i in range(K)]
            # print(K+n)

            Le, soft = turbo_decoder(LD_extrinsic_de, K, N1, n, Eb_linear, La, nsd, states, out, in_nsd, Turbo_interleaver, R)
            Le_d = [0 for _ in range(len(Le))]
            for j in range(len(Channel_interleaver)):
                Le_d[j] = Le[Channel_interleaver[j] - 1]
            # print(iter, 'message back to MIMO: ', Le_d)
            LA = np.split(np.array(Le_d), int(len(Y)))
            LA = [list(i) for i in LA]
            # print(iter, 'new LA: ', len(LA), len(LA[0]), '\n')
        L = []
        for num in soft:
            if num <= 0:
                L.append(0)
            elif num > 0:
                L.append(1)
        # print(len(L))
        # print(L)
        # print(u)
        # print(soft)
        count = 0
        for j in range(len(L)):
            if L[j] != u[j]:
                count += 1
        Count += count
    Eb.append(round(Eb_dB, 1))
    ber = round(Count/(num_codeword*info_length), 6)
    BER.append(ber)
    info.append(num_codeword)
    table.append([Eb_dB, num_codeword, ber])
    print(round(Eb_dB, 1), 'BER: ', ber, Count)
    Eb_dB += 0.2
    if Eb_dB < 2.6:
        num_codeword += 20
    else:
        num_codeword *= 3

print(tabulate(table))
# print(x, EB)
plt.plot(Eb, BER, '--', color='blue')
# plt.legend()
plt.show()


