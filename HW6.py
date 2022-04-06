from Functions import *

H = [[1, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0],
     [1, 0, 0, 1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1, 0, 0, 1]]

N = int(len(H[0]))
K = int(len(H))
M = N - K
delta = 0.5  # actually is delta**2
c = [0.2, 0.2, -0.9, 0.6, 0.5, -1.1, -0.4, -1.2]

check_function = [[] for _ in range(K)]
bit_function = [[] for _ in range(N)]
for i in range(K):
    check_function[i] = [i for i, x in enumerate(H[i]) if x == 1]
    for j in range(N):
        if H[i][j] == 1:
            bit_function[j].append(i)
print('check node connections: ', check_function)
print('bit node connections: ', bit_function, '\n')

L_Xi = []
for item in c:
    L_Xi.append(2 * item / delta)
# print(L_Xi)

L_qij = [[] for _ in range(K)]
for j in range(K):
    for k in range(len(check_function[j])):
        L_qij[j].append(L_Xi[check_function[j][k]])
        # L_qij[j].append(LLR_qij(q(1, c[check_function[j][k]], delta), q(-1, c[check_function[j][k]], delta)))
# print(L_qij, len(L_qij))

iter_num = 8
for m in range(iter_num):
    # print(m, L_qij)
    rji = [[] for _ in range(N)]
    for p in range(len(L_qij)):
        for i in range(len(L_qij[p])):
            compute_data = L_qij[p].copy()
            compute_data.remove(L_qij[p][i])
            rji[check_function[p][i]].append(LLR_rji(compute_data))
    # print(m, rji)
    L_qij_update = [[] for _ in range(K)]
    for b in range(len(rji)):
        for i in range(len(rji[b])):
            data = rji[b].copy()
            data.remove(rji[b][i])
            l_qij = L_Xi[b] + sum(data)
            L_qij_update[bit_function[b][rji[b].index(rji[b][i])]].append(round(l_qij, 4))
    # print(m, L_qij_update)
    L_Q = []
    for a in range(len(rji)):
        L_Q.append(LLR_q_update((2 * c[a]) / delta, rji[a]))
    print(f'LQ after {m+1} iterations', L_Q)
    L_qij = L_qij_update

    temp_decision = []
    for item in L_Q:
        if item < 0:
            temp_decision.append(1)
        elif item > 0:
            temp_decision.append(0)
    print(f'temp_decision after {m+1} iterations', temp_decision, '\n')
