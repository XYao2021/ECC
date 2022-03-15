import random
import math
import numpy as np
import itertools
# code_rate = 1/3
# r = [+0.8, +0.1, -1.2, +1.0, -0.5, +1.2, -1.8, +1.1, +0.2, +1.6, -1.6, -1.1]
# K, N = int(len(r)/(1/code_rate)), len(r)
#
# r_org = []
# r_interleaved = []
# for i in range(K):
#     r_org += r[3*i:3*i+2]
#     r_interleaved += [r[3*i], r[3 * i + 2]]
#
# print(r_org)
# print(r_interleaved)
#
# def interleaved_example(recv, k):
#     recv_list = []
#     for j in range(k):
#         recv_list.append(recv[2*j: 2*j+2])
#     recv_list_copy = recv_list.copy()
#     recv_list[1] = recv_list_copy[2]
#     recv_list[2] = recv_list_copy[1]
#
# def interleaved(recv, k):
#     recv_list = []
#     for j in range(k):
#         recv_list.append(recv[2*j: 2*j+2])
#     random.shuffle(recv_list)
#
# interleaved_example(r_interleaved, K)
# interleaved(r_interleaved, K)
# def max_star(data):  # Max-log-MAP
#     if len(data) < 2:
#         result = data[0]
#     else:
#         max_value = max(data)
#         abs_value = []
#         for i in range(len(data) - 1):
#             d = data.copy()
#             sub_list = d[i + 1: len(data)]
#             # print(sub_list, data[i])
#             abs_v = []
#             for z in range(len(sub_list)):
#                 abs_v.append(abs(data[i] - sub_list[z]))
#             abs_value = abs_value + abs_v
#         # print(max_value, abs_value, np.prod(abs_value))
#         # re = 0
#         # for j in range(len(abs_value)):
#         #     re += math.exp(-abs_value[j])
#         # print(math.log(1 + re))
#         # result = round(max_value + math.log(1 + re), 4)
#         result = round(max_value + math.log(1 + math.exp(-np.product(abs_value))), 4)
#         # print(math.log(1 + math.exp(-np.product(abs_value))))
#         # print(result, '\n')
#     return result


# [-1.2, 0.64, -1.84, -0.8] [-1.12, 1.36, 2.24, 0.72]
# a = max_star([-1.2, 0.64])
# b = max_star([a, -1.84])
# c = max_star([b, -0.8])

# a = max_star([-1.12, 1.36])
# b = max_star([a, 2.24])
def to_binary(Generate_function):
    G = []
    for item in Generate_function:
        if item >= 10:
            item = [int(item/10), item % 10]
            for i in range(len(item)):
                g = []
                # print(i, n_item[i], item, Generate_function.index(item))
                binary = format(item[i], 'b')
                for j in range(len(binary)):
                    if binary[j] == '0':
                        g.append(0)
                    elif binary[j] == '1':
                        g.append(1)
                G.append(g)
        else:
            g = []
            binary = format(item, 'b')
            for j in range(len(binary)):
                if binary[j] == '0':
                    g.append(0)
                elif binary[j] == '1':
                    g.append(1)
            G.append(g)
    n = len(max(G))
    # print(n)
    if len(G) > len(Generate_function):
        NG = [[] for i in range(len(Generate_function))]
        for i in range(len(Generate_function)):
            if len(G[2*i+1]) < n:
                for j in range(n-1):
                    G[2*1+1].insert(0, 0)
            NG[i] += G[2*i] + G[2*i+1]
        G = NG.copy()
    # print(G)
    return G
G = [27, 31]
# G = [7, 5]
to_binary(G)

def sys_state_generator(g_0, g_1, n):
    g_0_n = [0 for _ in range(len(g_0))]
    g_0_n[0] = 1
    fb = [i for i, x in enumerate(g_0) if x == 1]
    fb.remove(fb[0])
    states_org = list(itertools.product([0, 1], repeat=n))
    states = []
    for state in states_org:
        s = list(state)
        states.append(s)
    nsd = [0 for i in range(0, len(states))]
    out = [0 for i in range(0, len(states))]
    in_nsd = [0 for i in range(0, len(states))]
    for item in states:
        f_b = []
        for p in range(0, len(fb)):
            f_b.append(item[fb[p] - 1])
        s_0 = list(item)
        # s_0.insert(0, 0)
        s_0.insert(0, int((0 + sum(f_b)) % 2))  # systematic and recursive
        s_1 = list(item)
        # s_1.insert(0, 0)
        s_1.insert(0, int((1 + sum(f_b)) % 2))  # systematic and recursive
        out0_0 = out0_1 = out1_0 = out1_1 = 0
        for i in range(0, len(g_0)):
            out0_0 = out0_0 + s_0[i] * g_0[i]
            out0_1 = out0_1 + s_0[i] * g_1[i]
            out1_0 = out1_0 + s_1[i] * g_0[i]
            out1_1 = out1_1 + s_1[i] * g_1[i]
        out0 = [out0_0 % 2, out0_1 % 2]
        out1 = [out1_0 % 2, out1_1 % 2]
        s_0_next = s_0.copy()
        s_0_next.pop()
        s_1_next = s_1.copy()
        s_1_next.pop()
        for p in range(len(out0)):
            if out0[p] == 0:
                out0[p] = -1
        for q in range(len(out1)):
            if out1[q] == 0:
                out1[q] = -1
        nsd[states.index(item)] = [s_0_next, s_1_next]
        out[states.index(item)] = [out0, out1]
        # in_nsd[states.index(item)] = [[int((0 + sum(f_b)) % 2)], [int((1 + sum(f_b)) % 2)]]
        in_nsd[states.index(item)] = [[0], [1]]
    return states, nsd, out, in_nsd

# G = [27, 31]
# # G = [7, 5]
# a = to_binary(G)
# n = int(max(len(g_0), len(g_1))) - 1
# states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)
#
# for i in range(len(states)):
#     print(i, states[i])
#     print(i, nsd[i])
#     print(i, out[i])
#     print(i, in_nsd[i], '\n')
