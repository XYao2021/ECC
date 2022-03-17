import math
import numpy as np
import itertools
import random
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal, normal


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
    if len(G) > len(Generate_function):
        NG = [[] for i in range(len(Generate_function))]
        for i in range(len(Generate_function)):
            if len(G[2*i+1]) < n:
                for j in range(n-1):
                    G[2*1+1].insert(0, 0)
            NG[i] += G[2*i] + G[2*i+1]
        G = NG.copy()
    return G


def max_star(data):  # Max-log-MAP
    if len(data) < 2:
        result = data[0]
    elif len(data) == 2:
        max_value = max(data)
        result = round(max_value + math.log(1 + math.exp(-abs(data[0] - data[1]))), 4)
    elif len(data) > 2:
        d = data[0:2]
        for i in range(2, len(data)):
            d_max = round(max(d) + math.log(1 + math.exp(-abs(d[0] - d[1]))), 4)
            d = [d_max, data[i]]
            # print(i, d)
        result = round(max(d) + math.log(1 + math.exp(-abs(d[0] - d[1]))), 4)
    return result
#
# def max_star(data):  # Max-log-MAP
#     result = max(data)
#     return result

def Gamma_star(value1, la, lc, value2):
    if value1 == 0:
        v = -1
    else:
        v = 1
    result = round((((1 / 2) * v) * la + ((1 / 2) * lc) * value2), 4)
    return result

# author - Mathuranathan Viswanathan
# This code is part of the book Digital Modulations using Python
def awgn(s, SNRdB, L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal-to-noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
"""
    gamma = 10**(SNRdB/10)   #SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L*sum(abs(s)**2)/len(s)   #Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L*sum(sum(abs(s)**2))/len(s)  # if s is a matrix [MxN]
    N0 = P/gamma  # Find the noise spectral density
    if isrealobj(s):  # check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape)  # computed noise
    else:
        n = sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
    r = s + n  # received signal
    return r

def awgn_normal(s, mean, sigma):
    n = normal(mean, sigma, s.shape)
    r = s + n
    return r

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
        s_0.insert(0, int((0 + sum(f_b)) % 2))  # systematic and recursive
        s_1 = list(item)
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

def non_sys_state_generator(g_0, g_1, n):
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
        s_0 = list(item)
        s_0.insert(0, 0)
        out0_0 = 0
        out0_1 = 0
        for i in range(0, len(g_0)):
            out0_0 = out0_0 + s_0[i] * g_0[i]
            out0_1 = out0_1 + s_0[i] * g_1[i]
        out0 = [out0_0 % 2, out0_1 % 2]
        s_0_next = s_0
        s_0_next.pop()
        s_1 = list(item)
        s_1.insert(0, 1)
        out1_0 = 0
        out1_1 = 0
        for i in range(0, len(g_0)):
            out1_0 = out1_0 + s_1[i] * g_0[i]
            out1_1 = out1_1 + s_1[i] * g_1[i]
        out1 = [out1_0 % 2, out1_1 % 2]
        s_1_next = s_1
        s_1_next.pop()
        for p in range(len(out0)):
            if out0[p] == 0:
                out0[p] = -1
            elif out1[p] == 0:
                out1[p] = -1
        nsd[states.index(item)] = [s_0_next, s_1_next]
        out[states.index(item)] = [out0, out1]
        in_nsd[states.index(item)] = [[0], [1]]
    return states, nsd, out, in_nsd

def trellis_map_generator(n, K, nsd, states):
    ss = [[0 for _ in range(n)]]
    trellis = []
    for i in range(0, K-n):
        next_state = []
        for j in range(len(ss)):
            next_state = next_state + nsd[states.index(ss[j])]
        next_state = [i for n, i in enumerate(next_state) if i not in next_state[:n]]
        ss = next_state
        trellis.append(ss)
    for j in range(K-n, K):
        next_state = []
        for i in range(len(ss)):
            ns = ss[i].copy()
            ns.insert(0, 0)
            ns.pop()
            next_state.append(ns)
        next_state = [i for n, i in enumerate(next_state) if i not in next_state[:n]]
        ss = next_state
        trellis.append(ss)
    trellis.insert(0, [[0 for _ in range(n)]])
    return trellis

def trellis_map_generator_unterminated(n, K, nsd, states):
    ss = [[0 for _ in range(n)]]
    trellis = []
    for i in range(0, K):
        next_state = []
        for j in range(len(ss)):
            next_state = next_state + nsd[states.index(ss[j])]
        next_state = [i for n, i in enumerate(next_state) if i not in next_state[:n]]
        ss = next_state
        trellis.append(ss)
    trellis.insert(0, [[0 for _ in range(n)]])
    return trellis

def code_generator(u, n, states, nsd, out):
    next_state = [0 for _ in range(n)]
    codeword = []
    i = 0
    while i < len(u):
        i_d = states.index(next_state)
        if u[i] == 0:
            next_state = nsd[i_d][0]
            codeword = codeword + out[i_d][0]
        if u[i] == 1:
            next_state = nsd[i_d][1]
            codeword = codeword + out[i_d][1]
        i += 1
        if i == len(u):
            last_state = next_state
    t = 0
    while t < n:
        i_d = states.index(last_state)
        if u[t] == 0:
            last_state = nsd[i_d][0]
            codeword = codeword + out[i_d][0]
            # codeword.append(out[i_d]['0'][1])
        if u[t] == 1:
            last_state = nsd[i_d][1]
            # codeword.append(out[i_d]['1'][0])
            codeword = codeword + out[i_d][1]
        t += 1
    return codeword

def terminated_code_generator(u, n, states, nsd, out):
    next_state = [0 for _ in range(n)]
    codeword = []
    i = 0
    while i < len(u):
        i_d = states.index(next_state)
        if u[i] == 0:
            next_state = nsd[i_d][0]
            codeword = codeword + out[i_d][0]
            # codeword.append(out[i_d]['0'][1])
        if u[i] == 1:
            next_state = nsd[i_d][1]
            # codeword.append(out[i_d]['1'][0])
            codeword = codeword + out[i_d][1]
        i += 1
        if i == len(u):
            last_state = next_state
    t = 0
    while t < n:
        i_d = states.index(last_state)
        last_state.insert(0, 0)
        last_state.pop()
        codeword = codeword + out[i_d][0]
        u.append(0)
        t += 1
    return codeword, u

def unterminated_code_generator(u_1, n, states, nsd, out):
    next_state = [0 for _ in range(n)]
    codeword = []
    i = 0
    while i < len(u_1):
        i_d = states.index(next_state)
        if u_1[i] == 0:
            next_state = nsd[i_d][0]
            codeword = codeword + out[i_d][0]
        if u_1[i] == 1:
            next_state = nsd[i_d][1]
            codeword = codeword + out[i_d][1]
        i += 1
    return codeword

def BCJR_decoder(K, r, n, states, nsd, out, in_nsd, trellis_map, La, Lc, u, divide_length):
    Gamma_l = []
    for j in range(K):
        recv = r[(2 * j):(2 * j + 2)]
        gamma_l = []
        if j < K - n:
            for item in trellis_map[j]:
                next_state = nsd[states.index(item)]
                OUT = out[states.index(item)]
                in_out = in_nsd[states.index(item)]
                # print(j, item, next_state, OUT, in_out,'\n')
                for k in range(len(next_state)):
                    value = round(np.dot(recv, OUT[k]), 4)
                    gamma_l.append(
                        [in_out[k][0], item, next_state[k], Gamma_star(in_out[k][0], La[j], Lc, value)])
            Gamma_l.append(gamma_l)

        else:
            for things in trellis_map[j]:
                next_state = nsd[states.index(things)]
                OUT = out[states.index(things)]
                in_out = in_nsd[states.index(things)]
                # print(j, things, next_state, OUT, in_out, '\n')
                for k in range(len(next_state)):
                    if next_state[k] in trellis_map[j + 1]:
                        value = round(np.dot(recv, OUT[next_state.index(next_state[k])]), 4)
                        gamma_l.append([in_out[next_state.index(next_state[k])][0], things, next_state[k],
                                        Gamma_star(in_out[k][0], La[j], Lc, value)])
            Gamma_l.append(gamma_l)

    Gamma_beta = Gamma_l.copy()
    trellis_map_beta = trellis_map.copy()
    Gamma_beta.reverse()
    trellis_map_beta.reverse()

    alpha = [-500 for i in range(len(states) - 1)]
    alpha.insert(0, 0)
    initialize_value_alpha = -500
    initialize_value_beta = 0
    beta = [0 for i in range(len(states))]
    Alpha = [[] for i in range(K)]
    Beta = [[] for i in range(K)]

    for k in range(K):
        if k < n:
            n_alpha = alpha.copy()
            gamma_group_beta = [[] for i in range(len(states))]
            for thing in Gamma_l[k]:
                alpha[states.index(thing[2])] = n_alpha[states.index(thing[1])] + thing[3]
            for item in Gamma_beta[k]:
                gamma_group_beta[states.index(item[1])].append(beta[states.index(item[2])] + item[3])
            beta_new = []
            for j in range(len(gamma_group_beta)):
                if len(gamma_group_beta[j]) == 0:
                    gamma_group_beta[j] = [initialize_value_beta]
                beta_new.append(max_star(gamma_group_beta[j]))
            beta = beta_new.copy()
            Alpha[k] = alpha.copy()
            Beta[k] = beta.copy()
        elif K - n > k >= n:
            gamma_group = [[] for i in range(len(states))]
            gamma_group_beta = [[] for i in range(len(states))]
            for item in Gamma_l[k]:
                gamma_group[states.index(item[2])].append(round((alpha[states.index(item[1])] + item[3]), 4))
            for thing in Gamma_beta[k]:
                gamma_group_beta[states.index(thing[1])].append(round((beta[states.index(thing[2])] + thing[3]), 4))
            for i in range(len(gamma_group)):
                alpha[i] = max_star(gamma_group[i])
            for j in range(len(gamma_group_beta)):
                beta[j] = max_star(gamma_group_beta[j])
            Alpha[k] = alpha.copy()
            Beta[k] = beta.copy()
        elif k >= K - n:
            gamma_group = [[] for i in range(len(states))]
            gamma_group_beta = [[] for i in range(len(states))]
            for item in Gamma_l[k]:
                if item[2] in trellis_map[k + 1]:
                    gamma_group[states.index(item[2])].append(round((alpha[states.index(item[1])] + item[3]), 4))
            for thing in Gamma_beta[k]:
                gamma_group_beta[states.index(thing[1])].append(round((beta[states.index(thing[2])] + thing[3]), 4))
            alpha_new = []
            beta_new = []
            for i in range(len(gamma_group)):
                if len(gamma_group[i]) == 0:
                    gamma_group[i] = [initialize_value_alpha]
                alpha_new.append(max_star(gamma_group[i]))
            for j in range(len(gamma_group_beta)):
                if len(gamma_group_beta[j]) == 0:
                    gamma_group_beta[j] = [initialize_value_beta]
                beta_new.append(max_star(gamma_group_beta[j]))
            Alpha[k] = alpha_new.copy()
            Beta[k] = beta_new.copy()
            alpha = alpha_new.copy()
            beta = beta_new.copy()
    Alpha.insert(0, [0])
    Beta.insert(0, [0 for i in range(len(states))])
    Beta.reverse()
    # print('Alpha', Alpha, '\n')
    # print('Beta', Beta, '\n')

    L = []
    LLR = []
    BER = []
    for b in range(K):
        in_0, in_1 = [], []
        for item in Gamma_l[b]:
            if item[0] == 0:
                in_0.append(round((Alpha[b][states.index(item[1])] + Beta[b + 1][
                    states.index(item[2])] + item[3]), 4))
            elif item[0] == 1:
                in_1.append(round((Alpha[b][states.index(item[1])] + Beta[b + 1][
                    states.index(item[2])] + item[3]), 4))
        L_l = round((max_star(in_1) - max_star(in_0)), 4)
        LLR.append(L_l)
        if L_l < 0:
            L.append(0)
        elif L_l > 0:
            L.append(1)
        # if len(L) % divide_length == 0:   #change part for different length of codewords
        #     count = 0
        #     for i in range(len(L)):
        #         if L[i] != u[i]:
        #             count += 1
        #     if len(L) != 0:
        #         P = count/(len(L))
        #         BER.append([count, P])
        #         print('temporary BER: ', P)
    # print('this is LLR: ', LLR)
    # print('the information bits/: ', L, '\n', len(L))
    return LLR, L

def interleaved_example(m):
    m_new = m.copy()
    m_new[1] = m[2]
    m_new[2] = m[1]
    return m_new

def interleaved(P2, info_bits):
    random.shuffle(info_bits)
    r_interleaved = []
    for j in range(len(info_bits)):
        r_interleaved += [info_bits[j], P2[j]]
    return r_interleaved

def extrinsic(LLR, La, r, Lc):
    extrinsic_list = []
    # print(r)
    for i in range(len(LLR)):
        # print(i, 'this is value: ', LLR[i], La[i], Lc, r[2*i])
        extrinsic_value = LLR[i] - La[i] - Lc*r[2*i]
        extrinsic_list.append(round(extrinsic_value, 4))
    return extrinsic_list

def puncture_turbo_code_generator(u, shuffle_index, n, states, nsd, out):
    codeword, u1 = terminated_code_generator(u, n, states, nsd, out)
    v0, v1 = [], []
    for i in range(0, int(len(codeword) / 2)):
        v0.append(codeword[2*i])
        v1.append(codeword[2*i+1])
    u_1 = [0 for i in range(len(u1))]
    for j in range(len(u_1)):
        u_1[j] = u1[shuffle_index[j]]
    new_codeword = unterminated_code_generator(u_1, n, states, nsd, out)
    v2 = []
    for k in range(0, int(len(new_codeword) / 2)):
        v2.append(new_codeword[2 * k + 1])
    vp = []
    for m in range(int(len(v1)/2)):
        vp.append(v1[2*m])
        vp.append(v2[2*m+1])
    v = []
    for a in range(len(vp)):
        v.append(v0[a])
        v.append(vp[a])
    return v

def unpuncture_turbo_code_generator(u, shuffle_index, n, states, nsd, out):
    codeword, u1 = terminated_code_generator(u, n, states, nsd, out)
    v0, v1 = [], []
    for i in range(0, int(len(codeword) / 2)):
        v0.append(codeword[2 * i])
        v1.append(codeword[2 * i + 1])
    u_1 = [0 for i in range(len(u1))]
    for j in range(len(u_1)):
        u_1[j] = u1[shuffle_index[j]]
    new_codeword = unterminated_code_generator(u_1, n, states, nsd, out)
    v2 = []
    for k in range(0, int(len(new_codeword) / 2)):
        v2.append(new_codeword[2 * k + 1])
    v = []
    for a in range(len(v0)):
        v.append(v0[a])
        v.append(v1[a])
        v.append(v2[a])
    return v
