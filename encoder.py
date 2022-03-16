import itertools

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
    # print('last_state: ', last_state)
    while t < n:
        i_d = states.index(last_state)
        # print(t, out[i_d])
        # print(t, nsd[i_d], '\n')
        next_s = last_state.copy()
        next_s.insert(0, 0)
        next_s.pop()
        codeword = codeword + out[i_d][nsd[i_d].index(next_s)]
        u.append(in_nsd[i_d][nsd[i_d].index(next_s)][0])
        last_state = next_s.copy()
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
            # codeword.append(out[i_d]['0'][1])
        if u_1[i] == 1:
            next_state = nsd[i_d][1]
            # codeword.append(out[i_d]['1'][0])
            codeword = codeword + out[i_d][1]
        i += 1

    return codeword


# snr_dB = 0.2
R = 1/2
info_length = 2000
divide_length = info_length/5

g_0, g_1 = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1]]
n = int(max(len(g_0), len(g_1))) - 1
states, nsd, out, in_nsd = sys_state_generator(g_0, g_1, n)
print(states, '\n', nsd, '\n', out, '\n', in_nsd)

u = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
shuffle_index = [0, 8, 15, 9, 4, 7, 11, 5, 1, 3, 14, 6, 13, 12, 10, 2]

codeword, u1 = terminated_code_generator(u, n, states, nsd, out)
print(codeword[0], u1, len(u1), len(codeword), '\n')

v = []
for i in range(0, int(len(codeword)/2)):
    v.append(codeword[i])

# u1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
u_1 = [0 for i in range(len(u1))]
for j in range(len(u_1)):
    u_1[j] = u1[shuffle_index[j]]
print(u_1)

new_codeword = unterminated_code_generator(u_1, n, states, nsd, out)
print(new_codeword, len(new_codeword))

for k in range(0, int(len(new_codeword)/2)):
    print(new_codeword[2*k+1])






