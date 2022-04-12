import numpy as np
import math
from Functions import *

N = 2  # number RX of antennas
M = 2  # number RX of antennas
R = 1/2
Mc = 2  # number of bits in each constellation symbol
snr_Es = 2  # snr_Eb = snr_Es + 10 * math.log(N/(R*M*Mc), 10)
snr_Eb = snr_Es + 10 * math.log(N/(R*M*Mc), 10)

Y = np.array([1+1.5j, 2-0.2j])
H = np.array([[-0.9 + 2j, 0.2 + 2j], [-0.3-1j, 2.5+0.5j]])

LA = [0, 0, 0, 0]
# LA = [1.2, -0.5, -1.5, 2]

LD, LD_extrinsic = MIMO_detector(H, Y, LA, 10**(snr_Eb/10))
print('LD: ', LD)
print('LD_extrinsic: ', LD_extrinsic)

