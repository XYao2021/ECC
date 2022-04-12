from MIMO_Functions import *
from turbo_functions import *

Y = np.array([1+1.5j, 2-0.2j])
H = np.array([[-0.9+2j, 0.2+2j], [-0.3-1j, 2.5+0.5j]])
LA = [0, 0, 0, 0]
SNR_bit = 2  #dB
print(Y)
print(H)

LD, LD_extrinsic = MIMO_detector(H, Y, LA, SNR_bit)
print(LD, LD_extrinsic)


