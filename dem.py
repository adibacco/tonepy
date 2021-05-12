import time
import numpy as np
from math import pi as pi
from scipy import signal as sig

Fs = 245760000
fc = 40000000

syms_qpsk = np.array([ 1. + 1.j, -1. + 1.j, -1. - 1.j, 1. - 1.j ])
syms_avg = np.zeros(4, dtype=complex)

def decodeSymbol(rx, max_i, max_q):
    if (rx.real > max_i/2):
        if (rx.imag > max_q/2):
            syms_avg[0] = syms_avg[0] + rx
            return 0
        elif (rx.imag < -max_q/2):
            syms_avg[3] = syms_avg[3] + rx
            return 3
        else:
            return 10
    elif (rx.real < -max_i/2):
        if (rx.imag > max_q/2):
            syms_avg[1] = syms_avg[1] + rx
            return 1
        elif (rx.imag < -max_q/2):
            syms_avg[2] = syms_avg[2] + rx
            return 2        
        else:
            return 10
    else:
        return 10


def LPF(signal, fc, Fs):
    """Low pass filter, Butterworth approximation.

    Parameters
    ----------
    signal : 1D array of floats
        Signal to be filtered.
    fc : float
        Cutt-off frequency.
    Fs : float
        Sampling frequency.

    Returns
    -------
    signal_filt : 1D array of floats
        Filtered signal.
    W : 1D array of floats
        The frequencies at which 'h' was computed, in Hz. 
    h : complex
        The frequency response.
    """
    o = 5  # order of the filter
    fc = np.array([fc])
    wn = 2*fc/Fs

    [b, a] = sig.butter(o, wn, btype='lowpass')
    [W, h] = sig.freqz(b, a, worN=1024)

    W = Fs*W/(2*pi)

    signal_filt = sig.lfilter(b, a, signal)
    return(signal_filt, W, h)


plot = True
if plot:
    import matplotlib.pyplot as plt

#inputfile = 'rx_qpsk_30mbd_seq_0213.bin'
#inputfile = 'rx_sin_20MHz.bin'
inputfile = 'prova4.bin'
#inputfile= 'rx_qpsk_30mbd_HG.bin'

iq = np.fromfile(inputfile, dtype = np.dtype('<i2'))
iq = iq[0:16128:]

i = iq[::2] 
q = iq[1::2] 

print(len(i))
print(len(q))


i = i - np.mean(i)
q = q - np.mean(q)

amp_max = max(max(i), max(q))

maxi = np.max(i)
maxq = np.max(q)


print("Samples " + str(len(i)))


[i_filt, W, h] = LPF(i, fc, Fs)
[q_filt, W, h] = LPF(q, fc, Fs)

plt.plot(i, '.-')
plt.plot(q, '.-')
plt.grid()
plt.show()

samples_i = sig.resample_poly(i_filt, 4, 1)
samples_q = sig.resample_poly(q_filt, 4, 1)


si = samples_i[154::32]
sq = samples_q[154::32]

"""
for o in range(32):
    plt.plot(samples_i[o+150::32], samples_q[o+150::32], '.')
"""


plt.plot(si, sq, '.')
plt.show()

max_i = np.max(si)
max_q = np.max(sq)


c = si + 1j*sq


decodeSymbols = np.vectorize(decodeSymbol,  otypes=[np.int8])
rx_syms = decodeSymbols(c, max_i, max_q)
syms_avg = syms_avg/len(rx_syms)
plt.plot(rx_syms, '.')
plt.show()


exit()



#samples_i = signal.resample_poly(i, 8, 1)
#samples_q = signal.resample_poly(q, 8, 1)


amp_max = max(max(samples_i), max(samples_q))

samples = (samples_i + 1j*samples_q)

F_sample = 245760000
sps = 8

N = len(samples)
phase = 0
freq = 0

print("Samples " + str(N))

plt.plot(samples_i)
plt.show()




