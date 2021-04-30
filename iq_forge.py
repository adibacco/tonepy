import scipy as sp
import numpy as np
import sys, getopt

plot = False

if plot:
    import matplotlib.pylab as plt


def replicate(max_len, i, q):

    blen = len(i)*2*2

    i = np.tile(i, max_len // blen)
    q = np.tile(q, max_len // blen)

    return [i, q]


def trim_signal(max_samples, i, q):
    i = i[99::]
    q = q[99::]

    idx = max_samples-1
    while ((np.abs(i[idx] - i[0]) > 0.5) or (np.abs(q[idx] - q[0]) > 0.5) ): 
        idx = idx - 1


    i = i[0:idx:]
    q = q[0:idx:]

    return [i, q]


def main(argv):
    inputfile = ''
    outputfile = 'out'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('iq_forge.py -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('iq_forge.py -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg


    iq_freqs = [ 10000000, 10000000]
    iq_amps  = [ 25000,    25000]

    iq_phase = [ 0, 0]  # np.pi/2

    Fsample = 491520000
    Npoints = 30720
    ampmax = 32000

    t = np.linspace(0, Npoints/Fsample, Npoints)

    k = 0

    i = iq_amps[0]*np.sin(2*np.pi*iq_freqs[0]*t + iq_phase[0])
    q = iq_amps[1]*np.sin(2*np.pi*iq_freqs[1]*t + iq_phase[1])

    tlen = 122880 // 4

    blen = 15360 // 4
    

#    [i, q] = trim_signal(4096, i, q)
#    blen = len(i)
#    [i, q] = replicate(122880, i, q)
#    tlen = len(i) 



    iq = np.vstack((i, q)).ravel('F')

    dt = np.dtype('<i2')  
    iq.astype(dtype=dt).tofile(outputfile+'-'+str(blen*4)+'-'+str(tlen*4)+'-.bin')

    if plot:
        plt.plot(t, i)
        plt.plot(t, q)

        plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])

