import scipy as sp
import matplotlib.pylab as plt
import numpy as np
import sys, getopt


def main(argv):
    inputfile = ''
    outputfile = 'out.bin'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('main.py -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg


    freqs = [ 10000000, 15000000, 17000000 ]
    amps  = [ 10000,    20000,    5000]

    Fsample = 491520000
    Npoints = 30720
    ampmax = 32604

    t = np.linspace(0, Npoints/Fsample, Npoints)

    k = 0
    for f in freqs:
        if (k == 0):
            i = amps[k]*np.cos(2*np.pi*f*t)
            q = amps[k]*np.sin(2*np.pi*f*t)
        else:
            i = i + amps[k]*np.cos(2*np.pi*f*t)
            q = q + amps[k]*np.sin(2*np.pi*f*t)
        k = k + 1

    mi = max(i)
    mq = max(q)

    i = i*(ampmax/mi)
    q = q*(ampmax/mq)


    iq = np.vstack((i, q)).ravel('F')

    dt = np.dtype('<i2')  
    iq.astype(dtype=dt).tofile(outputfile)


    plt.plot(t, i)

    plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])

