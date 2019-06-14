
from __future__ import division, print_function
import sys
import numpy as np
import scipy
from scipy.signal import fftconvolve
import isotopes as iso
iso.THRESHOLD = 1E-5 # devrait etre 1E-8 ! mais plus rapide
from proteins import averagine
import Fstore
import matplotlib 
import matplotlib.pylab as plt
import matplotlib.mlab as mlab

PTM = None      # used to add a PTM to the proteins in the deconvolution pattern
                      # eg: dec.PTM = "Zn"  for a protein with Zinc
                      # set to PTM=None   if not needed
FF = Fstore.Fstore()  # initialize
warn = True

# isotopic pattern definition, estimated from mass, using averagine hypothesis
def pattern(mass):
    """
    returns a isotopic pattern for a given protein mass
    return (deltaMass, intensities)
    """
    global warn
    if warn and PTM is not None:
        print("**** Warning, adding PTM : %s  ****"%(PTM,) )
        warn = False
    molecule = averagine(mass, mtype="monoisotop", PTM=PTM)
    U = FF.distrib(molecule)
    U.sort_by_mass()
    deltaM = [ion.mass-U.distrib[0].mass for ion in U.distrib]
    S = sum([I.proba for I in U.distrib])
    intens = [ion.proba/S for ion in U.distrib]
    return (deltaM, intens)

#############################################################################################
#############################################################################################
        ##########################     1D     ########################################
#############################################################################################
#############################################################################################

def samplez(axemass, deltamass, intensities, MassRef, z, width=0.0, irregular=False):
    """
    realizes the sampling of the isotopic pattern, described by deltamass, intensities,
    located at mass MassRef,
    on the axemass mass axis.
    width is an optional broadening factor (in m/z) generating gaussian blur.
    z is the charge of the molecule
    irregular flags non regular mass axes

    computed by Fourier transform, so it insures integral conservation.
    """
    nn = len(axemass)
    offset = axemass[0]
    spwidth = (axemass[-1]-offset)*(nn+1)/nn
    tax = np.hstack( [np.arange(nn), -np.arange(nn,0,-1)] )   /(2*spwidth)
    fid = np.zeros_like(tax)
    for mz,intens in zip(deltamass, intensities):
        if irregular:
            mcorr = mzcorr(axemass, mz/z+ MassRef/z)
            freq =  ( mcorr  - offset )
        else:
            freq =  ( mz/z + MassRef/z - offset )
        if freq<=spwidth and freq>0:
            if intens>1E-5:
                fid += intens * np.cos(2*np.pi*freq*tax)
    w = max(4*spwidth/(nn), width)  # 4 pixel gaussian broadening is a minimum 
    fid *= np.exp(-(tax*w)**2)
    spec = np.fft.rfft(fid)[:-1].real
    np.maximum(0, spec)     # remove negatives
    mspec = max(spec)
    spec[spec<(1E-3*mspec)] = 0.0   # clean small values
    spec /= spec.sum()   # normalise integral
    return spec

# to find the nearest pixel of a given mass in mass axis
def find_nearest(array,value):
    "from http://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array "
    import math
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def mzcorr(axe,mz):
    """
    for irregular axes, return a correct mz faking a regular axis
    """
    idx = find_nearest(axe,mz)
#   idx = np.searchsorted(axe, mz, side="left")
    m = axe[0] + (axe[-1]-axe[0])*idx/(len(axe)-1)
    return m

def sceneP(Axe, Zaxis, n=10, width=0.0, show=False):
    """
    build a typical scene, with n random species
    """
    N = len(Axe)
    Z = len(Zaxis)
    x = np.zeros(Z*N)
    j = np.random.randint(1, Z*N, size=n)
    inten = 10+90*abs(np.random.randn(n))
    x[j] = inten
    x = x.reshape(Z,N)
    y = Cvlmultiz(Axe, Zaxis, x, width=width)
    if show:
        print ("\n Original signal have the size =", x.shape)
        print ("\n Measured signal have the size =", y.shape)
    return x, y

def conv(x,pat):
    "performs the convolution, using FFT"
    res = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(pat))
    return res

def Cvlmz(Axe, x, z, width=0.0, irregular=False):
    """
    given a m/z axis,
    uses 20 m/z steps
    """
    nshift = 1   # this is the number of pixel by which we shift during building to avoid border effects
    mz1 = Axe[nshift]
    imz1 = 0
    step = 10.0
    y = np.zeros_like(Axe)
    xx = np.zeros_like(Axe)

    while mz1 < Axe[-nshift]:     # step over windows
        xx[:] = 0.0
        j = find_nearest(Axe, mz1+step)   # points will be treated in imz1..j
        dm,it = pattern(z*(mz1+step/2))   # mean pattern over step windows
        ss = samplez(Axe, dm, it, z*Axe[nshift], z=z, width=width, irregular=irregular)
        xx[imz1:j] = x[imz1:j]   # copy values in window
        y += conv(xx,ss)
        #y += np.convolve(xx,ss)
        mz1 += step
        imz1 = j
    return np.roll(y,-nshift)

def tCvlmz(Axe, y, z, width=0.0, irregular=False):
    """
    transpose of Cvlmz()
    """
    nshift = 1
    mz1 = Axe[nshift]
    imz1 = 0
    step = 10.0   # m/z size of step
    x = np.zeros_like(Axe)
    yy = np.zeros_like(Axe)

    while mz1 < Axe[-nshift]:     # step over windows
        yy[:] = 0.0
        j = find_nearest(Axe, mz1+step)   # points will be treated in imz1..j
        dm,it = pattern(z*(mz1+step/2))   # mean pattern over step windows
        ss = samplez(Axe, dm, it, z*Axe[nshift], z=z, width=width, irregular=irregular)
        yy[imz1:j] = y[imz1:j]   # copy values in window
        x += conv(yy,ss[::-1])
        mz1 += step
        imz1 = j
    return np.roll(x,nshift+1)

def Cvlmultiz(Axe, Zaxis, x, width=0.0, irregular=False):
    """
    as Cvlmz but multi-z
    x is a Z x N matrix
    """
    N = len(Axe)
    Z = len(Zaxis)
    if x.shape != (Z,N):
        raise Exception("size missmatch vector %s; axes %d x %d"%(x.shape, Z, N))
    y = np.zeros_like(Axe)
    xx = x.reshape((Z,N))
    for iz, z in enumerate(Zaxis):
        y += Cvlmz(Axe, xx[iz,:], z=z, width=width, irregular=irregular) 
    return y

def tCvlmultiz(Axe, Zaxis, y, width=0.0, irregular=False):
    """
    as tCvlmz but multi-z
    y is a N vector
    returns a Z x N matrix
    """
    if len(Axe) != len(y):
        raise Exception("size missmatch y: %d,   Axe: %d", len(y), len(Axe))
    x = np.zeros((len(Zaxis),len(Axe)))
    for iz, z in enumerate(Zaxis):
        x[iz,:] = tCvlmz(Axe, y, z=z, irregular=irregular) 
    return x

#############################################################################################
#############################################################################################
        ##########################     2D     ########################################
#############################################################################################
#############################################################################################

def samplez2D(axemass1,axemass2, deltamass, intensities, MassRef1, MassRef2, z1=1, z2=1, width1=0.0, width2=0.0):
    """
    realizes the 2D sampling of the isotopic pattern, described by deltamass, intensities,
    located at mass MassRef1, MassRef2
    on the axemass1, axemass2 mass axes.
    width is an optional broadening factor (in m/z) generating gaussian blur.
    z1, z2, is the charge of the molecule

    computed by Fourier transform, so it insures integral conservation.
    """
    nn1 = len(axemass1)
    nn2 = len(axemass2)
    offset1 = axemass1[0]
    offset2 = axemass2[0]
    spwidth1 = axemass1[-1]-offset1
    spwidth2 = axemass2[-1]-offset2
    tax1 = np.hstack( [np.arange(nn1), -np.arange(nn1,0,-1)] )   /(2*spwidth1)
    tax2 = np.hstack( [np.arange(nn2), -np.arange(nn2,0,-1)] )   /(2*spwidth2)
    fid = np.zeros((len(tax1), len(tax2)))
    for mz,intens in zip(deltamass, intensities):
        freq1 =  ( mz/z1 + MassRef1/z1 - offset1 )
        freq2 =  ( mz/z2 + MassRef2/z2 - offset2 )
        if freq1<=spwidth1 and freq1>0 and freq2<=spwidth2 and freq2>0:
            if intens>1E-5:
                for i in range(len(tax1)):
                    fid[i,:] += intens * np.cos(2*np.pi*freq2*tax2)*np.cos(2*np.pi*freq1*tax1[i])
    w1 = max(4*spwidth1/(nn1), width1)  # 4 pixel gaussian broadening is a minimum 
    w2 = max(4*spwidth2/(nn2), width2)  # 4 pixel gaussian broadening is a minimum 
    fid *= np.exp(-(tax2*w2)**2)
    fid = (fid.T*np.exp(-(tax1*w1)**2)).T
    # 2D FFT
    spec1 = np.fft.rfft(fid[:,0])[:-1].real
    fn1 = len(spec1)
    spec2 = np.fft.rfft(fid[0,:])[:-1].real
    fn2 = len(spec2)
    spec_t = np.zeros((len(tax1),fn2))
    spec = np.zeros((fn1,fn2))
    for i in range(len(tax1)):
        spec_t[i,:] = np.fft.rfft(fid[i,:])[:-1].real
    for i in range(fn2):
        spec[:,i] = np.fft.rfft(spec_t[:,i])[:-1].real
#    np.maximum(0, np.nan_to_num(spec))     # remove negatives and NaNs
    np.maximum(0, np.nan_to_num(spec))
    mspec = spec.max()
    spec[spec<(1E-3*mspec)] = 0.0   # clean small values
    ss = spec.sum()
    if ss >0.0:
        spec /= ss   # normalise integral
    return spec

def sceneP2D(Axe1, Axe2, Zaxis1, Zaxis2, n=20, show=False):
    """
    build a typical scene, with n random species
    """
    N1 = len(Axe1)
    N2 = len(Axe2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    x = np.zeros(Z1*N1*Z2*N2)
    j = np.random.randint(0, Z1*Z2, size=n)*N1*N2 +  np.random.randint(10, int(0.9*N1), size=n)*N2+np.random.randint(10, int(0.9*N2), size=n)
    inten = 80*abs(np.random.randn(n)) + 20
    x[j] = inten
    x = x.reshape((Z1,Z2,N1,N2))
    y = Cvlmultiz2D(Axe1, Axe2, Zaxis1, Zaxis2, x)
    if show:
        print ("\n x size=", x.shape)
        print ("\n y size=", y.shape)
    return x, y

def conv2D(x,pat):
    N1,N2 = x.shape
    res = np.fft.irfft(np.fft.rfft(x.reshape(N1*N2)) * np.fft.rfft(pat.reshape(N1*N2)))
    return res.reshape(x.shape)

def Cvlmultiz2D(AxeM1, AxeM2, Zaxis1, Zaxis2, x, width1=0.0, width2=0.0):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    #y = np.zeros((Z1,Z2,N1,N2))
    y = np.zeros((N1,N2))
    xx = x.reshape((Z1,Z2,N1,N2))
    for i, z1 in enumerate(Zaxis1):
        for j, z2 in enumerate(Zaxis2):
            #y[i,j,:,:] += Cvlmz2D(AxeM1, AxeM2, xx[i,j], z1, z2, width1=width1, width2=width2)
            y += Cvlmz2D(AxeM1, AxeM2, xx[i,j,:,:], z1, z2, width1=width1, width2=width2)
    return y

def tCvlmultiz2D(AxeM1, AxeM2, Zaxis1, Zaxis2, y, width1=0.0, width2=0.0):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    x = np.zeros((Z1,Z2,N1,N2))
    yy = y.reshape((N1,N2))
    #yy = y.reshape((Z1,Z2,N1,N2))
    for i, z1 in enumerate(Zaxis1):
        for j, z2 in enumerate(Zaxis2):
            x[i,j,:,:] += tCvlmz2D(AxeM1, AxeM2, yy, z1, z2, width1=width1, width2=width2)
    return x

def Cvlmz2D(AxeM1, AxeM2, x, z1, z2, width1=0.0, width2=0.0):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    nshift = 1
    mz1 = AxeM1[nshift]
    imz1 = 0
    step = 10.0
    y = np.zeros((N1,N2))
    xx = np.zeros((N1,N2))
    while mz1 < AxeM1[-nshift]:
        mz2 = AxeM2[nshift]
        imz2 = 0
        j1 = find_nearest(AxeM1, mz1+step)   # points will be treated in imz1..j
        while mz2 < AxeM2[-nshift]:     # step over windows
            xx[:,:] = 0.0
            j2 = find_nearest(AxeM2, mz2+step)
        #        print mz1, imz1, j
            dm,it = pattern(z2*(mz2+step/2))   # mean pattern over F2 step windows
            ss = samplez2D(AxeM1, AxeM2, dm, it, z1*AxeM1[nshift], z2*AxeM2[nshift], z1=z1, z2=z2, width1=width1, width2=width2)
            xx[imz1:j1,imz2:j2] = x[imz1:j1,imz2:j2]   # copy values in window
            y += conv2D(xx,ss)
            mz2 += step
            imz2 = j2
        mz1 += step
        imz1 = j1
    return np.roll(y,-nshift) 


def tCvlmz2D(AxeM1, AxeM2, y, z1, z2, width1=0.0, width2=0.0):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    nshift = 1
    mz1 = AxeM1[nshift]
    imz1 = 0
    step = 10.0
    x = np.zeros((N1,N2))
    yy = np.zeros((N1,N2))
    while mz1 < AxeM1[-nshift]:
        mz2 = AxeM2[nshift]
        imz2 = 0
        j1 = find_nearest(AxeM1, mz1+step)   # points will be treated in imz1..j
        while mz2 < AxeM2[-nshift]:     # step over windows
            yy[:,:] = 0.0
            j2 = find_nearest(AxeM2, mz2+step)
            dm,it = pattern(z2*(mz2+step/2))   # mean pattern over F2 step windows
            ss = samplez2D(AxeM1, AxeM2, dm, it, z1*AxeM1[nshift], z2*AxeM2[nshift], z1=z1, z2=z2, width1=width1, width2=width2)
            yy[imz1:j1,imz2:j2] = y[imz1:j1,imz2:j2]   # copy values in window
            x += conv2D(yy,ss[::-1,::-1])
            mz2 += step
            imz2 = j2
        mz1 += step
        imz1 = j1
    return np.roll(x,nshift+1)  

def _conv2D(x,pat):
    res = np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(pat))
    return res

