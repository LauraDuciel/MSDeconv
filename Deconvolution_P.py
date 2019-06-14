#!/usr/bin/python
# -*- coding: UTF8 -*-

from __future__ import division, print_function
import sys
import functools
import numpy as np
import scipy
#from scipy.signal import fftconvolve
import matplotlib.pylab as plt
from isotope import isotopes as iso
iso.THRESHOLD = 1E-5 # devrait etre 1E-8 ! mais plus rapide
from isotope.proteins import averagine
from isotope import Fstore

PTM = None      # used to add a PTM to the proteins in the deconvolution pattern
                      # eg: dec.PTM = "Zn"  for a protein with Zinc
                      # set to PTM=None   if not needed
FF = Fstore.Fstore()  # initialize
warn = True
mzSTEP = 10.0    # size of a m/z window for which the isotopic pattern is supposed to be invariant

standard = False  # determines algorithmic detail - False is ~ 4 x faster.
AX = 10
def memoize(obj):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    From https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    '''
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


# isotopic pattern definition, estimated from mass, using averagine hypothesis
@memoize
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
    # distribution de la molecule
    U = FF.distrib(molecule)
    U.sort_by_mass()
    # determiner le signal de distribution
    deltaM = [ion.mass-U.distrib[0].mass for ion in U.distrib]
    S = sum([I.proba for I in U.distrib])
    intens = [ion.proba/S for ion in U.distrib]
    return (deltaM, intens)

###############################################################
#######################   Function  ###########################
###############################################################

###########################################################  1D

def samplez(axemass, deltamass, intensities, MassRef, z, width=0.0, irregular=False, transposed=False):
    """
    realizes the sampling of the isotopic pattern, described by deltamass, intensities,
    located at mass MassRef,
    on the axemass mass axis.
    width is an optional broadening factor (in m/z) generating gaussian blur.
    z is the charge of the molecule
    irregular flags non regular mass axes

    computed by Fourier transform, so it insures integral conservation.

    if transposed, compute a transposed version of it.
    """
    nn = len(axemass)
    offset = axemass[0]
    spwidth = (axemass[-1]-offset)*(nn)/(nn-1) #*(nn+1)/nn  # Nyquist frequency
    tax = np.hstack( [np.arange(nn), -np.arange(nn,0,-1)] )   /(2*spwidth)
    fid = np.zeros_like(tax)
    for mz,intens in zip(deltamass, intensities):
        if irregular:
            mcorr = mzcorr(axemass, mz/z+ MassRef/z)
            freq =  ( mcorr  - offset )
        else:
            freq =  ( mz/z + MassRef/z - offset )
        if freq<spwidth and freq>0:
            if intens>1E-5:
                fid += intens * np.cos(2*np.pi*freq*tax)
    w = max(4*spwidth/(nn), width)  # 4 pixel gaussian broadening is a minimum 
    fid *= np.exp(-(tax*w)**2)
    if transposed:
        spec = np.fft.rfft(fid)[1:].real
    else:
        spec = np.fft.rfft(fid)[:-1].real
    np.maximum(0, spec)     # remove negatives
    mspec = max(spec)
    spec[spec<(1E-3*mspec)] = 0.0   # clean small values
    spec /= spec.sum()   # normalise integral
    return spec

def samplezt(axemass, deltamass, intensities, MassRef, z, width=0.0, irregular=False):
    """
    realizes the sampling of the isotopic pattern, described by deltamass, intensities,
    located at mass MassRef,
    on the axemass mass axis.
    width is an optional broadening factor (in m/z) generating gaussian blur.
    z is the charge of the molecule
    irregular flags non regular mass axes

    computed by Fourier transform, so it insures integral conservation.
    """
    return samplez(axemass, deltamass, intensities, MassRef, z, width=width, irregular=irregular, transposed=True)

# If your code breaks here, it is because you forgot to do
# Deconvolution_fast.AX = axemass
# from your caller code.
# This speeds up considerably by easing the job for memoize.
# on a len(AX) == 4000, I get a x9 speed-up on my laptop.

@memoize
def Samplez(icenter, z, width=0.0, irregular=False):
    return _Samplez(AX, icenter, z, width=width, irregular=irregular)

def _Samplez(axemass, icenter, z, width=0.0, irregular=False):
    """same as pattern() + samplez() + roll() but returns FFT of spectrum"""
    center = axemass[icenter]
    dm,it = pattern(z*center)           # mean pattern
    ss = samplez(axemass, dm, it, z*center, z=z, width=width, irregular=irregular)
    ss = np.roll(ss,-icenter)    # move back mean pattern to origin
    r = np.fft.rfft(ss)
    return r

@memoize
def Samplezt(icenter, z, width=0.0, irregular=False):
    return _Samplezt(AX, icenter, z, width=width, irregular=irregular)

def _Samplezt(axemass, icenter, z, width=0.0, irregular=False):
    """same as pattern() + samplezt() + roll() but returns FFT of spectrum"""
    center = axemass[icenter]
    dm,it = pattern(z*center)           # mean pattern
    ss = samplezt(axemass, dm, it, z*center, z=z, width=width, irregular=irregular)
    ss = np.roll(ss,-icenter)    # move back mean pattern to origin
    r = np.fft.rfft(ss[::-1])
    return r

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
#    idx = np.searchsorted(axe, mz, side="left")
    m = axe[0] + (axe[-1]-axe[0])*idx/(len(axe)-1)
    return m

def scenePf(Axe, Zaxis, n=10, width=0.0, show=False):
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

def halfconv(x,Pat):
    "performs the convolution, using FFT, but Pat is already FFTed"
    res = np.fft.irfft(np.fft.rfft(x) * Pat)
    return res

def Cvlmz(Axe, x, z, width=0.0, irregular=False):
    """
    given a m/z axis,
    """
    imz1 = 0
    mz1 = Axe[imz1]
    step = mzSTEP
    y = np.zeros_like(Axe)
    xx = np.zeros_like(Axe)

    while mz1 < Axe[-1]:     # step over windows
        xx[:] = 0.0
        j = find_nearest(Axe, mz1+step)   # points will be treated in imz1..j
        xx[imz1:j] = x[imz1:j]            # copy values in window
        center = (mz1+step/2)               # mass and location of mean pattern used for convolution
        icenter = find_nearest(Axe, center)
        center = 1.0*Axe[icenter]

        if standard:
            dm,it = pattern(z*center)           # mean pattern
            ss = samplez(Axe, dm, it, z*center, z=z, width=width, irregular=irregular)
            ss = np.roll(ss,-icenter)    # move back mean pattern to origin
            y += conv(xx,ss)
        else:
            SS = Samplez(icenter, z=z, width=width, irregular=irregular)
            y += halfconv(xx,SS)

        mz1 += step
        imz1 = j
    return y

def tCvlmz(Axe, y, z, width=0.0, irregular=False):
    """
    transpose of Cvlmz()
    """
    imz1 = 0
    mz1 = Axe[imz1]
    step = mzSTEP   # m/z size of step
    x = np.zeros_like(Axe)
    yy = np.zeros_like(Axe)
    while mz1 < Axe[-1]:     # step over windows
        yy[:] = 0.0
        j = find_nearest(Axe, mz1+step)   # points will be treated in imz1..j
        yy[imz1:j] = y[imz1:j]   # copy values in window
        center = (mz1+step/2)               # mass and location of mean pattern used for convolution
        icenter = find_nearest(Axe, center)
        center = 1.0*Axe[icenter]

        if standard:
            dm,it = pattern(z*center)           # mean pattern
            ss = samplezt(Axe, dm, it, z*center, z=z, width=width, irregular=irregular)
            ss = np.roll(ss,-icenter)    # move back mean pattern to origin
            x += conv(yy,ss[::-1])
        else:
            SS = Samplezt(icenter, z=z, width=width, irregular=irregular)
            x += halfconv(yy,SS)

        mz1 += step
        imz1 = j
    return x


#Build transform y = Kx  / y = Convolmz(x)
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
        x[iz,:] = tCvlmz(Axe, y, z=z, width=width, irregular=irregular) 
    return x


###########################################################  2D
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

def Cvlmultiz2D(AxeM1, AxeM2, Zaxis1, Zaxis2, x, width1=0.0, width2=0.0, irregular=False):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    #y = np.zeros((N1,N2))
    y = np.zeros((N1,N2))
    xx = x.reshape((Z1,Z2,N1,N2))
    for i, z1 in enumerate(Zaxis1):
        for j, z2 in enumerate(Zaxis2):
            #y[i,j] += Cvlmz2D(AxeM1, AxeM2, xx[i,j,:,:], z1=z1, z2=z2, width1=width1, width2=width2)
            y += Cvlmz2D(AxeM1, AxeM2, xx[i,j,:,:], z1, z2, width1=width1, width2=width2, irregular=irregular)
    return y

def tCvlmultiz2D(AxeM1, AxeM2, Zaxis1, Zaxis2, y, width1=0.0, width2=0.0, irregular=False):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    x = np.zeros((Z1,Z2,N1,N2))
    #yy = y.reshape((N1,N2))
    yy = y.reshape((N1,N2))
    for i, z1 in enumerate(Zaxis1):
        for j, z2 in enumerate(Zaxis2):
            x[i,j,:,:] += tCvlmz2D(AxeM1, AxeM2, yy, z1, z2, width1=width1, width2=width2, irregular=irregular)
    return x

def _Cvlmz2D(AxeM1, AxeM2, x, z1, z2, width1=0.0, width2=0.0, irregular=False):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    nshift = 0
    mz1 = AxeM1[nshift]
    imz1 = 0
    step = mzSTEP
    y = np.zeros((N1,N2))
    xx = np.zeros((N1,N2))
    while mz1 < AxeM1[-1]:
        mz2 = AxeM2[nshift]
        imz2 = 0
        j1 = find_nearest(AxeM1, mz1+step)   # points will be treated in imz1..j
        center1 = (mz1+step/2)
        icenter1 = find_nearest(AxeM1, center1)
        while mz2 < AxeM2[-1]:     # step over windows
            xx[:,:] = 0.0
            j2 = find_nearest(AxeM2, mz2+step)
            center2 = (mz2+step/2)
            icenter2 = find_nearest(AxeM2, center2)
            dm,it = pattern(z2*center2)   # mean pattern over F2 step windows
            ss = samplez2D(AxeM1, AxeM2, dm, it, z1*center1, z2*center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
            ss = np.roll(ss,-icenter2)    # move back mean pattern to origin
            ss = np.roll(ss,-icenter1,axis=0)    # move back mean pattern to origin
            xx[imz1:j1,imz2:j2] = x[imz1:j1,imz2:j2]   # copy values in window
            y += conv2D(xx,ss)
            mz2 += step
            imz2 = j2
        mz1 += step
        imz1 = j1
    return y

def Cvlmz2D(AxeM1, AxeM2, x, z1, z2, width1=0.0, width2=0.0, irregular=False):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    nshift = 0
    mz1 = AxeM1[nshift]
    imz1 = 0
    step = mzSTEP
    y = np.zeros((N1,N2))
    xx = np.zeros((N1,N2))
    while mz1 < AxeM1[-1]:
        mz2 = AxeM2[nshift]
        imz2 = 0
        j1 = find_nearest(AxeM1, mz1+step)   # points will be treated in imz1..j
        center1 = (mz1+step/2)
        icenter1 = find_nearest(AxeM1, center1)
        while mz2 < AxeM2[-1]:     # step over windows
            xx[:,:] = 0.0
            j2 = find_nearest(AxeM2, mz2+step)
            center2 = (mz2+step/2)
            xx[imz1:j1,imz2:j2] = x[imz1:j1,imz2:j2]   # copy values in window

            if standard:
                icenter2 = find_nearest(AxeM2, center2)
                dm,it = pattern(z2*center2)   # mean pattern over F2 step windows
                ss = samplez2D(AxeM1, AxeM2, dm, it, z1*center1, z2*center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
                ss = np.roll(ss,-icenter2)    # move back mean pattern to origin
                ss = np.roll(ss,-icenter1,axis=0)    # move back mean pattern to origin
                y += conv2D(xx,ss)
            else:
                # AxeM1, AxeM2 should be global
                # SS = Samplez2D(AxeM1, AxeM2, center1, center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
                SS = Samplez2D_(center1, center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
                y += halfconv2D(xx,SS)
            
            mz2 += step
            imz2 = j2
        mz1 += step
        imz1 = j1
    return y

def tCvlmz2D(AxeM1, AxeM2, y, z1, z2, width1=0.0, width2=0.0, irregular=False):
    N1 = len(AxeM1)
    N2 = len(AxeM2)
    nshift = 0
    mz1 = AxeM1[nshift]
    imz1 = 0
    step = mzSTEP
    x = np.zeros((N1,N2))
    yy = np.zeros((N1,N2))
    while mz1 < AxeM1[-1]:
        mz2 = AxeM2[nshift]
        imz2 = 0
        j1 = find_nearest(AxeM1, mz1+step)   # points will be treated in imz1..j
        center1 = (mz1+step/2)
        icenter1 = find_nearest(AxeM1, center1)
        while mz2 < AxeM2[-1]:     # step over windows
            yy[:,:] = 0.0
            j2 = find_nearest(AxeM2, mz2+step)
            center2 = (mz2+step/2)
            yy[imz1:j1,imz2:j2] = y[imz1:j1,imz2:j2]   # copy values in window

            if standard:
                icenter2 = find_nearest(AxeM2, center2)
                dm,it = pattern(z2*center2)   # mean pattern over F2 step windows
                ss = samplez2Dt(AxeM1, AxeM2, dm, it, z1*center1, z2*center2, z1=z1, z2=z2, width1=width1, width2=width2,irregular=irregular)
                ss = np.roll(ss,-icenter2)    # move back mean pattern to origin
                ss = np.roll(ss,-icenter1, axis=0)    # move back mean pattern to origin
                x += conv2D(yy,ss[::-1,::-1])
            else:
                # AxeM1, AxeM2 should be global
                #SS = Samplez2Dt(AxeM1, AxeM2, center1, center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
                SS = Samplez2Dt_(center1, center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular)
                x += halfconv2D(yy,SS)

            mz2 += step
            imz2 = j2
        mz1 += step
        imz1 = j1
    return x

def samplez2D(axemass1,axemass2, deltamass, intensities, MassRef1, MassRef2, z1=1, z2=1, width1=0.0, width2=0.0, irregular=False, transposed=False):
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
    spwidth1 = (axemass1[-1]-offset1)*(nn1+1)/nn1
    spwidth2 = (axemass2[-1]-offset2)*(nn2+1)/nn2
    tax1 = np.hstack( [np.arange(nn1), -np.arange(nn1,0,-1)] )   /(2*spwidth1)
    tax2 = np.hstack( [np.arange(nn2), -np.arange(nn2,0,-1)] )   /(2*spwidth2)
    fid = np.zeros((len(tax1), len(tax2)))
    for mz,intens in zip(deltamass, intensities):
        if irregular:
            mcorr1 = mzcorr(axemass1, mz/z1+ MassRef1/z1)
            freq1 =  ( mcorr1  - offset1 )
            mcorr2 = mzcorr(axemass2, mz/z2+ MassRef2/z2)
            freq2 =  ( mcorr2  - offset2 )
        else:
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
    spec1 = np.fft.rfft(fid[:,0])[1:].real
    fn1 = len(spec1)
    spec2 = np.fft.rfft(fid[0,:])[1:].real
    fn2 = len(spec2)
    spec_t = np.zeros((len(tax1),fn2))
    spec = np.zeros((fn1,fn2))
    if transposed:
        for i in range(len(tax1)):
            spec_t[i,:] = np.fft.rfft(fid[i,:])[1:].real
        for i in range(fn2):
            spec[:,i] = np.fft.rfft(spec_t[:,i])[1:].real
    else:   # direct
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


def samplez2Dt(axemass1,axemass2, deltamass, intensities, MassRef1, MassRef2, z1=1, z2=1, width1=0.0, width2=0.0, irregular=False):
    """
    realizes the 2D sampling of the isotopic pattern, described by deltamass, intensities,
    located at mass MassRef1, MassRef2
    on the axemass1, axemass2 mass axes.
    width is an optional broadening factor (in m/z) generating gaussian blur.
    z1, z2, is the charge of the molecule

    computed by Fourier transform, so it insures integral conservation.
    """
    return samplez2D(axemass1,axemass2, deltamass, intensities, MassRef1, MassRef2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular, transposed=True)

# If your code breaks here, it is because you forgot to do
# Deconvolution_fast.AX1 = axemass1
# Deconvolution_fast.AX2 = axemass2
# from your caller code.
# This speeds up considerably by easing the job for memoize.
# on a 100x200, I get a x6 speed-up on my laptop.
@memoize
def Samplez2D(center1, center2, z1, z2, width1, width2, irregular, transposed=False):
    return _Samplez2D(AX1, AX2, center1, center2, z1, z2, width1, width2, irregular)
@memoize
def Samplez2Dt(center1, center2, z1, z2, width1, width2, irregular, transposed=False):
    return _Samplez2Dt(AX1, AX2, center1, center2, z1, z2, width1, width2, irregular)

def _Samplez2D(axemass1, axemass2, center1, center2, z1, z2, width1, width2, irregular, transposed=False):
    """same as pattern() + samplez2D() + roll() but returns FFT of spectrum"""
    icenter1 = find_nearest(axemass1, center1)
    icenter2 = find_nearest(axemass2, center2)
    dm,it = pattern(z2*center2)           # mean pattern
    ss = samplez2D(axemass1,axemass2, dm, it, z1*center1, z2*center2, z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular, transposed=transposed)
    ssr = np.roll(ss,-icenter2)    # move back mean pattern to origin
    ssr = np.roll(ssr,-icenter1,axis=0)    # move back mean pattern to origin
    if transposed:
        r = np.fft.rfft(ssr[::-1,::-1].ravel() )
    else:
        r = np.fft.rfft(ssr.ravel() )
    return r

def _Samplez2Dt(axemass1, axemass2, center1, center2, z1, z2, width1, width2, irregular):
    """same as pattern() + samplez2D() + roll() but returns FFT of spectrum"""
    return _Samplez2D(axemass1=axemass1, axemass2=axemass2, center1=center1, center2=center2,
        z1=z1, z2=z2, width1=width1, width2=width2, irregular=irregular, transposed=True)

def conv2D(x,pat):
    res = np.fft.irfft( np.fft.rfft(x.ravel()) * np.fft.rfft(pat.ravel()) )
    return res.reshape(x.shape)

def halfconv2D(x,Pat):
    "performs the convolution, using FFT, but Pat is already FFTed"
    res = np.fft.irfft(np.fft.rfft(x.ravel()) * Pat)
    return res.reshape(x.shape)

###############################################################
#######################   matrix  ###########################
###############################################################

###########################################################  2D
def scenePm(K, n=10, show=False):
    """
    build a typical scene, with n random species
    """
    N,M = K.shape
    Z = M/N
    x = np.zeros(M)
    j = np.random.randint(50, M-50, size=n)
    inten = 10 +90*abs(np.random.randn(n))
    x[j] = inten
    y = np.dot(K,x)
    x = x.reshape(Z,N)
    if show:
        print ("signal original x: ", N)
        print ("signal de mesure y: ", N)
        plt.figure()
        plt.subplot(211)
        plt.plot( x.T)
        plt.subplot(212)
        plt.plot( y)
    return x, y
def read(fname):
    "read a numpy object stored into a file and returns it"
    import tables
    h5file = tables.open_file(fname, mode='r')
    mat = h5file.root.mat.read()
    h5file.close()
    return mat
def store(fname, mat, compress=True):
    "store a numpy object into a hdf5 file"
    import tables
    h5file = tables.open_file(fname, mode='w', title="Test Array")
    root = h5file.root
    if compress:
        filters = tables.Filters(complevel=5, complib='zlib')
    else:
        filters = None
    ca = h5file.create_carray(root, "mat", atom=tables.Float64Atom(), shape=mat.shape, filters=filters)
    ca[:,:] = mat[:,:]
    h5file.close()
def buildK(axemass, zaxis, width=0, irregular=False):
    """
    build matrix multi-z isotopic dictionary
    here the isotopic pattern is adapted to each mass.

    """
    N = len(axemass)
    Z = len(zaxis)
    mat = np.zeros((N, N*Z))
    for i,m in enumerate(axemass):
        sys.stdout.write("\r%.1f%% %f"%(100.*i/N,m))
        for j,z in enumerate(zaxis):
            deltamass, intensities = pattern(z*m)
            mat[:,i+N*j] += samplez(axemass, deltamass, intensities, m*z, width=width, z=z, irregular=irregular)
    return mat

###########################################################  2D

def buildK2D(axemass1, axemass2, Zaxis1, Zaxis2, width=0, show=False):
    """
    build matrix multi-z isotopic dictionary
    here the isotopic pattern is adapted to each mass.

    """
    from scipy.sparse import csc_matrix
    N1 = len(axemass1)
    N2 = len(axemass2)
    Z1 = len(Zaxis1)
    Z2 = len(Zaxis2)
    mat =  scipy.sparse.lil_matrix((N1*N2,N1*N2))
    z1 = z2 = 1
    try:
        dic = np.load('my_Dictionary.npy').item()     #importer le dictionnaire
    except IOError:
        dic = {}
    name = "K_N%dx%d_M%.0f-%.0fx%.0f-%.0f_Z%d.h5"%(N1,N2,axemass1[0],axemass1[-1],axemass2[0],axemass2[-1],len(Zaxis1))
    if name in dic.keys():
        if show:
            print (name, "matrix exist in the current dictionary")
        mat = dic[name]
        return mat
    else:
        if show:
            print ("Sorry!", name, "not found. Please wait, a calculation is in progress")
        for i,m1 in enumerate(axemass1):
            sys.stdout.write("\r%.1f%% %f"%(100.*i/N1,m1))
            for j,m2 in enumerate(axemass2):
                dm, it = FF.pattern(z2*m2)
                s = samplez2D(axemass1, axemass2, dm, it, m1, m2, z1=z1, z2=z2)
                if np.isnan(s).any():
                    print ("NaN in ", i,j)
                mat[j+N2*i,:] = s.ravel()
        store(name, mat.toarray())
        print ("Stored to File ",name)
        return mat

def convol2D(Kmat, Sigma):
    "Implement the 2D isotopic convolution"
    S = Kmat.dot(Sigma.ravel())
    return S.ravel()

