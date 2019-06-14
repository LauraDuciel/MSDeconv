from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import Deconvolution_fast as dec
FAST = True
if FAST:
    dec.standard = False
else:
    dec.standard = True
import PDS as PDS


class isotopic_analysis(object):
    """
    realize the isotopic analysis of a MS spectrum
    
    initialize
    - axeM_orig, numpy array: the mass axis
    - spectrum_orig  numpy array: intensity per mass values
    - mzmmin , mzmax determines the region to be analyzed
    - zmin, zmax determines the z range to be analyzed
    - sizeChunk size the chunks - experimental

    usage : 
    S = isotopic_analysis(mass_axis, intensity_values, m1, m2)
        to analyze from m1 to m2
        where mass_axis, intensity_values contain the experiment    S.tune_ap
    S.axeZ = [1,2,3]   # to tell which z are to be analyzed
    S.niter = 100  
    ... some tother tunable parameters
    S.solve()
    returns the multi-z analysis as a ZxMass numpy 2D array, and a m/z axis
        cut by small chunks and process them independently

    S.axeM is the mass axis == mass_axis
    S.result is the result spectrum
    so plot(S.axeM, S.result) will plot the deconvoluted spectrum

    tunable internal parameters
    - width = 0.0       an estimate of the line width in m/Z unit - important !
    - lamda = 0         the weight between entropy and l1 constraint
    - nbiter = 1000     the number of iterations
    - Nchunks           the number of chunks - computed automatically, but can be changed
    - chunklength       the chunk length  - computed automatically, but can be changed, should be even
    - axeZ              the list of Z-values to be analyzed
    - computeKhi2       if True, the normalized Khi2 is computed for each iteration and stored in S.Khi2 (slows down!)
    - verbose
    
    """
    def __init__(self, axeM_orig, spectrum_orig, mzmin, mzmax, axeZ, sizeChunk=None, verbose=False):
        m1, m2 = subrange(axeM_orig, mzmin, mzmax)
        if (m2-m1)%2 != 0:
            m2 = m2-1
        self.axeM = axeM_orig[m1:m2]
        self.spectrum = spectrum_orig[m1:m2]
        N = len(self.axeM)
        print ('Original data size: %d,  Processed Data size: %d'%(len(axeM_orig),N))
        self.axeZ = axeZ
        # initialize solver parameters
        self.width = 0.0   
        self.lamda = 0
        self.nbiter = 100
        self.PTM = None
        if sizeChunk is None:
            self.Nchunks = 1
            self.chunklength = N
        else:
            self.chunklength = sizeChunk
        if self.chunklength%2 != 0:
            self.chunklength = self.chunklength+1
            self.Nchunks = N // self.chunklength + 1
            print ('Nchunks:', self.Nchunks)
        self.noise, self.offset = findnoise_offset(self.spectrum)
        self.result = None
        self.computeKhi2 = False
        self.Khi2 = []
        self.verbose = verbose
        self.timer = True
        dec.gofast(self.axeM)

    def solve(self):
        """
        solves the complete problem chunk by chunk
        result are set in self.result
        self.Khi2 contains either
            - the final Khi2 if self.computeKhi2 is False     ( sqrt( sum( ((y_mes - y_calc)/noise)**2 ) ) )
            - the evolution of Khi2 over iterations if self.computeKhi2 is True
        """
        if self.verbose: print("solving for z-values:", self.axeZ)
        if self.timer: t0 = time.time()
        if self.PTM: dec.PTM = self.PTM
        Z = len(self.axeZ)
        N = len(self.axeM)
        X = np.zeros((Z,N))
        if self.computeKhi2:
            Khi2 = np.zeros(self.nbiter)  # used to compute global chi2
        else:
            Khi2 = 0.0
        for nch in range(self.Nchunks):
            start, end = self.computeChunks(nch)
            if self.verbose: print('chunk #%d/%d, from %.2f to %.2f'%(nch+1, self.Nchunks, self.axeM[start], self.axeM[end-1]))
            self._axeMframe = self.axeM[start:end]       # self._axeMframe is used in K1 and Kt1
            xx, ref = self.resolve(self.spectrum[start:end])
            X[:,start:end] += xx
            if self.computeKhi2:
                Nr = len(ref)
                Khi2[:Nr] += (ref**2)[:Nr]  # N can be > than len(Khi2list) if early convergence
            else:
                Khi2 += ref**2
        self.result = X
        self.Khi2 = np.sqrt( Khi2 )    # sqrt (sum(r^2))
        if self.timer: print("Elapsed time: %.1f sec"%(time.time()-t0))
        return self.result, self.Khi2

    def computeChunks(self, n):
        "returns axes of nth chunk"
        start = n*self.chunklength
        end = min((n+1)*self.chunklength, len(self.axeM))
        if (end-start)%2 != 0:
            end = end-1
        return start, end

    def resolve(self, y0):
        """
        resolve the problem locally on one chunk
        returns local solution and Khi2
        """
        Z = len(self.axeZ)
        N = len(self._axeMframe)
        if FAST:
            dec.AX = self._axeMframe
        scale = y0.max()
        y = y0/scale
        noise, offset = findnoise_offset(y)
# normK=np.sqrt(Z), 
        xrec_PDS, refspec = PDS.pds_vf(self.K1, self.Kt1, Z*N, y-offset, eta=noise*np.sqrt(N), normK=np.sqrt(Z), lamda=self.lamda, nbiter=self.nbiter, ref=None, Khi2=self.computeKhi2, show=False)
        return xrec_PDS.reshape(Z,N)*scale, refspec

    def K1(self, xx):
        Z = len(self.axeZ)
        N = len(self._axeMframe)
        x = xx.reshape(Z,N)  # xx is Z*N long
        w = self.width
        return dec.Cvlmultiz(self._axeMframe, self.axeZ, x, width=w, irregular=True)   #  result is N long
    def Kt1(self, y):
        w = self.width
        x = dec.tCvlmultiz(self._axeMframe, self.axeZ, y, width=w, irregular=True)
        return x.ravel()

    def _ycalc(self):
        self._axeMframe = self.axeM       # self._axeMframe is used in K1 and Kt1
        return self.K1(self.result)

    def ycalc(self):
        "try to compute y_calc the CORRECT way !"
        N = len(self.axeM)
        Y = np.zeros(N)
        for nch in range(self.Nchunks):
            start, end = self.computeChunks(nch)
            self._axeMframe = self.axeM[start:end]       # self._axeMframe is used in K1 and Kt1
            todo = self.result[:,start:end]
            noise, offset = findnoise_offset(self.spectrum[start:end])
            Y[start:end] += self.K1(todo-offset)+offset
        return Y

    def residue(self):
        return self.spectrum - self.ycalc()
    # display
    def plot_results(self):
        "display the results as one spectrum, using matplotlib"
        import matplotlib.pylab as plt
        fig, axes = plt.subplots()
        for i,z in enumerate(self.axeZ):
            axes.plot(self.axeM*z, self.result[i,:], label="Z=%d"%self.axeZ[i])
        axes.set_xlabel("monoisotopic mass")
        axes.legend()

    def multiplot_results(self, sharey=False):
        "display the results by Z values, using matplotlib"
        import matplotlib.pylab as plt
        Z = len(self.axeZ)
        fig, axes = plt.subplots(nrows=Z+1,sharex=True, sharey=sharey)
        axes[0].plot(self.axeM, self.spectrum, label="original")
        axes[0].plot(self.axeM, self.ycalc(), label="reconstructed")
        axes[0].plot(self.axeM, self.residue(), label="residue")

        axes[0].legend()
        ymax = self.result.ravel().max()
        for i,z in enumerate(self.axeZ):
            axes[i+1].plot(self.axeM, self.result[i,:], label="Z=%d"%self.axeZ[i])
            axes[i+1].legend()
            axes[i+1].set_ylim(ymax=1.05*ymax)
        axes[Z].set_xlabel("m/z")

class isotopic_analysis2D(isotopic_analysis):
    """
    realize the isotopic analysis of a 2D MS spectrum
    
    initialize
    - axeM_orig, numpy array: the mass axis
    - spectrum_orig  numpy array: intensity per mass values
    - mzmmin , mzmax determines the region to be analyzed
    - zmin, zmax determines the z range to be analyzed
    - sizeChunk size the chunks - experimental

    usage : 
    S = isotopic_analysis(mass_axis, intensity_values, m1, m2)
        to analyze from m1 to m2
        where mass_axis, intensity_values contain the experiment    S.tune_ap
    S.axeZ = [1,2,3]   # to tell which z are to be analyzed
    S.niter = 100  
    ... some tother tunable parameters
    S.solve()
    returns the multi-z analysis as a ZxMass numpy 2D array, and a m/z axis
        cut by small chunks and process them independently

    S.axeM is the mass axis == mass_axis
    S.result is the result spectrum
    so plot(S.axeM, S.result) will plot the deconvoluted spectrum

    tunable internal parameters
    - width = 0.0       an estimate of the line width in m/Z unit - important !
    - lamda = 0         the weight between entropy and l1 constraint
    - nbiter = 100      the number of iterations
    - Nchunks           the number of chunks - computed automatically, but can be changed
    - chunklength       the chunk length  - computed automatically, but can be changed, should be even
    - axeZ              the list of Z-values to be analyzed
    - computeKhi2       if True, the normalized Khi2 is computed for each iteration and stored in S.Khi2 (slows down!)
    - verbose
    
    """
    def __init__(self, axeM1, axeM2, spectrum_orig, zmin1=1, zmin2=1, zmax1=3,  zmax2=3):
        self.axeM1 = axeM1
        self.axeM2 = axeM2
        self.spectrum = spectrum_orig
        N1 = len(self.axeM1)
        N2 = len(self.axeM2)
        print ('Original data size: %d,  Processed Data size: %d'%(len(axeM1)*len(axeM2),N1*N2))
        self.axeZ1 = list(range(zmin1, zmax1+1))
        self.axeZ2 = list(range(zmin2, zmax2+1))
        # initialize solver parameters
        self.width1 = 0.0   
        self.width2 = 0.0   
        self.lamda = 0
        self.nbiter = 100
        self.noisefactor = 1.0
        #self.chunklength = sizeChunk
        #if self.chunklength%2 != 0:
        #    self.chunklength = self.chunklength+1
        #self.Nchunks = N // self.chunklength + 1

        #print ('Nchunks:', self.Nchunks)
        self.result = None
        self.computeKhi2 = False
        self.Khi2 = []
        self.verbose = True
        self.timer = True

    def solve(self):
        """
        solves the complete problem chunk by chunk
        result are set in self.result
        self.Khi2 contains either
            - the final Khi2 if self.computeKhi2 is False     ( sqrt( sum( ((y_mes - y_calc)/noise)**2 ) ) )
            - the evolution of Khi2 over iterations if self.computeKhi2 is True
        """
        if self.verbose: print("solving for z-values:", self.axeZ1, self.axeZ2)
        if self.timer: t0 = time.time()
        xx, ref = self.resolve(self.spectrum)
        self.result = xx
        self.Khi2 = ref

        if self.timer: print("Elapsed time: %.1f sec"%(time.time()-t0))
        return

    def computeChunks(self, n):
        "returns axes of nth chunk"
        start = n*self.chunklength
        end = min((n+1)*self.chunklength, len(self.axeM))
        if (end-start)%2 != 0:
            end = end-1
        return start, end

    def resolve(self, y0):
        """
        resolve the problem locally on one chunk
        returns local solution and Khi2
        """
        Z1 = len(self.axeZ1)
        N1 = len(self.axeM1)
        Z2 = len(self.axeZ2)
        N2 = len(self.axeM2)
        if FAST:
            dec.AX1 = self.axeM1
            dec.AX2 = self.axeM2
        scale = y0.max()
        y = y0/scale
        self.noise, self.offset = findnoise_offset(y.ravel())
# normK=np.sqrt(Z), 
        xrec_PDS, refspec = PDS.pds_vf(self.K1, self.Kt1, Z1*Z2*N1*N2, y.ravel()-self.offset, eta=self.noisefactor*self.noise*np.sqrt(N1*N2),
            normK=np.sqrt(Z1*Z2), lamda=self.lamda, nbiter=self.nbiter, ref=None, Khi2=self.computeKhi2, show=False)
        return xrec_PDS.reshape(Z1,Z2,N1,N2)*scale, refspec

    def K1(self, xx):
        Z1 = len(self.axeZ1)
        N1 = len(self.axeM1)
        Z2 = len(self.axeZ2)
        N2 = len(self.axeM2)
        x = xx.reshape(Z1,Z2,N1,N2)  # xx is Z1*Z2*N1*N2 long
        w1 = self.width1
        w2 = self.width2
        y = dec.Cvlmultiz2D(self.axeM1, self.axeM2, self.axeZ1, self.axeZ2, x, width1=w1, width2=w2, irregular=True) 
        return y.ravel()
        
    def Kt1(self, yy):
        Z1 = len(self.axeZ1)
        N1 = len(self.axeM1)
        Z2 = len(self.axeZ2)
        N2 = len(self.axeM2)
        y = yy.reshape(N1,N2)
        w1 = self.width1
        w2 = self.width2
        x = dec.tCvlmultiz2D(self.axeM1, self.axeM2, self.axeZ1, self.axeZ2, y, width1=w1, width2=w2, irregular=True)
        return x.ravel()

    def ycalc(self):
        return self.K1(self.result.ravel()).shape(N1,N2)

    def _ycalc(self):
        "try to compute y_calc the CORRECT way !"
        N = len(self.axeM)
        Y = np.zeros(N)
        for nch in range(self.Nchunks):
            start, end = self.computeChunks(nch)
            self._axeMframe = self.axeM[start:end]       # self._axeMframe is used in K1 and Kt1
            todo = self.result[:,start:end]
            noise, offset = findnoise_offset(self.spectrum[start:end])
            Y[start:end] += self.K1(todo-offset)+offset
        return Y

    def residue(self):
        return self.spectrum - self.ycalc()

    def multiplot_results(self):
        import matplotlib.pylab as plt
        Z1 = len(self.axeZ1)
        N1 = len(self.axeM1)
        Z2 = len(self.axeZ2)
        N2 = len(self.axeM2)
        fig,axlist = plt.subplots(ncols=Z1,nrows=Z2,sharex=True,sharey=True,figsize=(8,8))
        for i, ax1 in enumerate(axlist):
            for j, ax2 in enumerate(ax1):
                ax2.contour(self.axeM2, self.axeM1, self.result[i,j,:,:], label="Z1=%d Z2=%d"%(self.axeZ1[i],self.axeZ2[j]))

# utilities
def read_spect(filename):
    "read .xy or .csv data files"
    if filename.endswith('.xy'):
        return read_xy(filename)
    elif filename.endswith('.csv'):
        return read_csv(filename)

def read_csv( filename):
    " read a csv file using pandas"
    sp = pd.read_csv(filename, sep=',', header=None, names=('mz','int'))
    return sp.mz.__array__(), sp.int.__array__()
def read_xy( filename):
    " read a xy file using pandas"
    sp = pd.read_csv(filename, sep=' ', header=None, names=('mz','int'))
    return sp.mz.__array__(), sp.int.__array__()

def subrange( axe, frm, to):
    """
    computes a truncated mass axis
    """
    return [dec.find_nearest(axe,frm), dec.find_nearest(axe,to)+1]

# calcul de K fonctionnelle avec son adjoint
def verifT( K, KT, x, y):
    rslt1 = np.dot(y, K(x))
    rslt2 = np.dot(KT(y), x)
    rslt = rslt1 - rslt2
    return rslt

def findnoise_offset(fid, nbseg0 = 100):
    """
    Routine for determining the noise level from a numpy buffer "fid"
    cut the data in segments then make the standard deviation
    return makes the average on the 25% lowest segments
    
    nbseg=100   nb of segment to cut the spectrum
    """
    nbseg = min(nbseg0, len(fid)//100)  # no seg should be less than 100 points
    less = len(fid)%nbseg     # rest of division of length of data by nb of segment
    restpeaks = fid[less:]   # remove the points that avoid to divide correctly the data in segment of same size.
    newlist = np.array(np.hsplit(restpeaks, nbseg))    #Cutting in segments
    nlevels = newlist.std(axis = 1)
    olevels = newlist.mean(axis = 1)
    nlevels.sort()
    if nbseg < 4:
        noiselev = nlevels[0]
    else:
        noiselev = np.mean(nlevels[0:nbseg//4])
    olevels.sort()
    if nbseg < 4:
        offlev = olevels[0]
    else:
        offlev = np.mean(olevels[0:nbseg//4])
    return noiselev, offlev





