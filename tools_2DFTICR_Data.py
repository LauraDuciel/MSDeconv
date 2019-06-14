#!/usr/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function, division
import time
import scipy
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import PDS
import matplotlib.pylab as plt
import numpy as np
import json
import math
import sys
import spike
from spike import FTICR
import Deconvolution_fast as dec
import isotopic_analysis as ia
from isotope import isotopes as iso
iso.THRESHOLD = 1E-5 # devrait être 1E-5 ! mais plus rapide
from isotope.proteins import averagine
from isotope import Fstore
import os


def pieceofdata(d, NX, NY, lowmassX=None, lowmassY=None, highmassX=1500, highmassY=700):
    """ 
    This function allows to get the coordinates of all pieces of the data you have.
    d is a SPIKE.FTICR dataset
    NX x NY is the size of one piece. (NX and NY are in points unit)
    lowmassX,lowmassY,highmassX,highmassY are the limits of the sampling - Warn: in m/z!
    It returns a table that contains all the pieces of data - of fixed size - contained between the limits.

    e.g. for a 262000 x 8200 total size in points, x = 262000 and y = 8200, if you take NX = 1024 and 
    NY = 256 you will get in return a table containing 8200 tuples of coordinates from the data-set.
    """
    warn1 = 0 #just a Flag used at the end
    warn2 = 0 #Flag also
    Tabl = [] #It will contain the coords of spectrum parts.
    #Setting of the limits where the spectrum needs to be analyzed. 
    #By default these are the lowmass (or - if too small - 500 m/z) and highmass on both axis.
    if lowmassX is None:
        lowmassX = d.axis2.lowmass
    lowmassX = max(lowmassX, 250)
    if lowmassY is None:
        lowmassY = d.axis1.lowmass
    lowmassY = max(lowmassY, 400)
    if highmassX is None:
        highmassX = d.axis2.highmass
    if highmassY is None:
        highmassY = d.axis1.highmass
    #lowmass and highmass are in m/z unit, this is why mztoi is used below to convert them in points.
    y = int(d.axis1.mztoi(highmassY))
    while y < int(d.axis1.mztoi(lowmassY)-NY): #compute the parts of spectrum
        x =  int(d.axis2.mztoi(highmassX))
        while x < int(d.axis2.mztoi(lowmassX))-NX:
            Tabl.append([ y, y + NY, x, x + NX])
            x += NX
        if x < int(d.axis2.mztoi(lowmassX)): #Only if the size (axis2) of the zone to cut is not a multiple of 
                                            #the size of one part (axis2). Then the last part will not be of same size.
            warn2 = 1
            Tabl.append([y, y + NY, x, x + (int(d.axis2.mztoi(lowmassX))-x)])
        y += NY
    if y < int(d.axis1.mztoi(lowmassY)): 
    #same thing than previously but for axis1, the resting part is added if the sizes are not "compatible". 
        warn1 = 1
        x =  int(d.axis2.mztoi(highmassX)) 
        while x < int(d.axis2.mztoi(lowmassX))-NX: 
            Tabl.append([y, y + (int(d.axis1.mztoi(lowmassY))-y), x, x + NX])
            x += NX 
        if x < int(d.axis2.mztoi(lowmassX)): 
        #The check needs to be perform for axis2 when completing axis1, some parts are again added here.
            warn2 = 1
            Tabl.append([y, y + (int(d.axis1.mztoi(lowmassY))-y), x, x + (int(d.axis2.mztoi(lowmassX))-x)])
    if warn1 != 0:
        print('Warn: some parts will not have the size required NX*NY as ysize was not a multiple of NY') 
        #to warn users that some parts will not have equal sizes.
    if warn2 != 0:
        print('Warn: some parts will not have the size required NX*NY as xsize was not a multiple of NX')

    return(Tabl)

def prepare(d, zoom):
    """
    Prepare the data part (=zoom) for solving.
    """
    
    if d.unit[0] == 'm/z':
        i1,j1,i2,j2 = mztoi(d,zoom)
    else:
        i1,j1,i2,j2 = zoom
        
    i1,j1,i2,j2 = min(i1,j1), max(i1,j1), min(i2,j2), max(i2,j2)
    
    buf = d.buffer[int(i1):int(j1),int(i2):int(j2)]
    AxeM1 = d.axis1.mass_axis()[int(j1):int((i1)):-1]
    AxeM2 = d.axis2.mass_axis()[int(j2):int((i2)):-1]
    
    return (buf[::-1,::-1], AxeM1, AxeM2)

def analysis(d, zone, width1=0.0, width2=0.0, iterations=100, noisefactor=1.0, computeKhi2=False, axeZ1=[1,2], axeZ2=[1,2]):
    """
    To analyze one small piece of the spectrum.
    d is the imported data-set.
    zone is the zone you want to analyze - in points - it is a list composed of the 4 coords.

    S2 which is returned is a rich object.
    """
    dec.memoize_clear()
    d.unit = 'points'
    ssi,AxeM1,AxeM2 = prepare(d, zone)
    print(zone)
    N1=len(AxeM1)
    N2=len(AxeM2)
    dec.AX1 = AxeM1
    dec.AX2 = AxeM2                                       
    S2 = ia.isotopic_analysis2D(AxeM1, AxeM2, ssi)    
    S2.axeZ1 = axeZ1 # list of z-values to analyze
    S2.axeZ2 = axeZ2  # list of z-values to analyze
    S2.width1 = width1
    S2.width2 = width2
    S2.nbiter = iterations
    S2.noisefactor = noisefactor
    S2.computeKhi2 = computeKhi2 
    S2.solve()
    S2.zone = zone
    del ssi
    return S2

def pp(S, z1, z2, threshold=1E6):
    """
    Peak pick on the result of deconvolution S, as returned by analysis().
    threshold is the detection threshold of peaks. Here chose to 1E6 but to be determine for each pp().

    Return lists of peaks coordinates (local maxima) on both axis and their intensities.

    WARN - To be changed -
        - Loop over the Z values
    """
    zone = S.zone #Getting the zone coordinates
    z1lo, z1up, z2lo, z2up = (zone[0],zone[1],zone[2],zone[3])
    buff = np.flip(S.result[z1,z2,:,:].real) #Getting the result spectrum after deconvolution - Here for one couple of Z values
    #np.flip is VERY IMPORTANT, otherwise, the plot is reversed, due tu an inversion of S.result
    tbuff = buff[1:-1,1:-1]  # truncated version for shifting
    listpk=np.where(((tbuff > threshold*np.ones(tbuff.shape))&    # thresholding
                    (tbuff > buff[:-2 , 1:-1]) & # laplacian - kind of
                    (tbuff > buff[2: , 1:-1]) &
                    (tbuff > buff[1:-1 , :-2]) &
                    (tbuff > buff[1:-1 , 2:])))
    listpkF1 = int(z1lo) + listpk[0] +1  #Adding the coordinates of the local maximum detected
    listpkF2 = int(z2lo) + listpk[1] +1        # absolute coordinates
    listint = tbuff[listpk[0], listpk[1]] #Adding the intensity of the peak at the previous found coordinates 
    return listpkF1, listpkF2, listint

def center2D(yx, yo, xo, intens, widthy, widthx):
    """
    the 2D centroid, used to fit 2D spectra - si center()
    xy is [x0, y_0, x_1, y_1, ..., x_n-1, y_n-1] - is 2*n long for n points,
    returns [z_0, z_1, ... z_n-1]
    NB: same as in Peaks from Spike package. No modifications.
    """
    y = yx[::2]
    x = yx[1::2]
    return intens*(1 - ((x-xo)/widthx)**2)*(1 - ((y-yo)/widthy)**2)

def centroid2D_OnDeconvoluted(S, z1,z2,listpkF1, listpkF2, listint, npoints_F1=3, npoints_F2=3):
    """
    S is an "analysis()" rich object, on which the peaks obtained from peak picking will be fitted.
    does the same as centroid2D, but in the local S coordinates rather than in dataset coord.

    ATTENTION - a changer
    - boucler sur Z
    - rendre les coord finales en m
    """
    from scipy import polyfit
    from scipy.optimize import curve_fit
    #coordinate changing functions
    def s_to_d(s1,s2):
        "change from 2D S local coord to dataset coord"
        i1,j1,i2,j2 = S.zone
        i1,j1,i2,j2 = min(i1,j1), max(i1,j1), min(i2,j2), max(i2,j2)
        return s1+i1, s2+i2
    def d_to_s(d1, d2):
        "change from 2D dataset coord to S local coord"
        i1,j1,i2,j2 = S.zone
        i1,j1,i2,j2 = min(i1,j1), max(i1,j1), min(i2,j2), max(i2,j2)
        return d1-i1, d2-i2
    widthF1 = []
    widthF2 = []
    listpkF1bis = []
    listpkF2bis = []
    listintbis = []
    nF1 = npoints_F1 #nF1 and nF2 are the number of points that will be explored around the original maximum.
    nF2 = npoints_F2
    noff1 = (int(nF1)-1)/2 #Next lines to check if the number of points is sufficient.
    noff2 = (int(nF2)-1)/2
    if (2*noff1+1 != nF1) or (nF1<3) or (2*noff2+1 != nF2) or (nF2<3):
        raise NPKError("npoints must odd and >2",data=S)
    for pk in range(len(listint)): #To find the best place - maximum - around to fit the centroid
        st1 = int(round(listpkF1[pk]-noff1))
        end1 = int(round(listpkF1[pk]+noff1+1))
        st2 = int(round(listpkF2[pk]-noff2))
        end2 = int(round(listpkF2[pk]+noff2+1))
        yxdata = np.array( [(y,x) for y in range(st1, end1) for x in range(st2, end2)] ).ravel() 
        #important for curve_fit to be able to work.
        # yx = np.array([[y,x] for y in range(1,4) for x in range(10,12)]).ravel()
        # => array([ 1, 10,  1, 11,  2, 10,  2, 11,  3, 10,  3, 11])  will be decoded by center2d
        sst1,sst2 = d_to_s(st1,st2) #Conversion 
        send1,send2 = d_to_s(end1,end2)
        zdata = S.result[z1,z2, sst1:send1, sst2:send2].ravel()
        try:
                                            # yx, yo, xo, intens, widthy, widthx
            popt, pcov = curve_fit(center2D, yxdata, zdata, p0=[listpkF1[pk], listpkF2[pk], listint[pk], 1.0, 1.0],
                bounds=([st1,st2,0,0.5,0.5], [end1, end2, 4*listint[pk], 2*npoints_F1, 2*npoints_F2 ]) ) 
                # Realizes the fit of centroid
            listpkF1bis.append(popt[0])  # A FAIRE: Passer en m
            listpkF2bis.append(popt[1])
            listintbis.append(popt[2])
            widthF1.append(popt[3])
            widthF2.append(popt[4])
        except RuntimeError:
            print ( "peak %d_%d centroid could not be fitted"%(listpkF1[pk],listpkF2[pk]))

    return(listpkF1bis,listpkF2bis,listintbis,widthF1,widthF2)

def multi_analysis(d, zones, iterations=100, axeZ1=[1,2],axeZ2=[1,2],width1=1.0,width2=0.8, save=True,noisefactor=1.0):
    """
    To analyze in 'one time' multiple pieces of the spectrum.
    d is the imported data-set.
    zones is a list of the zones you want to analyze - in points - each zone is a list composed of the 4 coords.

    loop over analysis, 
    if save is True, store the resulting analysis in a hdf5 file, else it does not save anything.
    """
    d.pks={}
    for i in range(0,len(zones)):
        dec.memoize_clear()
        print("***ZONE: ",zones[i],"***")
    # ATTENTION - A FAIRE: Definir width1,width2 dans la boucle pour chaque zone à déconvoluer
    #             A FAIRE: definir Z1 et Z2 pour etre sur que parent > fragment
        d.unit='points'
        S = analysis(d, zones[i], iterations=iterations, axeZ1=axeZ1, axeZ2=axeZ2, width1=width1,width2=width2,noisefactor=noisefactor)
        if save:
            fname = "Results_{0}_{1}_{2}_{3}.h5".format(*zones[i]) #Allows to store the results in compressed HDF5 files.
            #Useful to keep a trace a intermediate results during the process. 
            store(fname, S.result)
        thresh = S.result.max()/100.0 #Definition of the detection threshold - Maybe to optimize. 
        for z1 in S.axeZ1: #Loop over the Z values. (Z values being the charges of ions)
            for z2 in S.axeZ2:
                listpkF1, listpkF2, listint = pp(S,z1-1,z2-1,threshold = thresh)
                print("*** Z1, Z2, Nb peaks: ",z1-1,z2-1,len(listpkF1),"***")
                listpkF1bis,listpkF2bis,listintbis,widthF1,widthF2 = centroid2D_OnDeconvoluted(S,z1-1,z2-1,listpkF1, listpkF2, listint)
                d.pks["Peaks_Zone{0}{1}{2}{3}_Z1{4}_Z2{5}".format(*zones[i],z1,z2)]=[
                            listpkF1,
                            listpkF2,
                            listint,
                            [ d.axis1.itomz(p) for p in listpkF1bis], #Convert and store coordinates in mass
                            [ d.axis2.itomz(p) for p in listpkF2bis],
                            listintbis,
                            listint, #Store intensities
                            widthF1, #width values are stored to get an information on precision.
                            widthF2] 
    return 

def complete_deconv(d, prefix="TEST",lowmassX=300, lowmassY=400, highmassX=1200, highmassY=700, iterations=100, axeZ1=[1,2],axeZ2=[1,2],dm1_m2=3.4E-6,dm2_m2=1.0E-7, save=True,noisefactor=0.9, NX=1.5*1024, NY=256):
    """
    To analyze in 'one time' all pieces of the spectrum in the indicated zone.
    d is the imported data-set.
    lowmassX=lowmassX, lowmassY=lowmassY, highmassX=highmassX, highmassY=highmassY are 4 coordinates to limit the part to analyze in the full spectrum.

    create chunks, determine widths and loop analysis() over chunks.
    if save is True, store the resulting analysis in a hdf5 file, else it does not save anything.
    """
    d.pks={}
    d.axeZ1 = axeZ1
    d.axeZ2 = axeZ2
    dirName = "{0}_DeconvResults".format(prefix)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    Tabl = pieceofdata(d, NX=NX, NY=NY, lowmassX=lowmassX, lowmassY=lowmassY, highmassX=highmassX, highmassY=highmassY)
    for N in range(0,len(Tabl)):
        print("***Chunk ", N+1," over ",len(Tabl),"***")
        dec.memoize_clear()
        zoom = Tabl[N]
        #Adding 2m/z border to the zone
        zoom_mz = list(isz(d,zoom))
        k = 2 #size of the safety border in m/z
        zoom_mz_safe = [zoom_mz[0]+k,zoom_mz[1]-k,zoom_mz[2]+k,zoom_mz[3]-k]
        mean_m1 = (zoom_mz_safe[0]+zoom_mz_safe[1])/2
        mean_m2 = (zoom_mz_safe[2]+zoom_mz_safe[3])/2
        width1 = round(dm1_m2*(mean_m1)**2, 2)
        width2 = round(dm2_m2*(mean_m2)**2, 2)
        print("***Obtained widths values: widht1=",width1," width2=", width2,"***")
        zoom_safe = list(mztoi(d,zoom_mz_safe)) #this is the list of coordinates with the safety border of 2m/z included
        zoom_safe = [math.ceil((zoom_safe[0]-1) // 2.) * 2,math.ceil((zoom_safe[1]+1) // 2.) * 2,math.ceil((zoom_safe[2]-1) // 2.) * 2,math.ceil((zoom_safe[3]+1) // 2.) * 2] #trick to make sure the limits are even, odd number create a bug with numpy rfft
#the following is to make sure we do not go bellow the given limits
        if zoom_safe[0] < math.ceil(d.axis1.mztoi(highmassY) // 2.) * 2:
            zoom_safe[0] = math.ceil(d.axis1.mztoi(highmassY) // 2.) * 2
        if zoom_safe[1] > math.ceil(d.axis1.mztoi(lowmassY) // 2.) *2:
            zoom_safe[1] = math.ceil(d.axis1.mztoi(lowmassY) // 2.) *2 + 2
        if zoom_safe[2] < math.ceil(d.axis2.mztoi(highmassX) // 2.) * 2:
            zoom_safe[2] = math.ceil(d.axis2.mztoi(highmassX) // 2.) * 2
        if zoom_safe[3] > math.ceil(d.axis2.mztoi(lowmassX) // 2.) *2:
            zoom_safe[3] = math.ceil(d.axis2.mztoi(lowmassX) // 2.) *2 + 2
        print("*** Currently analyzed zone: ",zoom_safe," in points, i.e. ",zoom_mz_safe," in m/z. ***")
    # ATTENTION - A FAIRE: definir Z1 et Z2 pour etre sur que parent > fragment
        d.unit='points'
        S = analysis(d, zoom_safe, iterations=iterations, axeZ1=d.axeZ1, axeZ2=d.axeZ2, width1=width1,width2=width2,noisefactor=noisefactor)
        print("Noise: ", S.noise, "Offset:", S.offset, "Khi2:", S.Khi2)
        if save:
            fname = "{0}_DeconvResults/_{1}_{2}_{3}_{4}.h5".format(prefix,*zoom_safe) #Allows to store the deconvolution results in compressed HDF5 files.
            store(fname, S.result)
        thresh = S.result.max()/100.0 #Definition of the detection threshold - Maybe to optimize. 
        for z1 in S.axeZ1: #Loop over the Z values. (Z values being the charges of ions)
            for z2 in S.axeZ2:
                listpkF1, listpkF2, listint = pp(S,z1-1,z2-1,threshold = thresh)
                print("*** Z1, Z2, Nb peaks: ",z1-1,z2-1,len(listpkF1),"***")
                listpkF1bis,listpkF2bis,listintbis,widthF1,widthF2 = centroid2D_OnDeconvoluted(S,z1-1,z2-1,listpkF1, listpkF2, listint)
                d.pks["Peaks_Zone{0}{1}{2}{3}_Z1{4}_Z2{5}".format(*zoom_safe,z1,z2)]=[listpkF1,
                                                                                    listpkF2,
                                                                                    listint,
                                                                                    [ d.axis1.itomz(p) for p in listpkF1bis], #Convert and store coordinates in mass
                                                                                    [ d.axis2.itomz(p) for p in listpkF2bis],
                                                                                    listintbis, #Store intensities
                                                                                    widthF1, #width values are stored to get an information on precision.
                                                                                    widthF2] 
    d.full_pks = {}
    for z1 in d.axeZ1:
        for z2 in d.axeZ2:
            d.full_pks["pks_{0}{1}_f1".format(z1,z2)] = []
            d.full_pks["pks_{0}{1}_f2".format(z1,z2)] = []
            d.full_pks["pks_{0}{1}_intensaftcent".format(z1,z2)] = []
            d.full_pks["pks_{0}{1}_intensbefcent".format(z1,z2)] = []
            d.full_pks["pks_{0}{1}_widthF2".format(z1,z2)] = []
            d.full_pks["pks_{0}{1}_widthF1".format(z1,z2)] = []
    for key in d.pks:
        for z1 in d.axeZ1:
            for z2 in d.axeZ2:
                if "Z1{0}_Z2{1}".format(z1,z2) in key:
                    for e in d.pks[key][3]: 
                        d.full_pks["pks_{0}{1}_f1".format(z1,z2)].append(e)
                    for e in d.pks[key][4]:
                        d.full_pks["pks_{0}{1}_f2".format(z1,z2)].append(e)
                    for e in d.pks[key][2]:
                        d.full_pks["pks_{0}{1}_intensbefcent".format(z1,z2)].append(e)
                    for e in d.pks[key][5]:
                        d.full_pks["pks_{0}{1}_intensaftcent".format(z1,z2)].append(e)
                    for e in d.pks[key][6]:
                        d.full_pks["pks_{0}{1}_widthF1".format(z1,z2)].append(e)
                    for e in d.pks[key][7]:
                        d.full_pks["pks_{0}{1}_widthF2".format(z1,z2)].append(e)
    with open('{0}_DeconvResults/full_deconv_pks_list.txt'.format(prefix), 'w') as outfile:  
        json.dump(d.full_pks, outfile)
    dec.memoize_clear()
    return 

def sz(d, z):
    """
    To get the size and resolution of a zone
    """
    a1,b1,a2,b2 = z
    i1 = d.axis1.mztoi(a1)
    j1 = d.axis1.mztoi(b1)
    i2 = d.axis2.mztoi(a2)
    j2 = d.axis2.mztoi(b2)
    sz1 = i1-j1
    sz2 = i2-j2
    print ("%d, %d, %d, %d  => %d x %d = %d"%(j1,i1,j2,i2, sz1,sz2,sz1*sz2))
    print ("resolutions : %.3f x %0.3f"%( (b1-a1)/sz1,(b2-a2)/sz2))

def isz(d, zone):
    """
    take a zone in index and convert in m/z.
    """

    a1,b1,a2,b2 = zone
    sz1 = abs(a1-b1)
    sz2 = abs(a2-b2)
    print ("%d, %d, %d, %d  => %d x %d = %d"%(a1,b1,a2,b2, sz1, sz2,sz1*sz2))
    i1 = d.axis1.itomz(a1)
    j1 = d.axis1.itomz(b1)
    i2 = d.axis2.itomz(a2)
    j2 = d.axis2.itomz(b2)
    print(i1,j1,i2,j2)
    print ("resolutions : %.3f x %0.3f"%( (i1-j1)/sz1,(i2-j2)/sz2))
    return (i1,j1,i2,j2)

def mztoi(d, z):
    """
    convert m/z window(=zone) to index
    """
    a1,b1,a2,b2 = z
    i1 = d.axis1.mztoi(a1)
    j1 = d.axis1.mztoi(b1)
    i2 = d.axis2.mztoi(a2)
    j2 = d.axis2.mztoi(b2)
    return (min(i1,j1), max(i1,j1), min(i2,j2), max(i2,j2) )

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
    with tables.open_file(fname, mode='w', title="Test Array") as h5file:
        root = h5file.root
        if compress:
            filters = tables.Filters(complevel=5, complib='zlib')
        else:
            filters = None
        ca = h5file.create_carray(root, "mat", atom=tables.Float64Atom(), shape=mat.shape, filters=filters)
        ca[...] = mat[...]

def _centroid2D(d,listpkF1, listpkF2, listint,npoints_F1=3, npoints_F2=3):
    """
    WARN: Not to use. Prefer using centroid2D_OnDeconvoluted().
    Perform a centroid on all peaks from original dataset d starting from lists of coordinates as returned by pp().
    NB: Adapted from Spike Package.
    """
    from scipy import polyfit
    from scipy.optimize import curve_fit
    widthF1 = []
    widthF2 = []
    listpkF1bis = []
    listpkF2bis = []
    listintbis = []
    nF1 = npoints_F1
    nF2 = npoints_F2
    noff1 = (int(nF1)-1)/2
    noff2 = (int(nF2)-1)/2
    if (2*noff1+1 != nF1) or (nF1<3) or (2*noff2+1 != nF2) or (nF2<3):
        raise NPKError("npoints must odd and >2 ",data=d)
    for pk in range(len(listint)):
        st1 = int(round(listpkF1[pk]-noff1))
        end1 = int(round(listpkF1[pk]+noff1+1))
        st2 = int(round(listpkF2[pk]-noff2))
        end2 = int(round(listpkF2[pk]+noff2+1))
        yxdata = np.array( [(y,x) for y in range(st1, end1) for x in range(st2, end2)] ).ravel()
        # yx = np.array([[y,x] for y in range(1,4) for x in range(10,12)]).ravel()
        # => array([ 1, 10,  1, 11,  2, 10,  2, 11,  3, 10,  3, 11])  will be decoded by center2d
        zdata = d.buffer[st1:end1, st2:end2].ravel()
        try:
                                            # yx, yo, xo, intens, widthy, widthx
            popt, pcov = curve_fit(center2D, yxdata, zdata, p0=[listpkF1[pk], listpkF2[pk], listint[pk], 1.0, 1.0],
                bounds=([st1,st2,0,0.5,0.5], [end1, end2, 4*listint[pk], 2*npoints_F1, 2*npoints_F2 ]) ) # fit
        except RuntimeError:
            print ( "peak %d_%d centroid could not be fitted"%(listpkF1[pk],listpkF2[pk]))
        listpkF1bis.append(popt[0])
        listpkF2bis.append(popt[1])
        listintbis.append(popt[2])
        widthF1.append(popt[3])
        widthF2.append(popt[4])
    return(listpkF1bis,listpkF2bis,listintbis,widthF1,widthF2)

def pp_and_center(S):
    """ A FAIRE """
    thresh = estimate_thr(S)
    listpkF1, listpkF2, listint = pp(S, threshold=thresh)
    allpkF1, allpkF2, allint, widthF1, widthF2 = centroid2D_OnDeconvoluted(S, listpkF1, listpkF2, listint)
    return allpkF1, allpkF2, allint, widthF1, widthF2


if __name__ == '__main__':
    dec.memoize_clear()
    data_name = "yeast_extract_p9kk_dn200_small_mr.msh5"
    d = FTICR.FTICRData(name=data_name, mode="onfile")
    #d._absmax = 76601098.15623298
    start = time.time()
    complete_deconv(d, prefix="YeastExtract_20190510_1024x256test",lowmassX=300, lowmassY=400, highmassX=1200, highmassY=700, 
        iterations=100, axeZ1=[1,2],axeZ2=[1,2],dm1_m2=3.4E-6,dm2_m2=1.0E-7, save=True,noisefactor=0.9, NX=1024, NY=256)
    end = time.time()
    dec.memoize_clear()
    print("Wall time: ", end-start)