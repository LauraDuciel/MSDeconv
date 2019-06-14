#!/usr/bin/python

from __future__ import print_function
from __future__ import division
import spike
from spike import FTICR
import sys
import os
import time
import numpy as np
import scipy
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from isotope import isotopes as iso
iso.THRESHOLD = 1E-5 # devrait être 1E-5 ! mais plus rapide
from isotope.proteins import averagine
from isotope import Fstore
import Deconvolution_fast as dec
import isotopic_analysis as ia
import tools_2DFTICR_Data as tools2D
import PDS

dec.memoize_clear()
data_name = "/home/laura/Desktop/isotopic-deconv-basic-code_local/Data/yeast_2D_000002_dn200_mr.msh5"
d = FTICR.FTICRData(name=data_name, mode="onfile")
#d._absmax = 76601098.15623298
start = time.time()
tools2D.complete_deconv(d, prefix="Big_YeastExtract_20190612_3x1024x256",lowmassX=300, lowmassY=400, highmassX=1200, highmassY=700, 
	iterations=100, axeZ1=[1,2],axeZ2=[1,2],dm1_m2=3.4E-6,dm2_m2=1.0E-7, save=True,noisefactor=1.0, NX=3*1024, NY=256)
end = time.time()
dec.memoize_clear()
print("Wall time: ", end-start)
