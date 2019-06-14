
*This Program are provided under the Licence CeCILL 2.1*

    Authors : Laura Duciel, Afef Cherni, Marc-André Delsuc (madelsuc@unistra.fr)
	Date : June 2019

*This document is associated to the manuscript*

**Deconvolution of isotopic pattern in 2D-FTICR MS Spectrometry of peptides and proteins**

*Laura Duciel Afef Cherni and Marc-André Delsuc¨*

## Abstract
Mass Spectrometry (MS) is a largely used analytical technique in biology with applications such as the determination of molecule masses or the elucidation of structural data. 
Fourier Transform Ion Cyclotron Resonance MS is one implementation of the technique allowing high resolution and mass accuracy and based on trapping ions in circular orbits thanks to powerful Electromagnetic fields. 
This method has been extended to two-dimensional analysis, this gives signals containing a lot more information, for a reasonable amount of time and samples.
However, the data provided by such experiments cannot be stored in a classical manner and require some tools and architecture to store and compress data so that they remain accessible and usable without necessitating too much computer memory. 
The developed python program permits to turn these huge raw data into exploitable FTICR MS HDF5 formatted datasets which can be relatively rapidly and efficiently deconvolved and analysed using a Primal-Dual Splitting algorithm.

## Content
This folder contains the jupyter notebooks and code to reproduce the figures in the paper.

It requires in addition:

- Spike (Public library): pip install spike-py
- Isotope available at: https://bitbucket.org/delsuc/isotope

### Acknowledgements

This work has received funding from the European Union's Horizon 2020 research and Innovation programme under grant agreement EU-FTICR-MS No 731077. We thank Peter O'Connor for the gift of the dataset and enlighting discussion on the project, and Émilie Chouzenoux for advices and help on the algorithmic development.