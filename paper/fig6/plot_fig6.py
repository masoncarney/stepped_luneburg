import os, sys
import numpy as np
import pylab as pl

try:
    from intensity_map import int_map
except:
    raise ImportError("Cannot import module int_map. Try updating the PYTHONPATH variable to include the path to the directory containing intensity_map.py")

'''
Wrapper script for function int_map(). Requires multiple files output from stepped_luneburg() to be read in and creates a new intensity map output and figure file for each iteration.

Default parameters:
    int_map(infile=None,outfile="luneburg_int_map.dat",pixels=500.,frac=0.5,sky_mag=13.,figoutfile="luneburg_int_map.eps",mag_hist_plot=True,verbose=False)
'''

###################################################################

# loop through power-law exponents and layers for Luneburg model

datadir = "./"
plotdir = "./"

mode = 'grid'
nrays = 1024.
layers = 40.
plaw_exp = 0.55
pixels = 500.
frac = 0.5

myinfile = (os.path.join(datadir+"luneburg_out_%i_%.2f_%i_%s.dat"%(layers,plaw_exp,nrays,mode))).replace('_0.','_0d')
myoutfile = (os.path.join(datadir+"luneburg_int_map_%i_%.2f_%i_%s.dat"%(layers,plaw_exp,nrays,mode))).replace('_0.','_0d')
myfigoutfile = os.path.join(plotdir+"fig6.pdf")
int_map(infile=myinfile,outfile=myoutfile,pixels=pixels,frac=frac,figoutfile=myfigoutfile)