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

datadir = "/my/path/to/output/files/"
plotdir = "/my/path/to/plots/"

mode = 'grid'
nrays = 1024.
layers = [5.,10.,20.,40.]
plaw_exp = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
pixels = 500.
frac = 0.5

for i in range(len(layers)):
    for j in range(len(plaw_exp)):
        myinfile = (os.path.join(datadir+"luneburg_out_%i_%.2f_%i_%s.dat"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        myoutfile = (os.path.join(datadir+"luneburg_int_map_%i_%.2f_%i_%s.dat"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        myfigoutfile = (os.path.join(plotdir+"luneburg_int_map_%ilyr_%.2fexp_%irays_%s.pdf"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        int_map(infile=myinfile,outfile=myoutfile,pixels=pixels,frac=frac,figoutfile=myfigoutfile)