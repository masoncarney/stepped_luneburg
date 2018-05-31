import os, sys
import numpy as np
import pylab as pl

try:
    from enclosed_intensity import enc_int
except:
    raise ImportError("Cannot import module enc_int. Try updating the PYTHONPATH variable to include the path to the directory containing enclosed_intensity.py")

'''
Wrapper script for function enc_int(). Requires multiple files output from stepped_luneburg() to be read in and creates a new enclosed intensity output file for each iteration.

Default parameters: 
    enc_int(infile=None,outfile="luneburg_enc_int.dat",frac=0.5, mode=None, star_num=None,plot=True,figoutfile="luneburg_enc_int.eps"):
'''

###################################################################

# loop through power-law exponents and layers for Luneburg model

datadir = "/my/path/to/output/files/"

mode = 'grid'
nrays = 1024.
layers = [5.,10.,20.,40.]
plaw_exp = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
frac = 0.5

for i in range(len(layers)):
    for j in range(len(plaw_exp)):
        myinfile = (os.path.join(datadir,"luneburg_out_%i_%.2f_%i_%s.dat"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        myoutfile = (os.path.join(datadir,"luneburg_enc_int_%i_%.2f_%i_%s.dat"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        enc_int(infile=myinfile,outfile=myoutfile,frac=frac,mode=mode,plot=True)