import os, sys
import numpy as np
import pylab as pl

try:
    from luneburg_lens import stepped_luneburg
except:
    raise ImportError("Cannot import module stepped_luneburg. Try updating the PYTHONPATH variable to include the path to the directory containing luneburg_lens.py")

'''
Wrapper script for function stepped_luneburg(). Creates a separate Luneburg output file for each iteration.

Default parameters:
    stepped_luneburg(outfile="luneburg_output.dat",steps=20.,exp=0.5,nrays=100.,amp_cut=0.01,center=[0.,0.,0.],mode=None,modify=False,plot=True,figoutfile="luneburg_lens_raytrace.pdf",verbose=False,logfile="luneburg_log.dat")
'''

###################################################################

# loop through power-law exponents and layers for Luneburg model runs

datadir = "/my/path/to/output/files/"

mode = 'grid'
nrays = 1024.
layers = [5.,10.,20.,40.]
plaw_exp = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]

for i in range(len(layers)):
    for j in range(len(plaw_exp)):
        myoutfile = (os.path.join(datadir,"luneburg_out_%i_%.2f_%i_%s.dat"%(layers[i],plaw_exp[j],nrays,mode))).replace('_0.','_0d')
        stepped_luneburg(outfile=myoutfile,steps=layers[i],nrays=nrays,mode=mode,amp_cut=0.01,exp=plaw_exp[j],plot=False)
