import os, sys
import numpy as np
import pylab as pl
from matplotlib.ticker import MultipleLocator
from itertools import cycle

'''
Script to plot the total intensity vs. power-law exponent for multiple Luneburg lens runs with different amplitude cutoffs.
Intensity on the y-axis corresponds to the sum of the total intensity output from the stepped_luneburg() function.
'''

###################################################################
### Update matplotlib rc params

params = {'axes.labelsize'          : 20,
		'font.size'         : 20,
		'legend.fontsize'   : 12,
		'xtick.labelsize'   : 18,
		'ytick.labelsize'   : 18,
		'xtick.major.size'  : 8,
                'ytick.major.size'  : 8,
		'xtick.minor.size'  : 3,
                'ytick.minor.size'  : 3,
		'xtick.direction'   : 'in',
		'ytick.direction'   : 'in',
		'text.usetex'       : True,
		'savefig.dpi'       : 100,
		'savefig.bbox'      : 'tight',
		'savefig.pad_inches': 0.05
}
pl.rcParams.update(params)

###################################################################

datadir  = [
    './amp_cut_0d1/',
    './amp_cut_0d01/',
    './amp_cut_0d001/',
    ]
savedir  = './'


layers     = 40 # must be an integer
plaw_exp   = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
nrays      = 1024.
amp_cut    = [
    0.1,
    0.01,
    0.001
    ]

inset = True # set to True for zoom in on best lens power-law exponent

### define output figure file name
figoutfile = "fig5.pdf"

###################################################################

### create figure
fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
pl.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)

### define color cycle for plotting
linecycler = cycle(['-','--',':'])

### define file list for plotting
infile_list = [('luneburg_out_%i_%.2f_%i_grid.dat'%(layers,exp,nrays)).replace('_0.','_0d') for exp in plaw_exp]

### loop through file list and add to plot
for directory in datadir:
    int_list = []
    for infile in infile_list:
        
        intensity = np.loadtxt(os.path.join(directory,infile),unpack=True,usecols=[6])
        
        ### calculate total intensity output from lens
        int_list.append(intensity.sum())
        
    ### plot theta vs. power-law exponent
    pl.plot(plaw_exp, int_list, next(linecycler), color='k', lw=1.0)

### add legend
legend = pl.legend(([r"$%g$"%j for j in amp_cut]),frameon=False,loc='lower left',handlelength=2,title="$\mathrm{amplitude}$ \n $\mathrm{cutoff}$")
pl.setp(legend.get_title(),fontsize='xx-small')

### minor ticks
yminticks = MultipleLocator(2)
pl.gca().yaxis.set_minor_locator(yminticks)

pl.xlim([min(plaw_exp),max(plaw_exp)])
pl.ylim([min(int_list)-5,max(int_list)+5])

pl.xlabel(r'$\mathrm{Power}$-$\mathrm{law \ Exponent}\ i\ \mathrm{in}\  n = (2-r^2)^{i}$')
pl.ylabel(r'$\mathrm{Total \ Intensity}$')

if(inset==True):
    ### define new inset axes
    ins = pl.axes([0.6, 0.6, .33, .33])

    ### define file list for plotting
    infile_list = [('luneburg_out_%i_%.2f_%i_grid.dat'%(layers,exp,nrays)).replace('_0.','_0d') for exp in plaw_exp]

    ### loop through file list and add to plot
    for directory in datadir:
        int_list = []
        for infile in infile_list:
            
            intensity = np.loadtxt(os.path.join(directory,infile),unpack=True,usecols=[6])
            
            ### calculate total intensity output from lens
            int_list.append(intensity.sum())
            
        ### plot theta vs. power-law exponent
        pl.plot(plaw_exp, int_list, next(linecycler), color='k', lw=1.0)

    
    ### set axes limits and ticks
    pl.ylim([608.5,611.5])
    pl.xlim([0.54, 0.56])
    pl.xticks([0.55])
    pl.yticks([609.,610.,611.])
    ### minor ticks
    yminticks = MultipleLocator(0.2)
    pl.gca().yaxis.set_minor_locator(yminticks)

print ""
print ">>> Saving enclosed intensity plot to "+os.path.join(savedir,figoutfile)
print ""
pl.savefig(os.path.join(savedir,figoutfile))
pl.show()
pl.close()
