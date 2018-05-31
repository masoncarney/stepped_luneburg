import os, sys
import numpy as np
import pylab as pl
from matplotlib.ticker import MultipleLocator
from itertools import cycle

'''
Script to plot theta vs. power-law exponent for multiple Luneburg lens runs.
Theta corresponds to the angle containing 'frac' of normalized enclosed intensity from the enc_int() function.
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

datadir = "./"
savedir = "./"

layers     = [5,10,20,40] # must be integers
plaw_exp   = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
nrays      = 1024.
frac       = 0.5 # fraction of enclosed intensity for y-axis theta value

### define output figure file name
figoutfile = "fig4.pdf"

###################################################################

### create figure
fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
pl.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)

### define color cycle for plotting
pl.gca().set_color_cycle(['b','r','m','k','#FF8000','brown'])
#linecycler = cycle([':','-.','--','-'])

for i in range(len(layers)):

    ### define file list for plotting
    infile_list = [('luneburg_enc_int_%i_%.2f_%i_grid.dat'%(layers[i],exp,nrays)).replace('_0.','_0d') for exp in plaw_exp]

    ### loop through file list and add to plot
    theta_frac_list = []
    for infile in infile_list:
        theta,enc_int,enc_int_norm = np.loadtxt(os.path.join(datadir,infile),unpack=True,usecols=[0,1,2])
        
        ### find angle containing fraction 'frac' of enclosed intensity
        theta_frac = theta[enc_int_norm <= frac].max()
        theta_frac_list.append(theta_frac)

    ### plot theta vs. power-law exponent
    pl.plot(plaw_exp, theta_frac_list, lw=2.0)
    
### add legend
legend = pl.legend(([r"$%i$"%j for j in layers]),frameon=False,loc='lower left',handlelength=2,title="$\mathrm{lens}$ \n $\mathrm{layers}$")
pl.setp(legend.get_title(),fontsize='xx-small')

### minor ticks
yminticks = MultipleLocator(0.5)
pl.gca().yaxis.set_minor_locator(yminticks)

pl.xlim([min(plaw_exp),max(plaw_exp)])
pl.ylim([0.,max(theta_frac_list)])

pct = frac*100.
pl.xlabel(r'$\mathrm{Power}$-$\mathrm{law \ Exponent}\ i\ \mathrm{in}\  n = (2-r^2)^{i}$')
pl.ylabel(r'$\theta_{'+str(pct)+'\%} \mathrm{ \ Enclosed \ Intensity \ [deg]}$')

print ""
print ">>> Saving enclosed intensity plot to "+os.path.join(savedir,figoutfile)
print ""
pl.savefig(os.path.join(savedir,figoutfile))
pl.show()
pl.close()
