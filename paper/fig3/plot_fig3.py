import os, sys
import numpy as np
import pylab as pl
from matplotlib.ticker import MultipleLocator

'''
Script to plot enclosed intensity vs. theta exponent for multiple Luneburg lens runs.
Theta corresponds to the angle away from the initial wavefront focal point.
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

layers     = 40 # must be integers
plaw_exp   = [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
nrays      = 1024.

### define output figure file name
figoutfile = "fig3.pdf"

###################################################################

### define file list for plotting
infile_list = [('luneburg_enc_int_%i_%.2f_%i_grid.dat'%(layers,exp,nrays)).replace('_0.','_0d') for exp in plaw_exp]

### create figure
fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
pl.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)

### define color cycle for plotting
pl.gca().set_color_cycle(['b','g','r','c','y','m','k','#FF8000','brown'])

### loop through file list and add to plot
for infile in infile_list:
    theta,enc_int,enc_int_norm = np.loadtxt(os.path.join(datadir,infile),unpack=True,usecols=[0,1,2])
    
    ### plot enclosed intensity vs. angle
    pl.plot(np.log10(theta), enc_int_norm, lw=3.0)

### add legend
legend = pl.legend(([r"$%.2f$"%j for j in plaw_exp]),frameon=False,loc='upper left',handlelength=1.5,title="$\mathrm{power}$-$\mathrm{law}$ \n $\mathrm{exponent}$")
pl.setp(legend.get_title(),fontsize='xx-small')

### add line marking enclosed intensity at theta = 1 deg 
pl.axvline(np.log10(1.),color='k',linestyle='--',lw=1.5)

### minor ticks
xminticks = MultipleLocator(0.1)
yminticks = MultipleLocator(0.05)
pl.gca().xaxis.set_minor_locator(xminticks)
pl.gca().yaxis.set_minor_locator(yminticks)

pl.xlim([-2.,2.])

pl.xlabel(r'$\mathrm{log}(\theta) \ [\mathrm{deg}]$')
pl.ylabel(r'$\mathrm{Normalized \ Enclosed \ Intensity}$')
#pl.title(r'$\mathrm{%i \ Layer \ Luneburg \ Lens}$'%layers)

print ""
print ">>> Saving enclosed intensity plot to "+os.path.join(savedir,figoutfile)
print ""
pl.savefig(os.path.join(savedir,figoutfile))
pl.show()
pl.close()
