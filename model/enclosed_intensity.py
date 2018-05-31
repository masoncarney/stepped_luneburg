import os, sys
import numpy as np
import pylab as pl
from lens_setup import orion
from matplotlib.ticker import MultipleLocator

###################################################################
### Update matplotlib rc params

params = {'axes.labelsize'          : 20,
		'font.size'         : 20,
		'legend.fontsize'   : 10,
		'xtick.labelsize'   : 16,
		'ytick.labelsize'   : 16,
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

def enc_int(infile=None,outfile="luneburg_enc_int.dat",frac=0.5,mode=None,star_num=None,plot=True,figoutfile="luneburg_enc_int.pdf"):
    '''Calculate the cumulative enclosed intensity as a function of angle away from the Luneburg lens focal point. 

    Default parameters: 
    enc_int(infile=None,outfile="luneburg_enc_int.dat",frac=0.5, mode=None, star_num=None,plot=True,figoutfile="luneburg_enc_int.pdf"):
    
    This routine uses the output data file from luneburg_lens.py to calculate the enclosed intensity of rays exiting the bottom hemisphere of the lens as a function of the angle away from the expected wavefront focal point. The data is saved to an output file that records the angle away from the focal point, the total enclosed intensity at each angle, and the normalized enclosed intensity at each angle. The data can also be plotted with normalized cumulative intensity as a function of log of the angle from the focal point.
    
    Returns three arrays for theta, enclosed intensity, and normalized enclosed intensity
    
    ==========
    Parameters
    
    infile: [string] name of the input file for ray position and intensity data (default = None)

    outfile: [string] name of the output file for angle and enclosed intensity data (default = "luneburg_enc_int.dat")
    
    frac: [float] fraction of total enclosed intensity to mark for plotting; measure of lens performance (default = 0.5)
    
    mode: [string] initial pattern of the wavefront incident on the upper hemisphere of the Luneburg lens (default = None)
        options are 'stars', 'random', 'grid', or 'line'
        'stars' - star pattern on the sky (default = Orion) that can be manually updated/expanded by the user
        'random' - random distribution of points in the x-z plane within a given radius
        'grid' - uniform grid of points in the x-z plane within a given radius, nrays should be a square number
        'line' - single line of points along the x axis
            *note: if mode='stars' and star_num = ##: intensity output from the entire star pattern is read in from infile, so other stars in pattern will contaminate enclosed intensity function for star ## in star pattern

    star_num: [int] integer corresponding to the desired star from a user-provided star pattern; only required if mode='stars' (default = None)
    
    plot: [boolean True/False] option to show plot of normalized enclosed intensity as a function of log(angle) and save the figure (default = True)
    
    figoutfile: [string] name of the output figure file if plot = True (default = "luneburg_enc_int.pdf")

    ==========    
    Usage
     
    # import the enc_int function
    >> from enclosed_intensity import enc_int

    # calculate enclosed intensity from myfile.dat for 'grid' pattern, plot results
    >> enc_int(infile="myfile.dat",mode='grid'):
   
    # calculate enclosed intensity from myfile.dat for a specific star from the user-provided 'stars' pattern
    >> enc_int(infile="myfile.dat",mode='stars',star_num=2):
   
    # calculate enclosed intensity from myfile.dat for a 'line' pattern, write output to myfile_out.dat, and plot results to myplot.pdf
    >> enc_int(infile="myfile.dat",outfile="myfile_out.dat",mode='line',figoutfile='myplot.pdf'):
 
    # calculate enclosed intensity from myfile.dat for a 'grid' pattern with a high fraction of total intensity marked for plotting
    >> enc_int(infile="myfile.dat",mode='grid',frac=0.9):
 
    # calculate enclosed intensity and call theta, enclosed intensity, and normalized enclosed intensity from function return
    >> theta, enclosed_int, enclosed_int_norm = enc_int(infile="myfile.dat",mode='grid'):

    ==========    
    '''
    
    #####################################
    #         READ INPUT FILE           #
    #####################################

    ### import data
    try:
        xpos,ypos,zpos,xdirct,ydirct,zdirct,intensity = np.loadtxt(infile,skiprows=1,usecols=[0,1,2,3,4,5,6],unpack=True)
    except:
        raise IOError("Please provide a valid input file.")
        
    #####################################
    #   CALCULATE ENCLOSED INTENSITY    #
    #####################################

    ### if mode=='stars', correct for angle of star away from lens zenith at [0,1,0]

    if(mode=='stars' and star_num is not None):
        stars = orion(2)
        corr_ang = np.zeros(len(stars))
        for i in range(len(stars)):
            corr_ang[i] = 180./np.pi*np.arctan(np.sqrt(stars[i][0]**2+stars[i][2]**2)/stars[i][1])
        theta = abs(180.-(180./np.pi*np.arccos(ypos))-corr_ang[star_num])
        theta_rad = theta*np.pi/180.
    elif(mode=='stars' and star_num is None):
        raise ValueError("You must specify a star number when using mode='stars'")
    else:
        theta = abs(180.-(180./np.pi*np.arccos(ypos)))
        theta_rad = theta*np.pi/180.
    
    ### sort theta values in ascending order and the corresponding intensity

    theta_int = zip(theta,intensity)
    theta_int_sort = sorted(theta_int)
    
    theta_sorted = np.array([i[0] for i in theta_int_sort])
    intensity_sorted = np.array([i[1] for i in theta_int_sort])
    
    ### create theta bins for enclosed intensity calculation
    
    theta_bins = np.arange(0.,90.+1.e-3,0.1)
    enc_int = np.zeros(theta_bins.shape)

    ### store running total of enclosed intensity for each theta bin
    
    for i in range(len(theta_bins)):
        int_bin=0.
        if(i==0):
            if( theta_sorted[i] == theta_bins[i] ):
                int_bin+=intensity_sorted[i]
                enc_int[i]=int_bin
        elif(theta_bins[i]==theta_bins.max()):
                enc_int[i]=enc_int[i-1]
        else:
            for j in range(len(theta_sorted)):
                if( (theta_sorted[j] > theta_bins[i]) and (theta_sorted[j] < theta_bins[i+1]) ):
                    int_bin+=intensity_sorted[j]
            enc_int[i]=int_bin+enc_int[i-1]

    ### normalize enclosed intensity
    
    enc_int_norm = enc_int/enc_int.max()
    
    ### find theta bin and intensity value corresponding to "frac" of enclosed intensity
    
    frac_enc_int = theta_bins[enc_int_norm <= frac].max()
    
    #####################################
    #           WRITE OUTPUT            #
    #####################################

    ### write out theta, enclosed intensity, and normalized enclosed intensity 

    print ""
    print ">>> Writing results for theta, enclosed intensity, and normalized enclosed intensity to "+outfile
    print ""
    
    data_out = np.column_stack((theta_bins,enc_int,enc_int_norm))
    data_out_header = "Theta (deg), Enclosed Intensity, Normalized Enclosed Intensity"
    np.savetxt(outfile,data_out,header=data_out_header,fmt="%.8f")

    #####################################
    #      PLOT ENCLOSED INTENSITY      #
    #####################################
    
    if(plot==True):
        
        fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
        pl.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)

        pl.plot(np.log10(theta_bins), enc_int_norm, lw=3.0, color='k')
        pl.axvline(np.log10(frac_enc_int),color='k',linestyle='--',lw=1.5)

        ### minor ticks
        xminticks = MultipleLocator(0.1)
        yminticks = MultipleLocator(0.05)
        pl.gca().xaxis.set_minor_locator(xminticks)
        pl.gca().yaxis.set_minor_locator(yminticks)
        
        pl.xlim([-2.,2.])

        pl.xlabel(r'$\mathrm{log}(\theta) \ [\mathrm{deg}]$')
        pl.ylabel(r'$\mathrm{Fraction \ of \ Enclosed \ Intensity}$')
        
        print ">>> Saving enclosed intensity plot as "+figoutfile
        print ""
        pl.savefig(figoutfile)
        pl.show()

    return (theta_bins, enc_int, enc_int_norm)

########################################################################
