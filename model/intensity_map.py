import os, sys
import numpy as np
import pylab as pl
from matplotlib import colors
from matplotlib.ticker import MultipleLocator

###################################################################
### Update matplotlib rc params

params = {'axes.labelsize'          : 24,
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

def int_map(infile=None,outfile="luneburg_int_map.dat",pixels=500.,frac=0.5,sky_mag=13.,figoutfile="luneburg_int_map.pdf",mag_hist_plot=True,verbose=False):
    '''Create a 2D bullseye-style intensity map of haloes produced by the Luneburg lens.

    Default parameters:
    int_map(infile=None,outfile="luneburg_int_map.dat",pixels=500.,frac=0.5,sky_mag=13.,figoutfile="luneburg_int_map.pdf",mag_hist_plot=True,verbose=False)
    
    This routine uses the output data file from luneburg_lens.py to plot a 2D intensity map with a central region containing a user-supplied fraction of the total output intensity. The image is then normalized such that the central region is set equal to 1, and the remaining output intensity is recorded in concentric rings. The number of rings depends on the 'frac' parameter, as the bin size for the rings is set equal to the radius containing fraction 'frac' of the total output intensity. The final intensity map is plotted in log scale and tick labels are set to show the maximum radius of the pixel map normalized to 1.
    
    The aim of this routine is to show the strength of haloes produced by the Luneburg lens relative to the user-supplied fraction of the total output intensity. The user-supplied fraction 'frac' should reflect what the user considers to be sufficient lens performance for imaging. The intensity map is plotted in log scale for easy comparison to stellar magnitudes, with a color bar that displays the magnitude of the haloes as magnitude = -2.5*log(intensity). Haloes that have relative magnitudes greater than (i.e. dimmer) the sky brightness background magnitude are washed out, as they will not be detectable over the sky background noise transmitted through the lens.
    
    Example: The user-supplied fraction is 0.5. The user considers capturing 50% of the total output intensity from the Luneburg lens ray tracing routine to be sufficient lens performance for imaging. The central region of the intensity map will contain 50% of the total output intensity and the image is normalized so that the central region is equal to 1. The remainder of intensity is stored in concentric rings that correspond to lens haloes, each with a width equal to the radius of the central region containing 50% of the output intensity. In log scale, if any of the haloes have a relative magnitude greater than (i.e. dimmer) the sky brightness background magnitude (ring_mag > 'sky_mag') then they are set equal to 'sky_mag' as they will not be visible above the sky background. Haloes that are brighter than the sky background (ring_mag < 'sky_mag') will remain visible and potentially contaiminate the image, depending on their magnitude.
    
    Returns float values for radius, angle, and intensity contained within central region of intensity map.
    
    Optionally, this routine can plot a magnitude vs. radius histogram with the mag_hist_plot option. The plot uses the data from the intensity map to show the halo positions and halo strengths relative to the normalized central region containing 'frac' of the total intensity output.
    
    ==========
    Parameters
    
    infile: [string] name of the input file for ray position and intensity data (default = None)

    outfile: [string] name of the output file for 2D map Luneburg lens intensity output in ascii format (default = "luneburg_int_map.dat")
    
    pixels: [float] number of pixels per edge for intensity map of Luneburg lens output (default = 500)
    
    frac: [float] fraction of total Luneburg lens output intensity to be normalized to one; a lens performance threshold (default = 0.5)
    
    sky_mag: [float] sky brightness background magnitude (default = 13)
    
    figoutfile: [string] name of the output figure file (default = "luneburg_int_map.pdf")
        
    mag_hist_plot: [boolean True/False] option to create a magnitude vs radius histogram plot from the intensity map, output file name is figoutfile"_hist".extension
    
    verbose: [boolean True/False] option to print out more information to the terminal (default = False)
    
    ==========
    Usage 
    
    # import the enc_int function
    >> from intensity_map import int_map

    # create intensity map from myfile.dat and plot magnitude vs. radius histogram
    >> int_map(infile="myfile.dat"):
   
    # create intensity map from myfile.dat and save to myplot.pdf, but no histogram plot 
    >> int_map(infile="myfile.dat",figoutfile="myplot.pdf",mag_hist_plot=False):
 
    # create intensity map from myfile.dat with 1000 pixels
    >> int_map(infile="myfile.dat",pixels=1000):

    # create intensity map from myfile.dat with 1000 pixels, with central region containing 80% of Luneburg lens output intensity
    >> int_map(infile="myfile.dat",pixels=1000,frac=0.8):
 
    # create intensity map from myfile.dat and call radius, angle, and intensity of central region containing 'frac' of intensity output from function returns
    >> frac_radius, frac_theta, frac_int = int_map(infile="myfile.dat"):

    ==========
    '''

    #####################################
    #         READ INPUT FILE           #
    #####################################

    ### import data
    
    try:
        xpos,ypos,zpos,intensity = np.loadtxt(infile,skiprows=1,usecols=[0,1,2,6],unpack=True)
    except:
        raise IOError("Please provide a valid input file.")
    

    #####################################
    #      CREATE INTENSITY MAP         #
    #####################################

    if(verbose==True):
        print ""
        print "Creating a %gx%g pixel intensity map."%(pixels,pixels)
        print "Central region will contain approximately %.2f percent of total intensity output."%(frac*100.)
        print "Rings with magnitude >%.1f are washed out due to the sky brightness background"%sky_mag

    ### zip and sort radius and intensity values from data file

    r_data = np.sqrt(xpos**2 + zpos**2)
    r_lens = 0.5*(pixels)

    r_int_data = zip(r_data,intensity)
    r_int_data_sort = sorted(r_int_data)

    r_ypos_data = zip(r_data,ypos)
    r_ypos_data_sort = sorted(r_ypos_data)

    ypos_sorted = np.array([i[1] for i in r_ypos_data_sort])
    r_data_sorted = np.array([r_lens*i[0] for i in r_int_data_sort])
    intensity_sorted = np.array([i[1] for i in r_int_data_sort])
    
    ### create square arrays for x and y values based on value_range

    value_range = np.arange(pixels)
    y_vals = value_range[:,np.newaxis] + np.zeros_like(value_range)
    x_vals = value_range + np.zeros_like(value_range)[:,np.newaxis]

    ### create a new 2D array for radial values centered at (0.5*pixels,0.5*pixels)

    r_map = np.sqrt((x_vals - 0.5*(pixels))**2 + (y_vals - 0.5*(pixels))**2)

    ### calculate value for fraction 'frac' of total intensity

    total_int = intensity_sorted.sum()
    frac_int = frac*total_int

    ### calculate the cumulative intensity and index of 'frac' of cumulative intensity

    cum_intensity = np.zeros(intensity_sorted.shape)
    for i in range(intensity_sorted.shape[0]):
        if(i == 0):
            cum_intensity[i]=intensity_sorted[i]
        else:
            cum_intensity[i]=intensity_sorted[i]+cum_intensity[i-1]

    indices_frac_int = np.where(cum_intensity < frac_int)

    ### find radius and angle that marks fraction 'frac' of cumulative intensity

    r_frac_int = r_data_sorted[max(indices_frac_int[0])]
    ypos_frac_int = ypos_sorted[max(indices_frac_int[0])]
    theta_frac_int = abs(180.-(180./np.pi*np.arccos(ypos_frac_int))) # theta is measured from bottom of lens
    
    ### normalize intensity map such that the intensity area containing fraction 'frac' is the set to 1

    int_normalize = frac_int/(np.pi*r_frac_int**2)
    int_limit = 10.**(-sky_mag/2.5)
    
    ### mask out intensity map for points outside of max lens radius
    
    r_map = np.ma.masked_greater(r_map,r_lens)

    ### innermost circle of bullseye will contain fraction 'frac' of total intensity and be normalized to equal zero

    r_map[np.where(r_map <= r_frac_int)] = np.log10(frac_int/(np.pi*r_frac_int**2)/int_normalize)
    
    ### calculate halo ring radii with bins equal to the radius the central region containing fraction 'frac' of total output intensity

    r_bin = r_frac_int
    r_ring = np.arange(0.,r_lens,r_bin) # array of ring radii

    # add r_lens+increase as last element of ring radii array to ensure rings cover all pixels out to r_lens
    
    r_max = r_lens+0.001
    r_ring = np.concatenate((r_ring,np.array([r_max])))

    ### remove radii and intensity indices used in central region from sorted lists

    r_data_sorted = r_data_sorted[max(indices_frac_int[0])+1:]
    center_intensity_sorted = intensity_sorted[indices_frac_int]
    intensity_sorted = intensity_sorted[max(indices_frac_int[0])+1:]
    
    ### calculate the normalized intensity contained within each remaining ring and correct for ring area
    ### ring intensity values are stored as log(area-corrected, normalized intensity) in r_map

    list_ring_int = []
    for i in range(len(r_ring)):
        if(i == 0):
            pass
        elif(i == 1):
            ring_area = np.pi*r_ring[i]**2
            ring_intensity = sum(center_intensity_sorted)
            central_intensity = ring_intensity
            if(verbose==True):
                ring_mag = -2.5*np.log10((ring_intensity/ring_area)/int_normalize)
                print ""
                print "Central region contains %.2f percent of total output intensity: %.4f"%(ring_intensity/total_int*100.,ring_intensity)
                print "Central region width: r = 0.00 to r = %0.2f pixels"%r_ring[i]
                print "Central region magnitude : %.2f"%(ring_mag)
        else:
            ring_intensity = sum(intensity_sorted[np.where((r_data_sorted < r_ring[i]) & (r_data_sorted >= r_ring[i-1]))])
            ring_area = np.pi*((r_ring[i])**2-(r_ring[i-1])**2)
            if(ring_intensity == 0.):
                list_ring_int.append(0.)
                r_map[np.where((r_map < r_ring[i]) & (r_map >= r_ring[i-1]))] = np.log10(int_limit)
            elif(np.log10((ring_intensity/ring_area)/int_normalize) > np.log10(int_limit)):
                list_ring_int.append((ring_intensity/ring_area)/int_normalize)
                r_map[np.where((r_map < r_ring[i]) & (r_map >= r_ring[i-1]))] = np.log10((ring_intensity/ring_area)/int_normalize)
            else:
                r_map[np.where((r_map < r_ring[i]) & (r_map >= r_ring[i-1]))] = np.log10(int_limit)
                list_ring_int.append((ring_intensity/ring_area)/int_normalize) 
            if(verbose==True):
                print ""
                print "Ring %i contains %.2f percent of total output intensity: %.4f"%(i-1,ring_intensity/total_int*100.,ring_intensity)
                print "Ring width     : r = %0.2f to r = %0.2f pixels"%(r_ring[i-1],r_ring[i])
                if(ring_intensity == 0.):
                    print "Ring magnitude : N/A, set to sky background magnitude"
                else:
                    ring_mag = -2.5*np.log10((ring_intensity/ring_area)/int_normalize)
                    print "Ring magnitude : %.2f"%(ring_mag)


    #####################################
    #      WRITE INFO TO TERMINAL       #
    #####################################

    ### print output to terminal
    
    print ""
    print "Total intensity output                           : %.2f"%total_int
    print ""
    print "Radius containing "+str(frac*100.)+" percent intensity         : %.2f pixels"%r_frac_int
    print "Angle containing "+str(frac*100.)+" percent intensity          : %.2f degrees"%theta_frac_int
    print "Lens resolving power at "+str(frac*100.)+" intensity threshold : %0.2f degrees"%(2.*theta_frac_int)
    print ""
    
    #####################################
    #           WRITE OUTPUT            #
    #####################################

    ### write out pixel coordinates and intensity to outfile

    print ""
    print ">>> Writing results for each pixel x position, z position, intensity, log(intensity), and magnitude to "+outfile
    print ""
    
    i_x = np.zeros(r_map.shape)
    i_z = np.zeros(r_map.shape)
    data_out = np.zeros(r_map.shape)
    data_out_log = np.zeros(r_map.shape)
    
    for i in np.arange(r_map.shape[0]):
        for j in np.arange(r_map.shape[1]):
            i_x[i,j] = i
            i_z[i,j] = j
            data_out[i,j] = 10.**r_map[i,j]
            data_out_log[i,j] = r_map[i,j]
            
    data_out_mag = -2.5*data_out_log
    data_out_stack = np.column_stack((i_x.ravel(),i_z.ravel(),data_out.ravel(),data_out_log.ravel(),data_out_mag.ravel()))
    data_out_header = "x (pixels), z (pixels), Intensity, log(Intensity), Magnitude"

    np.savetxt(outfile,data_out_stack,header=data_out_header,fmt="%.8f")

    #####################################
    #        PLOT INTENSITY MAP         #
    #####################################

    
    fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
    pl.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)
    
    ### plot the pixelated intensity map as log(area-corrected, normalized int)
    pl.imshow(r_map, cmap="Spectral_r", interpolation='nearest',vmin=(15/-2.5),vmax=0)
    pl.gca().set_ylim(pl.gca().get_ylim()[::-1]) # reverse y axis
    
    ### convert color bar ticks from log(int) scale to magnitude | int=10**(-mag/2.5)
    cbarticks = np.arange(1,15+1,2)/(-2.5)
    cbartick_lbl = [r'$%g$'%(-2.5*i) for i in cbarticks]
    cbar = pl.colorbar(ticks=cbarticks)
    cbar.set_ticklabels(cbartick_lbl)
    cbar.set_label(r"$\mathrm{Magnitude}$")
    
    ### set tick labels for normalized axes coordinates relative to center
    xticks = np.arange(0.,pixels+1,r_lens/2.)
    yticks = np.arange(0.,pixels+1,r_lens/2.)
    pl.gca().set_xticks(xticks)
    pl.gca().set_yticks(yticks)
    # labels are normalize axes coordinates relative to center
    pl.gca().set_xticklabels([r"$%.1f$"%((i-r_lens)/r_lens) for i in xticks])
    pl.gca().set_yticklabels([r"$%.1f$"%((i-r_lens)/r_lens) for i in yticks])
    
    ### minor ticks
    xminticks = yminticks = MultipleLocator(r_lens/10.)
    pl.gca().xaxis.set_minor_locator(xminticks)
    pl.gca().yaxis.set_minor_locator(yminticks)

    pl.gca().set_xlim(-(0+r_lens/10.),pixels+r_lens/10.)
    pl.gca().set_ylim(-(0+r_lens/10.),pixels+r_lens/10.)
   
    pl.xlabel(r'${\rm X}$')
    pl.ylabel(r'${\rm Z}$')
    
    print ""
    print ">>> Saving intensity map plot as "+figoutfile
    print ""

    pl.savefig(figoutfile)
    
    pl.show()
    pl.close()
    

##############################################################################

    #####################################
    #    PLOT MAGNITUDE HISTORGRAM      #
    #####################################

    if(mag_hist_plot == True):
        
        fig = pl.figure(figsize=(6.5,5), dpi=100, facecolor='w', edgecolor='k')
        pl.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0., hspace=0.)

        ### define ring radii for histogram plot
        # first point will be zero, remove last point for histogram binning consistency
        r_ring_hist = list(r_ring[:])
        r_histbins = r_ring[:]
        r_ring_hist.remove(max(r_ring_hist))
        
        ### define ring intensity for histogram plot
        # set first point to 1, corresponds to central region containing fraction 'frac' of total intensity
        # remainder of intensity resides in haloes and is plotted relative to the central region
        ring_int_hist = list_ring_int[:]
        ring_int_hist.insert(0,1)

        ### plot histogram of halo intensity vs. radius 
        pl.hist(r_ring_hist,r_histbins,weights=ring_int_hist,log=True,color='grey')
        
        ### convert y-axis ticks to magnitude from log(int) scale | int=10**(-mag/2.5)
        ylim_mag = 16 # integer | magnitude for yaxis lower limit
        yticks = [10**(-i/2.5) for i in range(ylim_mag)]
        # i%2!=0 for even magnitude y tick labels, i%2==0 for odd magnitude y tick labels
        ytick_lbl = [' ' if i%2!=0 else r'$%g$'%i for i in range(ylim_mag)]
        pl.gca().set_yticks(yticks)
        pl.gca().set_yticklabels(ytick_lbl)

        ### set x tick labels for normalized axes coordinates relative to center
        xticks = np.arange(0.,r_lens+1,r_lens/10.)
        pl.gca().set_xticks(xticks)
        # labels are normalize axes coordinates relative to center
        pl.gca().set_xticklabels([r"$%.1f$"%(i/r_lens) for i in xticks])
        
        ### minor ticks
        xminticks = MultipleLocator(r_lens/100.)
        pl.gca().xaxis.set_minor_locator(xminticks)        
        
        pl.ylim(10**(-ylim_mag/2.5),1.)
        pl.xlim(0.,r_lens)
        pl.xlabel(r'$r$')
        pl.ylabel(r'$\mathrm{Magnitude}$')
        
        ### mark sky brightness background magnitude on plot
        pl.gca().axhline(10**(-sky_mag/2.5),color='k',linestyle='--')        
        
        #figoutfile_hist = figoutfile.split('.')[0]+'_hist.'+figoutfile.split('.')[1]
        
        figoutfile_dir  = figoutfile.rsplit('/')[0]
        figoutfile_root = figoutfile.rsplit('/')[1]
        figoutfile_name = figoutfile_root.rsplit('.')[0]+'_hist.'+figoutfile_root.rsplit('.')[1]
        figoutfile_hist = os.path.join(figoutfile_dir,figoutfile_name)
        
        print ">>> Saving magnitude histogram plot as "+figoutfile_hist
        print ""
        pl.savefig(figoutfile_hist)

        pl.show()
        pl.close()

    return (r_frac_int,theta_frac_int,central_intensity)
        
########################################################################
