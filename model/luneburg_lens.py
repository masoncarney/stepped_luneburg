import os, sys
import numpy as np
import pylab as pl
import time
from matplotlib.ticker import MultipleLocator

from lens_plotting import draw_ray,draw_sph,draw_sph_2Dproj
from lens_ray_tracing import snell_vec,sphere_hit
from lens_setup import n_luneburg,rotate,orion,rand_grid,uniform_grid

###################################################################
### Update matplotlib rc params

params = {'axes.labelsize'          : 24,
                'font.size'         : 28,
		'legend.fontsize'   : 10,
		'xtick.labelsize'   : 14,
		'ytick.labelsize'   : 14,
		'xtick.major.size'  : 6,
                'ytick.major.size'  : 6,
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

def stepped_luneburg(outfile="luneburg_output.dat",steps=20.,exp=0.5,nrays=100.,amp_cut=0.01,center=[0.,0.,0.],mode=None,modify=False,plot=True,figoutfile="luneburg_raytrace.pdf",verbose=False,logfile="luneburg_log.dat"):
    """Create a stepped Luneburg lens model: a spherical lens of varying index of refraction

    Default parameters:
    stepped_luneburg(outfile="luneburg_output.dat",steps=20.,exp=0.5,nrays=100.,amp_cut=0.01,center=[0.,0.,0.],mode=None,modify=False,plot=True,figoutfile="luneburg_lens_raytrace.pdf",verbose=False,logfile="luneburg_log.dat")
    
    This routine traces the path of a wavefront of rays through the Luneburg lens using Snell's Law at each lens layer. The radius of the lens surface is normalized to 1. Normalized radii for each lens layer are calculated based on the number of steps provided. The lens index of refraction is calculated as n = (2-r^2)^exp and increases in discrete steps from the lens surface to the lens center with the maximum refractive index at center. 
    
    The mode for the wavefront of rays is initialized such that all incoming rays are parallel (i.e. incoming from infinity) and each ray has an amplitude of 1. Rays enter the lens if they are incident on the top hemisphere. Rays incident on the bottom hemisphere of the lens are discarded. All rays that enter the lens are propagated until the ray exits the lens or falls below a designated amplitude threshold. 
    
    The position, direction, and intensity for rays that exit the bottom hemisphere of the lens are stored and accumulated so that an image can be simulated on the bottom surface of the lens. These ray parameters are written to an output file for later processing.
 
    A four-panel figure can be created to visualize ray tracing through the Luneburg lens. 
      Plot a) shows the lens layers and the position of the incident wavefront in the x-y plane (side view). 
      Plot b) shows the lens layers and the position of the incident wavefront in the x-z plane (top view). 
      Plot c) shows the lens layers and each ray propagating through each lens layer in the x-y plane (side view). Rays striking the outermost surface of the lens are shown with a large orange dot. Rays striking any lens layer interface are shown with a small red dot. Rays exiting the lens are shown with a yellow dot. Rays marked with a black 'X' have been dropped from the routine due to the amplitude cutoff or due to exiting the top hemisphere of the lens.
      Plot d) shows a 2D projection of the rays exiting the bottom hemisphere of the lens in order to simulate an image produced by the lens. Exiting rays are shown as small blue dots, and an alpha value is applied to each dot that corresponds to intensity (i.e. exiting rays with less intensity are more transparent).
        ***note: if mode='stars' then only the stars in the star pattern are plotted in Plots a), b), c), and an unplotted 'grid' pattern of rays is rotated to the stellar position and used to initialize the wavefront of incoming rays from each star

    Optionally, this routine can calculate the lens layer indices of refraction with a modified refractive index power law equation where n = (2-r^(2*exp)) for performance comparison to the original refractive index power law equation.
    
    Optionally, this routine can write all terminal output to a log file if verbose = True.

    ==========
    Parameters
    
    outfile: [string] name of the output file (default = "luneburg_output.dat")
    
    steps: [float] number of steps/layers for the Luneburg lens model (default = 20)
    
    exp: [float] power-law exponent for the varying index of refraction of the stepped lens model (default = 0.5)
            
    nrays: [float] numbers of rays to initialize for the wavefront incident on the Luneburg lens; should be a square number (default = 100.)

    amp_cut: [float] amplitude cutoff value for rays propagating through the lens: rays with amplitude below this value after Snell's law are dropped by the program (default = 0.01, 1% amplitude of ray prior to Snell's law)

    center: [list of floats] x, y, and z coordinates of the lens center (default = [0.,0.,0.])
        *** CAUTION: non-zero centers are not properly implemented yet ***

    mode: [string] initial pattern of the wavefront incident on the upper hemisphere of the Luneburg lens (default = None)
        options are 'stars', 'random', 'grid', or 'line'
        'stars' - star pattern on the sky (default = Orion) that can be manually updated/expanded by the user
        'random' - random distribution of points in the x-z plane within a given radius
        'grid' - uniform grid of points in the x-z plane within a given radius, nrays should be a square number
        'line' - single line of points along the x axis

    modify: [boolean True/False] option to modify power law used to calculate the refractive index: True for modified power law, False for original power law (default = False)
        original: n = (2-r^2)^exp
        modified: n = (2-r^(2*exp))

    plot: [boolean True/False] option to show plot of ray tracing through the lens and save the figure (default = True)
        *note: False recommended if nrays is very large
    
    figoutfile: [string] name of the output figure file if plot = True (default = "luneburg_raytrace.pdf")
    
    verbose: [boolean True/False] option to print out more information to the terminal and save to a log file (default = False)
    
    logfile: [string] name of log file containing terminal output if verbose=True (default="luneburg_log.dat")
    
    ========== 
    Usage
    
    # import the stepped_luneburg function
    >> from luneburg_lens import stepped_luneburg

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays
    >> stepped_luneburg(mode='grid'):

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays with a non-zero center 
    *** CAUTION: non-zero centers are not properly implemented yet ***
    >> stepped_luneburg(mode='grid',center=[0.1,0.1,0.1]):

    # run a Luneburg model for 'grid' wavefront pattern of incoming rays without producing a plot
    >> stepped_luneburg(mode='grid',plot=False):

    # run a Luneburg model for 'line' wavefront pattern of incoming rays for a lens with 40 layers, 1000 rays, and amplitude cutoff at 10% of the ray amplitude prior to Snell's law
    >> stepped_luneburg(mode='line',steps=40.,nrays=1000.,amp_cut=0.1):
        
    # run a Luneburg model for 'random' wavefront pattern of incoming rays with the modified refractive index power law equation and power law exponent 0.6 
    >> stepped_luneburg(mode='random',exp=0.6,modify=True):
   
    # run a Luneburg model for 'stars' wavefront pattern of incoming rays, save to myfile_out.dat, save figure to myplot.pdf, and save log file mylog.dat
    >> stepped_luneburg(outfile="myfile_out.dat",mode='stars',figoutfile="myplot.pdf",verbose=True,logfile="mylog.dat"):
 
    ==========    
    """
    
    print ""
    print ">>> Running Luneburg lens model in directory "+os.getcwd()
    print ""
    
    #####################################
    #      INITIALIZE LENS SETUP        #
    #####################################

    if(verbose==True):
        
        # define logger class to write to terminal and log file
        class logger :
            def __init__(self, *logs) :
                self.logs = logs

            def write(self, text) :
                for l in self.logs :
                    l.write(text)

        log = file(logfile, 'w')
        sys.stdout = logger(sys.stdout, log)
        
        print ""
        print "Initializing Luneburg lens parameters..."
        print "Number of steps     : %g"%steps
        print "Power-law exponent  : %.2f"%exp
        print "Lens center         : x=%.1f, y=%.1f, z=%.1f"%(center[0],center[1],center[2])
        t0 = time.time()
        
    ### take user-defined variables to initialize Luneburg lens setup
    c0       = np.array(center)                 #lens center
    r_step   = 1./steps                         #step radius normalized to unity
    r_sph    = np.arange(0.,1.+1.e-3,r_step)    #array of radii for each surface layer in the lens

    ### check option to use a modified version of the equation for refractive index
    ### 1 to use modified form of power law (2-r**(2*exp))
    ### 0 to use original form of power law (2-r**2)**exp
    if(modify==True):
        mod_eq = 1
        if(verbose==True):
            print ""
            print "Calculating indices of refraction with modified power law: n = (2-r^2)^exp"
    else:
        mod_eq = 0 
        if(verbose==True):
            print ""
            print "Calculating indices of refraction with original power law: n = (2-r^(2*exp))"
    
    ### calculate indices of refraction for each layer step 
    n_sph = n_luneburg(r_sph,exp,mod_eq)        

    ### normalize radius of lens to unity
    r_sph_norm = r_sph/max(r_sph)

    ### initialize figure to plot ray tracing
    if(plot==True):
        pl.figure(figsize=(12,10))
        pl.subplots_adjust(left=0.1,bottom=0.08,right=0.95,top=0.94,hspace=0.35,wspace=0.35)
        pl.subplot(223)

    #####################################
    #      INITIALIZE INPUT RAYS        #
    #####################################

    if(verbose==True):
        print ""
        print "Initializing wavefront of rays incident on the top hemisphere of the Luneburg lens: mode = "+mode
    
    ### storage lists for input rays and rays that exit the lens

    p_in        = []            # position of input ray
    d_in        = []            # direction of input ray
    a_in        = []            # amplitude of input ray
    #as_in       = []            # s-polarization amplitude of input ray
    #ap_in       = []            # p-polarization amplitude of input ray

    p_exit      = []            # position of exiting ray
    d_exit      = []            # direction of exiting ray
    a_exit      = []            # amplitude of exiting ray
    #as_exit     = []            # s-polarization amplitude of exiting ray
    #ap_exit     = []            # p-polarization of exiting ray

    ### choose type of input ray pattern

    ### 'stars'     = user-provided star pattern
    ### 'random'    = wavefront with random point distribution
    ### 'grid'      = wavefront with uniform grid point distribution
    ### 'line'      = line of points

    if(mode == 'stars'):

        ### designate star pattern to use
        
        stars = orion(2)       # creates Orion star pattern using orion.py
        star_plt = stars[:]

        plt_pts = []
        clr_cnt = 0
        clr = ['r','b','g','c','m','y','k']

        ang_xy,ang_xz = np.loadtxt('orion.dat',skiprows=1,usecols=[0,1],unpack=True)
        
        ang_xz = 90.-ang_xz # recalculate ang_xz as 90 deg - ang_xz; star pattern is offset from zenith instead of horizon for ray tracing
        ang_xy = ang_xy*np.pi/180. # convert ang_xy from deg to radians
        ang_xz = ang_xz*np.pi/180. # convert ang_xz from deg to radians
        ang_xy = list(ang_xy)
        ang_xz = list(ang_xz)

        ang_cnt = 0
        
        while(len(stars) > 0):

            star_pts = stars.pop(0)

            drct = np.array([c0[0]-star_pts[0],c0[1]-star_pts[1],c0[2]-star_pts[2]])

            init_pts_nonrot = []

            step = 2./(np.sqrt(nrays)-1)

            for i in np.arange(-1.,1.+1e-3,step):
                for j in np.arange(-1.,1.+1e-3,step):
                    pt = np.array([i,2.,j])
                    init_pts_nonrot.append(pt)
    
            init_pts = rotate(init_pts_nonrot,ang_xy[ang_cnt],ang_xz[ang_cnt])
            init_pts = rotate(init_pts,0,-np.pi/2.)
            init_pts = rotate(init_pts,-np.pi/2.,0)

            plt_pts.append(init_pts)

            for i in np.arange(0,len(init_pts)):
                p_in.append(np.array([init_pts[i][0],init_pts[i][1],init_pts[i][2]]))
                d_in.append(drct)
                a_in.append(1.0)
                #as_in.append(0.5)
                #ap_in.append(0.5)

            clr_cnt += 1
            ang_cnt += 1

    ### RANDOM WAVEFRONT OF POINTS

    elif(mode == 'random'):

        plt_pts = []

        init_pts = rand_grid(max(r_sph_norm),nrays,2.)
        plt_pts.append(init_pts)
        
        for i in np.arange(0,len(init_pts)):
            p_in.append(np.array([init_pts[i][0],init_pts[i][1],init_pts[i][2]]))

        for j in np.arange(0,len(p_in)):
            d_in.append(np.array([0.,-1.,0.]))
            a_in.append(1.0)
            #as_in.append(0.5)
            #ap_in.append(0.5)

    ### UNIFORM GRID WAVEFRONT OF POINTS

    elif(mode == 'grid'):

        init_pts = []
        plt_pts = []

        init_pts = uniform_grid(nrays,2.)
        plt_pts.append(init_pts)

        for i in np.arange(0,len(init_pts)):
            p_in.append(np.array([init_pts[i][0],init_pts[i][1],init_pts[i][2]]))

        for j in np.arange(0,len(p_in)):
            d_in.append(np.array([0.,-1.,0.]))
            a_in.append(1.0)
            #as_in.append(0.5)
            #ap_in.append(0.5)

    ### LINE OF POINTS

    elif(mode == 'line'):

        init_pts = []
        plt_pts = []

        nstep = 2./(nrays-1)
        for i in np.arange(-1.,1.+1.e-3,nstep):
            if(-1.e-5 < i < 1.e-5):
                pass
            else:
                init_pts.append(np.array([i,2.,0.]))

        plt_pts.append(init_pts)    

        for i in range(len(init_pts)):
            p_in.append(np.array([init_pts[i][0],init_pts[i][1],init_pts[i][2]]))

        for j in np.arange(0,len(p_in)):
            d_in.append(np.array([0.,-1.,0.]))
            a_in.append(1.0)
            #as_in.append(0.5)
            #ap_in.append(0.5)
    

    else:
        raise ValueError("Please choose a valid profile for the wavefront(s) incident on the lens: mode = 'stars', 'random', 'grid', or 'line'")
            

    #####################################
    #            RAY TRACING            #
    #####################################

    p_incoming         = []
    a_incoming         = []
    amp_dropped_cutoff = []
    amp_dropped_top    = []
    n_dropped_cutoff   = 0
    n_dropped_top      = 0
    n_exit_ray         = 0

    w = 1
    loop_exit = 1
    while(loop_exit != 0):
        # define storage arrays for the ray tracing stack
        # to be empty at the beginning of each loop
        p_out = []
        d_out = []
        a_out = []
        #as_out = []
        #ap_out = []

        # marker to exit loop when there are no more incoming rays
        if(len(p_in)==0):
            loop_exit=0

        while(len(p_in) > 0):
           # pop off each of the _in values one at a time
            p0 = p_in.pop(0)
            d0 = d_in.pop(0)
            a0 = a_in.pop(0)
            #as0 = as_in.pop(0)
            #ap0 = ap_in.pop(0)

            # now we have our first ray
            # take this ray and see if it hits any lens layers (sphere_hit)
            # store the output t from sphere_hit in an array

            t_nsph = np.zeros(len(r_sph_norm))

            i = 0
            for r in r_sph_norm:
                t_nsph[i] = sphere_hit(p0, d0, c0, r)
                i += 1
                
            # if zero hits, just jump back for the next ray    
            # if one hit, calculate t and the sphere you hit 

            # if all of the elements in t_nsph are -1, this ray [p0, d0] has hit
            # none of the spheres and we go to the next ray

            if(max(t_nsph) > 0.):
                    
                # if at least one element in t_nsph is positive, then we've hit a
                # sphere and we need to know which one.
                # we want the minimum positive value of t.
                # its location in t_nsph also tells us the two spheres (and two
                # refractive indicies) we need for snell's law

                t_nsph_index = np.arange(len(t_nsph))

                t_nsph_positive = t_nsph[t_nsph > 0.]

                t_nsph_positive_index = t_nsph_index[t_nsph > 0.]

                # t_min is the ray position that hits the closest sphere
                t_min = min(t_nsph_positive)

                sphere_number = t_nsph_positive_index[np.where(t_min == t_nsph_positive)]

                p1 = draw_ray(p0,d0,t_min,plot)

                # p1 is the point on the new sphere just hit    
                # and this still propagates in direction d0

                v_norm = p1 - c0

                # calculate the input n1 and n2 from the sphere number    
                # presumably this is n_sph(sphere_number) and n_sph(sphere_number+1)

                n1 = n_sph[sphere_number+1]
                n2 = n_sph[sphere_number]

                # make a test for v_norm and d0
                # if dotproduct is -ve, invert the v_norm vector

                if(np.dot(v_norm, d0) > 0.):
                    v_norm = -v_norm
                    tmp = n1
                    n1 = n2
                    n2 = tmp
                
                # use snell's law to calculate reflected and refracted rays

                new_ray = snell_vec(v_norm, d0, n1, n2)

                total_reflected_ampl = a0*new_ray[2]
                total_refracted_ampl = a0*(1.-new_ray[2])

                # drop rays below chosen amplitude
                ampl_drop = amp_cut

                # only include rays that hit the top hemisphere 
                # of outermost layer
                if((w == 1) and (p1[1] <= c0[1])):
                    # plot dropped rays with a black X
                    if(plot==True):
                        pl.plot(p1[0],p1[1],'kx',ms=12.0,mew=1.)                
                else:
                    if((w == 1) and (p1[1] > c0[1])):
                        p_incoming.append(p0)
                        a_incoming.append(a0)
                    
                    # store the reflected ray for next iteration (because it always works)
                    if(total_reflected_ampl > ampl_drop):
                    
                        p_out.append(p1)
                        d_out.append(new_ray[0])
                        a_out.append(a0*new_ray[2])
                        #as_out.append(as0*new_ray[3])
                        #ap_out.append(ap0*new_ray[4])

                    # if there is a refracted ray, store that too
                    if((new_ray[2] < 1.0) and (total_refracted_ampl > ampl_drop)):
                    
                        p_out.append(p1)
                        d_out.append(new_ray[1])
                        a_out.append(a0*(1.-new_ray[2]))
                        #as_out.append(as0*(1-new_ray[3]))
                        #ap_out.append(ap0*(1-new_ray[4]))
                
                    if(total_reflected_ampl < ampl_drop):
                        n_dropped_cutoff += 1
                        amp_dropped_cutoff.append(total_reflected_ampl)
                    if(total_refracted_ampl < ampl_drop):
                        n_dropped_cutoff += 1
                        amp_dropped_cutoff.append(total_refracted_ampl)
        
                    # end of while loop!
        
        if(verbose==True):
            print ""
            print "Total number of top-exiting rays dropped : ",n_dropped_top
            print "Total number of rays dropped due to low amplitude : ",n_dropped_cutoff
            print "END OF LOOP ",w
        w += 1
        
        # define _out rays as the new _in rays for next iteration of ray tracing stack
        p_in = p_out
        d_in = d_out
        a_in = a_out
        #as_in = as_out
        #ap_in = ap_out

        #positive p0.d0 (dot product) leaves the outermost sphere and never returns
        #test to see if p0 is close to largest sphere (within ~0.001 units)

        p_test = p_out[:]
        d_test = d_out[:]

        i = 0
        while(len(p_test) > 0 ):
            pt0 = p_test.pop(0)
            dt0 = d_test.pop(0)
            r_pt0 = np.sqrt((pt0[0]-c0[0])**2+(pt0[1]-c0[1])**2+(pt0[2]-c0[2])**2)
            
            # if p0 close to largest sphere, mark with orange spot
            if(np.abs(r_pt0-max(r_sph_norm)) <= 0.001):
                if(plot==True):
                    pl.plot(pt0[0],pt0[1],color='#CD8500',marker='o',ms=6.0)
                
                # use p0.d0 as test to determine if ray exits lens
                # positive dot product means that ray exits
                if(np.dot(pt0,dt0) > 0.):
                    if(plot==True):
                        pl.plot(pt0[0],pt0[1],marker='o',color='y',ms=4.0)

                    # only keep rays that exit bottom hemisphere
                    if(p_in[i][1] < c0[1]):
                        # store information about the exiting array
                        p_exit.append(p_in[i])
                        d_exit.append(d_in[i])
                        a_exit.append(a_in[i])
                        #as_exit.append(as_in[i])
                        #ap_exit.append(ap_in[i])
                        
                        # remove information about an exiting ray before the next iteration of ray tracing
                        p_in.pop(i)
                        d_in.pop(i)
                        a_in.pop(i)
                        #as_in.pop(i)
                        #ap_in.pop(i)
                        
                        i -= 1                        
                        n_exit_ray += 1
                        if(verbose==True):
                            print "Bottom-exiting ray ==> ray data saved."
                    else:
                        # plot dropped rays with a black X
                        if(plot==True):
                            pl.plot(p_in[i][0],p_in[i][1],'kx',ms=12.0,mew=1.)
                        amp_dropped_top.append(a_in[i])
                        p_in.pop(i)
                        d_in.pop(i)
                        a_in.pop(i)
                        #as_in.pop(i)
                        #ap_in.pop(i)
                        i -= 1
                        n_dropped_top += 1
                        if(verbose==True):
                            print "Top-exiting ray ==> ray dropped."
            i += 1                   
        
    #####################################
    #    WRITE RAY INFO TO TERMINAL     #
    #####################################

    print ""
    print "Total number of incoming rays                       : %g"%nrays
    print "Total number of rays incident on lens               : %g"%len(a_incoming)
    print "Total amplitude of rays incident on lens            : %.4f"%sum(a_incoming)
    print ""
    print "Total number of amplitude cutoff rays dropped       : %g"%n_dropped_cutoff
    print "Total amplitude lost due to amplitude cutoff        : %.4f"%sum(amp_dropped_cutoff)
    print ""
    print "Total number of top-exiting rays dropped            : %g"%n_dropped_top
    print "Total amplitude lost due to top-exiting rays        : %.4f"%sum(amp_dropped_top)
    print ""
    print "Total number of bottom-exiting rays                 : %g"%len(a_exit)
    print "Total amplitude of bottom-exiting rays              : %.4f"%sum(a_exit)
    print "Total intensity of bottom-exiting rays              : %.4f"%sum([i**2 for i in a_exit])
    print ""

    #####################################
    #           WRITE OUTPUT            #
    #####################################

    ### write out position (x,y,z), direction (x,y,z), and intensity of exiting ray
    print ""
    print ">>> Writing results for position, direction, and intensity of exiting rays to "+outfile
    print ""
    
    int_exit = [i**2 for i in a_exit] # convert ray amplitude to intensity
    data_out = np.column_stack((p_exit,d_exit,int_exit))
    data_out_header = "X position, Y position, Z position, X direction, Y direction, Z direction, Intensity"
    np.savetxt(outfile,data_out,header=data_out_header,fmt="%.8f")

    if(verbose==True):
        t1 = time.time()
        trun = t1-t0
        print ">>> DONE!"
        print "Total runtime = %.5f sec"%trun
        print ""
        log.close()


    #####################################
    #         PLOT RAY TRACING          #
    #####################################

    if(plot==True):
        
        ### draw lens (x-y plane) and add labels to ray tracing plot
        for r in r_sph_norm:
            draw_sph(c0,r,0)
        
        # set tick labels
        xticks = np.arange(-1.,1.1,0.5)
        yticks = np.arange(-1.,1.1,0.5)
        pl.gca().set_xticks(xticks)
        pl.gca().set_yticks(yticks)
        pl.gca().set_xticklabels([r"$%.1f$"%i for i in xticks])
        pl.gca().set_yticklabels([r"$%.1f$"%i for i in yticks])
        
        ### minor ticks
        xminticks = MultipleLocator(0.1)
        yminticks = MultipleLocator(0.1)
        pl.gca().xaxis.set_minor_locator(xminticks)
        pl.gca().yaxis.set_minor_locator(yminticks)
        
        pl.gca().set_xlim(-1.1,1.1)
        pl.gca().set_ylim(-1.1,1.1)

        pl.title(r'${\rm c)}$')
        pl.xlabel(r'${\rm X}$')
        pl.ylabel(r'${\rm Y}$')

        ### create plot of initial wavefront and input
        ### rays with lens edge-on (x-y plane)

        pl.subplot(221)
        
        for i in np.arange(len(plt_pts)):
            for j in np.arange(len(plt_pts[0])):
                if(mode == 'stars'):
                    #pl.plot(plt_pts[i][j][0],plt_pts[i][j][1],marker='o',color=clr[i],linewidth=0)
                    pass
                else:
                    pl.plot(plt_pts[i][j][0],plt_pts[i][j][1],marker='o',color='g',linewidth=0)
        if(mode == 'stars'):
            for i in np.arange(0,len(star_plt)):
                pl.plot(star_plt[i][0],star_plt[i][1],marker='*',color=clr[i],linewidth=0)   

        for r in r_sph_norm:
            draw_sph(c0,r,0)
            
        # set tick labels
        xticks = np.arange(-2.,2.1,0.5)
        yticks = np.arange(-1.,2.1,0.5)
        pl.gca().set_xticks(xticks)
        pl.gca().set_yticks(yticks)
        pl.gca().set_xticklabels([r"$%.1f$"%i for i in xticks])
        pl.gca().set_yticklabels([r"$%.1f$"%i for i in yticks])
        
        # minor ticks
        xminticks = MultipleLocator(0.1)
        yminticks = MultipleLocator(0.1)
        pl.gca().xaxis.set_minor_locator(xminticks)
        pl.gca().yaxis.set_minor_locator(yminticks)

        pl.gca().set_xlim(-1.1,1.1)
        pl.gca().set_ylim(-1.1,2.1)

        pl.title(r'${\rm a)}$')
        pl.xlabel(r'${\rm X}$')
        pl.ylabel(r'${\rm Y}$')
        
        ### create plot of initial input points and corresponding wavefronts
        ### perspective is looking at lens from above (x-z plane)

        pl.subplot(222)

        for i in np.arange(len(plt_pts)):
            for j in np.arange(len(plt_pts[0])):
                if(mode == 'stars'):
                    #pl.plot(plt_pts[i][j][0],plt_pts[i][j][2],marker='o',color=clr[i],linewidth=0)
                    pass
                else:
                    pl.plot(plt_pts[i][j][0],plt_pts[i][j][2],marker='o',color='g',linewidth=0,zorder=1)
        if(mode == 'stars'):
            for i in np.arange(0,len(star_plt)):
                pl.plot(star_plt[i][0],star_plt[i][2],marker='*',ms=9.0,color=clr[i],linewidth=0,zorder=1)

        for r in r_sph_norm:
            draw_sph(c0,r,0)
        

        # set tick labels
        xticks = np.arange(-1.,1.1,0.5)
        yticks = np.arange(-1.,1.1,0.5)
        pl.gca().set_xticks(xticks)
        pl.gca().set_yticks(yticks)
        pl.gca().set_xticklabels([r"$%.1f$"%i for i in xticks])
        pl.gca().set_yticklabels([r"$%.1f$"%i for i in yticks])
        
        # minor ticks
        xminticks = MultipleLocator(0.1)
        yminticks = MultipleLocator(0.1)
        pl.gca().xaxis.set_minor_locator(xminticks)
        pl.gca().yaxis.set_minor_locator(yminticks)

        pl.gca().set_xlim(-1.1,1.1)
        pl.gca().set_ylim(-1.1,1.1)

        pl.title(r'${\rm b)}$')
        pl.xlabel(r'${\rm X}$')
        pl.ylabel(r'${\rm Z}$')

        ### plot 2D projection of output from bottom half on lens
        ### perspective is looking at lens from below (x-z plane)

        pl.subplot(224)

        p_exit_copy = p_exit[:]
        a_exit_copy = a_exit[:]

        while(len(p_exit_copy) > 0):

            pt = p_exit_copy.pop(0)
            a1 = a_exit_copy.pop(0)

            intensity = a1*a1

            r = np.sqrt( (pt[0]*pt[0]) + (pt[1]*pt[1]) + (pt[2]*pt[2]) )
            d = np.sqrt( (pt[0]*pt[0]) + (pt[2]*pt[2]) )
            theta = np.arctan2(pt[2],pt[0])-np.pi/2.

            xplot = np.sin(theta) * np.arcsin(d/r) * 2. / np.pi
            zplot = np.cos(theta) * np.arcsin(d/r) * 2. / np.pi

            pl.plot(xplot,zplot,'o',markersize=2.5,color='b',mec='b',alpha=intensity[0],zorder=1)

        for r in r_sph_norm:
            draw_sph_2Dproj(c0,r,0)
            
        # set tick labels
        xticks = np.arange(-1.,1.1,0.5)
        yticks = np.arange(-1.,1.1,0.5)
        pl.gca().set_xticks(xticks)
        pl.gca().set_yticks(yticks)
        pl.gca().set_xticklabels([r"$%.1f$"%i for i in xticks])
        pl.gca().set_yticklabels([r"$%.1f$"%i for i in yticks])
        
        # minor ticks
        xminticks = MultipleLocator(0.1)
        yminticks = MultipleLocator(0.1)    
        pl.gca().xaxis.set_minor_locator(xminticks)
        pl.gca().yaxis.set_minor_locator(yminticks)

        pl.gca().set_xlim(-1.1,1.1)
        pl.gca().set_ylim(-1.1,1.1)
        
        pl.title(r'${\rm d})$')
        pl.xlabel(r'${\rm X}$')
        pl.ylabel(r'${\rm Z}$')

        if(verbose==True):
            print ""
            print "Saving ray tracing plot as "+figoutfile
            print ""
        pl.savefig(figoutfile)
        pl.show()
        pl.close()
        
########################################################################
