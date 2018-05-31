import os, sys
import numpy as np
import pylab as pl

def n_luneburg(r_sph,n_pow,eq):
    '''Generates indices of refraction for each shell radius based on Luneburg solutions with largest radius normalized to r=1

    ==========
    Parameters

    r_sph: [float or array of floats] radius (radii) of lens layer(s)
    
    n_pow: [float] power-law exponent
    
    eq: [boolean True/False] True for modified power law, False for original power law
        original: n = (2-r^2)^exp
        modified: n = (2-r^(2*exp))
    
    ==========    
    '''
    
    if(eq == 0):
        r_sph_norm = r_sph/(max(r_sph)+0.001)
        n = ((2-r_sph_norm**2)**(n_pow))
        n = np.append(n,1.0)
        return n

    if(eq == 1):
        r_sph_norm = r_sph/(max(r_sph)+0.001)
        n = (2-r_sph_norm**(2*n_pow))
        n = np.append(n,1.0)

        return n

def rotate(vec_list,ang1_list,ang2_list):
    '''Rotates points about two axes in two steps. It intakes a list of arrays for point positions (vec_list) with each array consisting of [x,y,z] values. Also intakes a list of two angles. It rotates the points in the x-y plane about the z axis for the first angle, then rotates the points again in the x-z plane about the y axis for the second angle.

    ==========
    Parameters

    vec_list: [list of array of floats] direction vector(s) to be rotated

    ang1_list: [list of array of floats] angle(s) for rotation in x-y plane
 
    ang2_list: [list of array of floats] angle(s) for rotation in x-z plane
    
    ==========    
    '''

    new_vec_list = []

    while(len(vec_list) > 0):

        vec = vec_list.pop(0)
        if(type(ang1_list) == type(list())):
            ang1 = ang1_list.pop(0)
        else:
            ang1 = ang1_list
        if(type(ang2_list) == type(list())):
            ang2 = ang2_list.pop(0)
        else:
            ang2 = ang2_list

        vec_rot1 = np.zeros(len(vec))
        
        vec_rot1[0] = vec[0]*np.cos(ang1) - vec[1]*np.sin(ang1)
        vec_rot1[1] = vec[0]*np.sin(ang1) + vec[1]*np.cos(ang1)
        vec_rot1[2] = vec[2]
    
        vec_rot2 = np.zeros(len(vec))

        vec_rot2[0] = vec_rot1[0]*np.cos(ang2) - vec_rot1[2]*np.sin(ang2)
        vec_rot2[1] = vec_rot1[1]
        vec_rot2[2] = vec_rot1[0]*np.sin(ang2) + vec_rot1[2]*np.cos(ang2)

        new_vec_list.append(vec_rot2)

    return(new_vec_list)


def orion(y_offset=2.):
    '''Creates the points for the constellation Orion. Starting at some input distance y from the origin, RA and DEC are used to rotate the points into the correct position. Returns a list of arrays with each array containing [x,y,z] coordinates for one star in the constellation
    
    ==========
    Parameters

    y_offset: [float] offset distance along y-axis, stellar position is initialized at [0.,y,0.]
    
    ==========    
    '''

    pt = np.array([0.,y_offset,0.])
    
    ra,dec = np.loadtxt('orion.dat',skiprows=1,usecols=[0,1],unpack=True)

    ### recalculate dec as 90 deg - dec; star pattern is offset from zenith instead of horizon for ray tracing
    dec = 90.-dec
    
    ### convert ra and dec from deg to radians
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.

    ra = list(ra)
    dec = list(dec)

    pt_list = []
    for i in range(len(ra)):
        pt_list.append(pt)

    stars = rotate(pt_list,dec,ra)
    stars = rotate(stars,[0]*len(stars),[-np.pi/2.]*len(stars))
    stars = rotate(stars,[-np.pi/2.]*len(stars),[0]*len(stars))

#    from draw_sph import *
#    for i in np.arange(len(stars)):
#        pl.subplot(211)
#        pl.plot(stars[i][0],stars[i][1],marker='o')
#        if(i == 0):
#            pl.plot(stars[i][0],stars[i][1],'ro',markersize=8)
#        if(i == 1):
#            pl.plot(stars[i][0],stars[i][1],'bo',markersize=8)
#        draw_sph(np.array([0,0,0]),1)
#    
#        pl.subplot(212)
#        pl.plot(stars[i][0],stars[i][2],marker='o')
#        if(i == 0):
#            pl.plot(stars[i][0],stars[i][2],'ro',markersize=8)
#        if(i == 1):
#            pl.plot(stars[i][0],stars[i][2],'bo',markersize=8)
#        draw_sph(np.array([0,0,0]),1)

    return(stars)


def rand_grid(r,n,y_offset):
    '''rand_grid distributes random points of total number n within a circle of radius r and returns the points as a list individual arrays [x,y,z]
    
    ==========
    Parameters

    r: [float] radius of lens surface
    
    n: [float] number of points to act as initial ray positions for wavefront of rays
    
    y_offset: [float] offset distance along y-axis, ray position is initialized at [0.,y,0.]
    
    ==========    
    '''

    #generate array of random points for r, theta

    p_r = np.random.random([n])
    p_th = np.random.random([n])

    r_pt = r*np.sqrt(p_r)
    theta_pt = 2*np.pi*p_th

    #convert to cartesian
    
    x = r_pt*np.cos(theta_pt)
    y = np.zeros(n)+y_offset
    z = r_pt*np.sin(theta_pt)

    pts = []

    for i in np.arange(0,len(x)):
        pts.append(np.array([x[i],y[i],z[i]]))

    return(pts)

def uniform_grid(n,y_offset):
    '''uniform_grid creates a uniform grid of points and returns the points as a list individual arrays [x,y,z]

    ==========
    Parameters

    n: [float] number of points to act as initial ray positions for wavefront of rays
    
    y_offset: [float] offset distance along y-axis, ray position is initialized at [0.,y,0.]
    
    ==========    
    '''

    #generate array of a uniform grid of points

    pts = []

    step = 2./(np.sqrt(n)-1)
    
    for i in np.arange(-1.,1.+1e-3,step):
        for j in np.arange(-1.,1+1e-3,step):
            a = np.array([i,y_offset,j])
            pts.append(a)

    return(pts)

########################################################################
