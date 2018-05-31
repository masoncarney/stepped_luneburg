import os, sys
import numpy as np

def norm(v):
    '''Normalizes a given vector and returns the normalized vector.
    
    ==========
    Parameters
    
    v: [array of floats] vector to be normalized
    
    ==========    
    '''
    
    # numerical instability
    # if ray goes through exact center, offset by small amount 
    # to avoid division by zero error in vnorm
    if(v[0]==0. and v[1]==0. and v[2]==0.):
        v=v+1.e-10
        
    vmag = np.sqrt(sum(v[i]*v[i] for i in range(len(v))))
    vnorm = np.array([v[i]/vmag for i in range(len(v))])
    
    return vnorm

def sphere_hit(p0, d0, c0, r):
    '''Uses a point in 3D space, direction vector, sphere center, and sphere radius to determine if the ray vector will intersect with the sphere. Returns minimum value t for the vector-sphere intersection, indicating that the vector hits the sphere closest to the initial point p0. If the ray vector does not intersect with a sphere then value of t = -1 is returned.
    
    ==========
    Parameters
    
    p0: [array of floats] initial point in 3D space [x,y,z]
    
    d0: [array of floats] initial direction vector in 3D space [x,y,z]
    
    c0: [array of floats] center of sphere in 3D space [x,y,z]
    
    r: [float] radius of sphere for determining vector-sphere intersection
    
    ==========    
    '''

    # from page: http://www.cs.umbc.edu/~olano/435f02/ray-sphere.html
    # routine finds intersection of sphere-ray using determinant

    t0 = 0.
    t1 = 0.
    t2 = 0.
    a = np.dot(d0,d0)
    b = 2.*np.dot(d0,(p0-c0)) 
    c = np.dot((p0-c0),(p0-c0)) - r**2.    
    det = b**2. - 4.*a*c

    # if det is negative, no hit    
    # if det is positive, ray hits sphere    
    # if positive, return lowest positive value of t    
    # if negative, return t = -1 to show that ray did not intersect

    if det < 0.:
        return -1.
    if det == 0.:
        t0 = -b/(2.*a)
        if t0 >= 0.:
            return t0
        else:
            return -1.
    if det > 0.:
        t1 = (-b + np.sqrt(det))/(2.*a)
        t2 = (-b - np.sqrt(det))/(2.*a)

        # numerical instability - in case we are really close to
        # a given sphere, but don't actually hit, we treat this as a non-hitting ray
        # and set it to -1 to prevent false images coming through
        if t1 < 1.e-10:
            t1 = -1.
        if t2 < 1.e-10:
            t2 = -1.

        if t1 < 0. and t2 < 0.:
            return -1.
        if (t1*t2) < 0.:
            return max(t1,t2)
        else:
            return min(t1,t2)


def snell_vec(nv,lv,n1,n2):
    '''Uses the normal vector, light ray direction vector, and two indices of refraction from either side of a given sphere radius to calculate the direction and amplitude of a reflected and refracted ray. Returns the direction vectors of the reflected and refracted rays, and the amplitude of the reflected ray.

    CAUTION: When dot product of direction vector and normal vector goes negative, routine fails and calculates a negative cth1 which produces false values for reflection coefficient

    ==========
    Parameters
    
    nn: [array of floats] direction vector for the normal plane [x,y,z]
    
    nl: [array of floats] direction vector for the light ray [x,y,z]
    
    n1: [float] index of refraction on incoming ray side of interface
    
    n2: [float] index of refraction on refracted ray side of interface
    
    ==========        
    '''

    #first, calculate reflected ray

    n = norm(nv) #normal vector normalized
    l = norm(lv) #ray direction vector normalized
    nr = n1/n2
    cth1 = np.dot(n,(-l))
    v_reflect = l + (2.*cth1*n)

    cth2_sq = 1. - nr*nr*(1.-(cth1*cth1))

    #this is cosine theta2 squared, a test for total internal reflection

    if(cth2_sq > 0.):
        cth2 = np.sqrt(cth2_sq)
        #we can safely find the square root of cth2 since cth2_sq is positive
        #calculate reflection coefficients for reflected ray (2 polarizations)
        rs = (n1*cth1-n2*cth2)/(n1*cth1+n2*cth2)
        rp = (n1*cth2-n2*cth1)/(n1*cth2+n2*cth1)
        Rs = rs*rs
        Rp = rp*rp
        R = (Rs+Rp)/2. #mean reflected energy
        
        #calculate the refracted ray
        v_refract = nr*l + (nr*cth1-cth2)*n
        
        return(v_reflect, v_refract, R, Rs, Rp)
    else:
        #if cth2_sq is less than zero, we have total internal reflection
        print "Warning: total internal reflection of ray"
        v_null = np.array([0.,0.,0.])
        return(v_reflect,v_null,1.0,1.0,1.0)
