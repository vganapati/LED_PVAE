#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:34:47 2020

@author: vganapa1
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    show_figures = True
except ModuleNotFoundError:
    show_figures = False
    
from scipy.special import factorial as fact

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def Zernike_Poly(pixels_x, pixels_y, Lx, Ly, wavelength, NA, m, n):
    
    X, Y = cartesian_coords(pixels_x, pixels_y, Lx, Ly, wavelength, NA)
    
    #unit circle
    circ=np.zeros(X.shape)
    circ[np.nonzero(np.sqrt(X**2+Y**2)<=1)]=1
    
    
    rho,phi=cart2pol(X,Y)
    
    #plt.imshow(np.rot90(phi))
    
    Z=np.zeros(rho.shape)
    
    
    if (n-np.abs(m))%2==0: #condition for R to be nonzero
        R=np.zeros(rho.shape)  
        for k in range((n-np.abs(m))//2+1):
            num=((-1)**k)*(fact(n-k))
            den=fact(k)*fact((n+np.abs(m))//2-k)*fact((n-np.abs(m))//2-k)
            if den!=0:
                R=R+(num/den)*rho**(n-2*k)
                
        if m==0:
            delta = 1
        else:
            delta = 0
            
        norm = np.sqrt(2*(n+1)/(1 + delta))
        
        if m>=0:
            Z=norm*R*np.cos(m*phi)
        else:
            Z=-norm*R*np.sin(m*phi)
    
    Z=Z*circ
    return Z




def cartesian_coords(pixels_x, pixels_y, Lx, Ly, wavelength, NA):
    
    '''
    pixels_x is the number of points in the source plane field in the x (row) direction
    pixels_y is the number of points in the source plane field in the y (column) direction
    Lx and Ly are the side lengths of the observation and source fields
    wavelength is the free space wavelength
    NA is the numberical aperture
    '''

    dx = Lx/pixels_x
    dy = Ly/pixels_y
    
    k = 1./wavelength # wavenumber
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,pixels_x) #freq coords in x
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,pixels_y) #freq coords in y
    
    FX,FY=np.meshgrid(fx, fy, indexing='ij')
    
    FX = FX/(NA*k)
    FY = FY/(NA*k)
    
#    H=np.zeros([pixels_x,pixels_y])
#    H[np.nonzero(np.sqrt(FX**2+FY**2)<=1)]=1.

    return FX, FY


def get_poly_mat(pixels_x, pixels_y, Lx, Ly, wavelength, NA,
        n_upper_bound = 5, show_figures = False):
    
    
    '''
    pixels_x is the number of points in the source plane field in the x (row) direction
    pixels_y is the number of points in the source plane field in the y (column) direction
    Lx and Ly are the side lengths of the observation and source fields
    wavelength is the free space wavelength
    NA is the numberical aperture
    
    m & n are the indices of the Zernike polynomial:
    n-m must be even
    m = integer ranging from -n to +n
    n = nonnegative integer
    '''
    count = 0
    
    for n in range(0,n_upper_bound+1):
        for m in range(-n,n+1):
            if (n-m)%2==0:
                
                Z = Zernike_Poly(pixels_x, pixels_y, Lx, Ly, wavelength, NA, m, n)
                
                if show_figures:
                    plt.figure()
                    plt.title('Zernike Poly m = ' + str(m) + ', n = ' + str(n))
                    plt.imshow(Z,interpolation='none')
                    plt.colorbar()
                
                Z = np.expand_dims(Z, axis = 2)
                
                if count ==0:
                    poly_mat = Z
                else:
                    poly_mat = np.concatenate((poly_mat,Z),axis = 2)
                    
                count += 1
                
    return poly_mat

if __name__ == '__main__':
    
    pixels_x = 64
    pixels_y = 64
    Lx = 100
    Ly = 100
    wavelength = 0.632 # [um] 
    NA = 0.13
    
    get_poly_mat(pixels_x, pixels_y, Lx, Ly, wavelength, NA,
        n_upper_bound = 5, show_figures = True)