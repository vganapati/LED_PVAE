#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:12:10 2020

@author: vganapa1
"""
import os
import numpy as np
import tensorflow as tf
import imageio
from skimage.transform import resize

transform_func_vec = [lambda img: img,
                      lambda img: np.rot90(img, k=1), 
                      lambda img: np.flipud(img), 
                      lambda img: np.fliplr(img),
                      lambda img: np.rot90(img, k=1),
                      lambda img: np.rot90(img, k=2),
                      lambda img: np.rot90(img, k=3)]

def find_fpm_params(N_obj, 
                    LED_radius,
                    LED_radius_inner,
                    upsample_factor = 2,
                    dpix_c = 6.5,
                    wavelength = 0.518,
                    NA = 0.5,
                    mag = 20.0,
                    ds_led_x = 4e3,
                    ds_led_y = 4e3,
                    z_led = 69.5e3,
                    ):

    # Wavelength of illumination 
    # wavelength = 0.518 #0.643 # [um] 
    
    # Numerical Aperture of Objective
    # NA = 0.5 #0.25 # 0.15 # 0.1
    
    # Magnification of the system 
    # mag = 8/4*10.0
    
    # # Size of pixel on sensor plane 
    # dpix_c = 6.5
    
    # LED array geometries
    # ds_led_x = 4e3 # [um] spacing between neighboring LEDs
    # ds_led_y = 4e3 # [um] spacing between neighboring LEDs
    
    # # z distance from LED array to sample [um]
    # z_led = 69.5e3 #74e3
    
    # Factor of how many more pixels are in the high resolution image in a single dimension
    # upsample_factor = 2 # Check to make sure factor is large \
                        # enough by looking at pupil in Fourier space

    
    # LED positions in LED units
    LED_vec = np.arange(0,1024)
    LED_x = LED_vec//32
    LED_y = LED_vec%32
    
    '''
    LED_radius = 2
    '''
    
    # LEDs that correspond to the center pixel of the sample
    LED_center_x = 13
    LED_center_y = 13
    
    # Center LED positions
    LED_x = LED_x - LED_center_x
    LED_y = LED_y - LED_center_y
    
    LEDs_used_boolean = np.zeros_like(LED_vec, dtype=np.bool)
    LEDs_used_boolean[np.nonzero(np.sqrt(LED_x**2 + LED_y**2) >= LED_radius_inner)] = True
    LEDs_used_boolean[np.nonzero(np.sqrt(LED_x**2 + LED_y**2) > LED_radius)] = False
    
    
    xx, yy = np.meshgrid(range(32),range(32))
    xx = xx - LED_center_x
    yy = yy - LED_center_y
    LitCoord = np.zeros_like(xx)
    LitCoord[np.nonzero(np.sqrt(xx**2 + yy**2) >= LED_radius_inner)] = True
    LitCoord[np.nonzero(np.sqrt(xx**2 + yy**2) > LED_radius)] = False
    
    print('Total LEDs used is: ' + str(np.sum(LEDs_used_boolean)))
    
    if (N_obj[0]%upsample_factor) or (N_obj[1]%upsample_factor):
        print('WARNING: N_obj is not a multiple of upsample_factor.')
        
    # Number of pixels of low-resolution images
    Np = (N_obj/upsample_factor).astype(np.int32)
    
    
    # Maximum spatial frequency of low-resolution images set by NA 
    um_m = NA/wavelength 
    
    # System resolution based on NA 
    dx0 = 1./um_m/2. 
    
    # Effective image pixel size on the object plane 
    dpix_m = dpix_c/mag 
    
    # FoV (object space)
    FoV = Np*dpix_m
    
    # Sampling size in Fourier plane
    du = 1./FoV 
    
    # Low pass filter set-up 
    m = (np.arange(0, Np[0], 1) - Np[0]/2)*du[0]
    n = (np.arange(0, Np[1], 1) - Np[1]/2)*du[1]
    
    # Generate a meshgrid 
    # mm: vertical
    # nn: horizontal 
    [mm,nn] = np.meshgrid(m,n, indexing='ij')
    # Find radius of each pixel from center 
    ridx = np.sqrt(mm**2+nn**2)
    
    # assume a circular pupil function, low pass filter due to finite NA
    w_NA = np.zeros(ridx.shape)
    w_NA[np.nonzero(ridx<um_m)] = 1.
    # Define aberration 
    aberration = np.ones([Np[0], Np[1]])
    # Define phase constant 
    phC = np.ones([Np[0], Np[1]])
    # Generate pupil function 
    pupil = w_NA * phC * aberration 
    

    
    # spacing in the object plane
    dx_obj = FoV/N_obj
    
    ###
    
    # angles for each LEDs
    dd = np.sqrt((LED_y*ds_led_y)**2+(LED_x*ds_led_x)**2+z_led**2)
    sin_theta_x = (LED_x*ds_led_x)/dd
    sin_theta_y = (LED_y*ds_led_y)/dd
    
    ### corresponding spatial freq for each LEDs
    xled = sin_theta_x/wavelength
    yled = sin_theta_y/wavelength
    
    ### spatial freq index for each plane wave relative to the center
    idx_u = xled/du[0]
    idx_v = yled/du[1]
    
    
    illumination_na_used = np.sqrt(sin_theta_x**2+sin_theta_y**2)
    
    # number of brightfield image LEDs
    NBF = len(np.nonzero(illumination_na_used<=NA)[0])
    print('number of brightfield LEDs is: ' + str(NBF))
    
    
    # maxium spatial frequency achievable based on the maximum illumination
    # angle from the LED array and NA of the objective
    um_p = np.max(illumination_na_used[LEDs_used_boolean])/wavelength+um_m
    
    synthetic_NA = um_p*wavelength
    print('Synthetic NA is: ' + str(synthetic_NA))
    # resolution achieved after freq post-processing
    dx0_p = 1./um_p/2.
    
    Ns = np.zeros([len(LED_vec),2])
    
    Ns[:,0]=idx_u
    Ns[:,1]=idx_v
    
        
    # Determine synthetic NAfilter
    NAfilter_synthetic = NAfilter(N_obj[0], N_obj[1], N_obj[0]*dx_obj[0], \
                                  N_obj[1]*dx_obj[1], wavelength, synthetic_NA)

    return Ns, Np, LED_vec,\
           pupil, NAfilter_synthetic, LEDs_used_boolean, LitCoord, \
           LED_x, LED_y, ds_led_x, ds_led_y, dd, NA, dpix_c, wavelength, mag, dx_obj, synthetic_NA, \
           LED_center_x, LED_center_y, z_led


def create_folder(folder_path):
    try: 
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise


def F(mat2D): # Fourier transform assuming center pixel as center
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(mat2D)))

def Ft(mat2D): # Inverse Fourier transform assuming center pixel as center
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(mat2D)))


def NAfilter(m, n, Lx, Ly, wavelength, NA):
    
    '''
    m is the number of points in the source plane field in the x (row) direction
    n is the number of points in the source plane field in the y (column) direction
    Lx and Ly are the side lengths of the observation and source fields
    wavelength is the free space wavelength
    NA is the numberical aperture
    '''

    dx = Lx/m
    dy = Ly/n
    
    k = 1./wavelength # wavenumber
    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/Lx,m) #freq coords in x
    fy=np.linspace(-1/(2*dy),1/(2*dy)-1/Ly,n) #freq coords in y
    
    FX,FY=np.meshgrid(fx, fy, indexing='ij')
    
    H=np.zeros([m,n])
    H[np.nonzero(np.sqrt(FX**2+FY**2)<=NA*k)]=1.

    return(H)  



def convert_uint_16(im_stack, normalizer, offset, add_poisson_noise, poisson_noise_multiplier):
    # Add noise to im_stack and save as uint16 PNGs
    im_stack = im_stack-offset
    im_stack = im_stack*normalizer
    if add_poisson_noise:
        im_stack = im_stack*poisson_noise_multiplier
        im_stack = np.random.poisson(im_stack)
        im_stack = im_stack/poisson_noise_multiplier
    im_stack = im_stack*(2**16-1)
        
    im_stack[np.nonzero(im_stack<0)] = 0
    im_stack[np.nonzero(im_stack>(2**16-1))] = 2**16 - 1
    im_stack = im_stack.astype(np.uint16)

    return im_stack
    
def find_normalizer(im_stack, reduce_max_factor=1.0, offset=None):
    if offset is None: # else use offset value passed
        offset = np.min(im_stack)
    im_stack = im_stack - offset
    normalizer = (1.0/np.max(im_stack))*reduce_max_factor
    return(normalizer, offset)
        
def get_paddings(pad_x):
    if pad_x%2:
        # Not divisible by 2
        pad_x_0 = np.floor(pad_x/2)
        pad_x_1 = np.ceil(pad_x/2)
    else:
        pad_x_0 = pad_x/2
        pad_x_1 = pad_x/2
    return int(pad_x_0), int(pad_x_1)


