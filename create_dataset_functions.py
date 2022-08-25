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


def find_led_params(N_obj, 
                    LED_radius,
                    upsample_factor = 2,
                    dpix_c = 6.5,
                    wavelength = 0.518,
                    NA = 0.5,
                    mag = 20.0,
                    ds_led_x = 4e3,
                    ds_led_y = 4e3,
                    z_led = 69.5e3,
                    ):

    '''
    Assumes an LED array that is 32 x 32 LEDs.
    LED_radius is the radius of used LEDs from the center LED.
    '''
    
    # LED positions in LED units
    LED_vec = np.arange(0,1024)
    LED_x = LED_vec//32
    LED_y = LED_vec%32
    
    # LEDs that correspond to the center pixel of the image sensor.
    LED_center_x = 13
    LED_center_y = 13
    
    # Center LED positions
    LED_x = LED_x - LED_center_x
    LED_y = LED_y - LED_center_y
    
    LEDs_used_boolean = np.zeros_like(LED_vec, dtype=np.bool)
    LEDs_used_boolean[np.nonzero(np.sqrt(LED_x**2 + LED_y**2) <= LED_radius)] = True
        
    xx, yy = np.meshgrid(range(32),range(32))
    xx = xx - LED_center_x
    yy = yy - LED_center_y
    LitCoord = np.zeros_like(xx)
    LitCoord[np.nonzero(np.sqrt(xx**2 + yy**2) <= LED_radius)] = True
    
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
    
    # resolution achieved after post-processing
    dx0_p = 1./um_p/2.
    
    Ns = np.zeros([len(LED_vec),2])
    
    Ns[:,0]=idx_u
    Ns[:,1]=idx_v
    
        
    # Determine synthetic NAfilter
    NAfilter_synthetic = NAfilter(N_obj[0], N_obj[1], N_obj[0]*dx_obj[0], \
                                  N_obj[1]*dx_obj[1], wavelength, synthetic_NA)

    return(Ns, Np, LED_vec,\
           pupil, NAfilter_synthetic, LEDs_used_boolean, LitCoord, \
           LED_x, LED_y, ds_led_x, ds_led_y, dd, NA, dpix_c, wavelength, mag, dx_obj, synthetic_NA, \
           LED_center_x, LED_center_y, z_led)


def create_folder(folder_path):
    try: 
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise


def F(mat2D): # Fourier transform assuming center pixel as center
    return(tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(mat2D))))

def Ft(mat2D): # Inverse Fourier transform assuming center pixel as center
    return(tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(mat2D))))


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


def downsamp_slice(x, cen, Np): 
    
    '''
    Image shift in Fourier space due to single LED illumination
    '''

    return tf.slice(x, [tf.cast(cen[0], tf.int32)-tf.cast(Np[0]//2, tf.int32), \
                        tf.cast(cen[1], tf.int32)-tf.cast(Np[1]//2, tf.int32)], [Np[0], Np[1]])
    

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



def convert_img_to_obj(img, NAfilter_synthetic, 
                       vary_phase, 
                       NAfilter_function,
                       synthetic_NA,
                       filter_obj_slices=True, randomize_filter=True):
    
    '''
    Convert brightfield image to a synthetic complex object
    '''
    
    # img ranges from 0 to 1
    
    if vary_phase:
        phase_max = np.pi/2*np.random.rand()
    else:
        phase_max = np.pi/2
    
    # print(phase_max)
    
    # obj = np.exp(1j*(img-0.5)*np.pi)
    obj = np.exp(1j*(img-0.5)*phase_max) # ranges from -phase_max to +phase_max
    
    
    if filter_obj_slices:
        if randomize_filter:
            NAfilter_synthetic = NAfilter_function(synthetic_NA*(np.random.rand()*.5+.5))
            
        # filter by sythetic NA
        O = F(obj)
        O = O*NAfilter_synthetic
        obj = Ft(O) #low resolution field
        
    return obj


def process_img_multislice(img_stack, 
                           NAfilter_synthetic,
                           N_obj, Ns, P, Np,
                           LED_vec, LEDs_used_boolean,
                           random_transform,
                           vary_phase,
                           num_slices,
                           H_scalar,
                           H_scalar_f,
                           filter_obj_slices,
                           random_flag,
                           randomize_filter,
                           NAfilter_function,
                           synthetic_NA):
    
    obj_stack = []
    for s in range(num_slices):
        if random_flag:
            img0 = np.random.rand(N_obj[0],N_obj[1])
        else:
            img0 = img_stack[s,:,:]
            
        if random_transform:
            
            # random flip or rot90
            transform_ind = np.random.randint(0,len(transform_func_vec))
            # print('random transform_ind')
            # print(transform_ind)
            img = transform_func_vec[transform_ind](img0)
            
            
            if img.shape[0] < N_obj[0] and img.shape[1] < N_obj[1]:
                # random resize
                resize_x = np.random.randint(img.shape[0], N_obj[0])
                resize_y = np.random.randint(img.shape[1], N_obj[1])
                img = resize(img, (resize_x, resize_y))
            
                # random shift
                pad_x = int(N_obj[0] - img.shape[0])
                pad_y = int(N_obj[1] - img.shape[1])
                
                pad_x_0 = np.random.randint(0, pad_x)
                pad_x_1 = pad_x - pad_x_0
                
                pad_y_0 = np.random.randint(0, pad_y)
                pad_y_1 = pad_y - pad_y_0
                
                img = np.pad(img,((pad_x_0,pad_x_1),(pad_y_0,pad_y_1)), mode = 'constant')
        else:
            img = img0
            
        obj = convert_img_to_obj(img, NAfilter_synthetic, vary_phase,
                                 NAfilter_function,
                                 synthetic_NA,
                                 filter_obj_slices,
                                 randomize_filter)

        obj_stack.append(obj)
    
    
    
    
    low_res_stack = create_low_res_stack_multislice(obj_stack, N_obj, Ns, \
                                                    P, Np, LED_vec, LEDs_used_boolean, \
                                                    num_slices, \
                                                    H_scalar, H_scalar_f)
    obj_stack = np.stack(obj_stack,axis=-1)
    return low_res_stack, obj_stack

def scalar_prop_kernel(N_obj,dx_obj,z,wavelength): # x is the row coordinate, y is the column coordinate
    Nx = N_obj[0]
    Ny = N_obj[1]
    Lx = Nx*dx_obj[0]
    Ly = Ny*dx_obj[1]
    
    fx=np.linspace(-1/(2*dx_obj[0]),1/(2*dx_obj[0])-1/Lx,Nx) #freq coords
    fy=np.linspace(-1/(2*dx_obj[1]),1/(2*dx_obj[1])-1/Ly,Ny) #freq coords
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')

    FX[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0
    FY[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0
    H = np.exp(1j*2*np.pi*(1./wavelength)*z*np.sqrt(1-(wavelength*FX)**2-(wavelength*FY)**2))
    
    FX,FY=np.meshgrid(fx,fy, indexing = 'ij')
    H[np.nonzero( np.sqrt(FX**2+FY**2) > (1./wavelength) )] = 0
    
    return(H)

def create_low_res_stack_multislice(obj_stack, N_obj, Ns, \
                                    P, Np, LED_vec, LEDs_used_boolean, \
                                    num_slices, H_scalar, H_scalar_f):
    
    '''
    Create stack of single LED images from the high resolution object stack
    '''

    count = 0
    for ind in LED_vec[LEDs_used_boolean]:
        low_res = create_low_res_multislice(obj_stack, N_obj, Ns, \
                                            P, Np, ind, num_slices,\
                                            H_scalar, H_scalar_f)    
        
        low_res = tf.expand_dims(low_res, axis=-1)

        if count == 0:
            low_res_stack = low_res
        else:
            low_res_stack = tf.concat([low_res_stack,low_res],-1)  
        count += 1
    
    return low_res_stack

def create_low_res_stack_multislice2(obj_stack, N_obj, Ns, 
                                     P, Np, 
                                     LED_vec_i, 
                                     num_slices, H_scalar, H_scalar_f, 
                                     batch_size):
    
    '''
    Create stack of single LED images from the high resolution object stack
    LED_vec_i is LED_vec[LEDs_used_boolean]
    '''

    # low_res_stack = []

    low_res_stack = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    
    for ind in tf.range(batch_size):
        LED_i = tf.gather(LED_vec_i,ind)
        low_res = create_low_res_multislice(obj_stack, N_obj, Ns, \
                                            P, Np, LED_i, num_slices,\
                                            H_scalar, H_scalar_f)    
        
        low_res_stack = low_res_stack.write(low_res_stack.size(), low_res)
        # low_res = tf.expand_dims(low_res, axis=-1)

    # low_res_stack = tf.concat(low_res_stack,axis=-1) 

    return(tf.transpose(low_res_stack.stack(),perm=[1,2,0]))

def create_low_res_multislice(obj_stack, 
                              N_obj, Ns, P, Np, LED_i, 
                              num_slices,
                              H_scalar, H_scalar_f):
    
    '''
    Creation of single low resolution image with a pixel shift given by Ns
    
    Ns is angle with respect to topmost slice
    
    '''
    
    # Ns_i = Ns[LED_i,:]
    Ns_i = tf.gather(Ns, LED_i)
    cen = (N_obj/2-Ns_i) + N_obj/2 # cen = (N_obj//2+Ns[LED_i,:]) + N_obj//2
    
    O = F(obj_stack[0])
    Psi0 = tf.pad(O,((N_obj[0]//2,N_obj[0]//2),(N_obj[1]//2,N_obj[1]//2)))
    Psi0 = downsamp_slice(Psi0, cen, N_obj)
    
    for s in range(1,num_slices):
        # scalar propagate Psi0 to next object
        Psi0 = Psi0*H_scalar
        
        #Multiply by object slice in real space
        Psi0 = F(obj_stack[s]*Ft(Psi0))
    
    # scalar propagate to the focal plane
    
    Psi0 = Psi0*H_scalar_f
    
    # Slice to dimensions of Np (downsample in real space) and filter by NA
    Psi0 = tf.slice(Psi0, tf.cast(N_obj//2,tf.int32) - tf.cast(Np//2, tf.int32), Np)*P
    
    psi0 = Ft(Psi0) #low resolution field
    
    intensity_i = psi0*tf.math.conj(psi0)
    
    intensity_i = tf.cast(intensity_i, dtype=tf.float64)

    return intensity_i


def process_dataset_multislice(x_train_stack, process_img_func, normalizer, normalizer_ang,
                               offset, offset_ang,
                               add_poisson_noise, poisson_noise_multiplier,
                               save_folder_name, random_flag, truncate_number_train, sub_folder_prefix = 'example_'):
    if random_flag:
        num_train = truncate_number_train
    else:
        num_train = x_train_stack.shape[0]
        
    for i in range(num_train):
        print(i)
        if random_flag:
            im_stack, obj_stack = process_img_func(x_train_stack[0,:,:,:])  
        else:
            im_stack, obj_stack = process_img_func(x_train_stack[i,:,:,:])  
        
        sub_folder_name = '{}/{}{:06d}'.format(save_folder_name, sub_folder_prefix, i)          
        # sub_folder_name = save_folder_name + '/' + sub_folder_prefix + str(i)
        create_folder(sub_folder_name)
        im_stack_converted = convert_uint_16(im_stack.numpy(), normalizer, offset, add_poisson_noise, poisson_noise_multiplier)
        np.save(sub_folder_name + '/im_stack.npy', im_stack_converted/float(2**16-1))
        file_name_obj = sub_folder_name + '/obj_stack.npy'
        np.save(file_name_obj, obj_stack)
        
        for z in range(im_stack_converted.shape[-1]):
            # imageio.imwrite as a png
            num_str = str(z)
            file_name = sub_folder_name + '/Photo' + '0'*(4-len(num_str)) + num_str + '.png'
            imageio.imwrite(file_name, im_stack_converted[:,:,z])


        # save object stack
        
        # save object real and imag
        obj_re = np.real(obj_stack)
        obj_im = np.imag(obj_stack)
        obj_re_converted = convert_uint_16(obj_re, normalizer_ang[0], offset_ang[0], add_poisson_noise, poisson_noise_multiplier) 
        obj_im_converted = convert_uint_16(obj_im, normalizer_ang[1],offset_ang[1], add_poisson_noise, poisson_noise_multiplier)

        sub_folder_reconstruction_name = '{}/{}'.format(sub_folder_name, 'reconstruction')  
        create_folder(sub_folder_reconstruction_name)
        
        # obj_ang = np.angle(obj_stack)
        # obj_ang =  obj_ang - np.min(obj_ang)
        # obj_ang_converted = convert_uint_16(obj_ang, normalizer_ang, False, poisson_noise_multiplier) # add_poisson_noise == False
        # sub_folder_reconstruction_name = '{}/{}'.format(sub_folder_name, 'reconstruction')  
        # create_folder(sub_folder_reconstruction_name)
        

        for s in range(obj_re_converted.shape[-1]):
            # imageio.imwrite as a png
            num_str = str(s)
            file_name = sub_folder_reconstruction_name + '/Photo' + '0'*(4-len(num_str)) + num_str + '.png'
            imageio.imwrite(file_name, obj_re_converted[:,:,s])
        
        for s in range(obj_im_converted.shape[-1]):
            # imageio.imwrite as a png
            num_str = str(s+obj_re_converted.shape[-1])
            file_name = sub_folder_reconstruction_name + '/Photo' + '0'*(4-len(num_str)) + num_str + '.png'
            imageio.imwrite(file_name, obj_im_converted[:,:,s])

        
def create_img_stack(x_train, num_slices, different_slices = False):
    x_train_stack = []
    
    for ss in range(num_slices):
        if different_slices:
            np.random.shuffle(x_train)
        x_train_stack.append(x_train.copy())
    
    x_train_stack = np.stack(x_train_stack, axis=1)
    return x_train_stack

def synthetic_filter_obj(obj, 
                         NAfilter_synthetic,
                         batch_size,
                         num_slices,
                         ):
    for b in range(batch_size):
        for s in range(num_slices):
            obj[b,:,:,s] = Ft(F(obj[b,:,:,s])*NAfilter_synthetic)
    return(obj)