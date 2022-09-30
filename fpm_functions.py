#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:11:06 2021

@author: vganapa1
"""
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
from dpc_algorithm import DPCSolver
import skimage.transform
from SyntheticMNIST_functions import F, Ft

def create_initial_phase(left,
                         right,
                         top,
                         bottom,
                         wavelength,
                         NA,
                         dpix_m,
                         upsample_factor):
    # Create L-R DPC image
    # dpc_lr = (left - right) / (left + right)
    
    # Create top-bottom DPC image
    # dpc_tb = (top - bottom) / (top + bottom)
    
    # initial_phase = (dpc_lr + dpc_tb)/2.

    na_in = 0.0
    dpc_num = 4 #number of DPC images captured for each absorption and phase frame
    pixel_size = dpix_m
    rotation = [0, 180, 90, 270] #degree
    
    dpc_stack = np.stack((bottom, top, left, right), axis=0)
    dpc_solver_obj = DPCSolver(dpc_stack, wavelength, NA, na_in, pixel_size, rotation, dpc_num=dpc_num)

    #parameters for Tikhonov regurlarization [u:absorption, p:phase] ((need to tune this based on SNR)
    dpc_solver_obj.setRegularizationParameters(reg_u = 1e-1, reg_p = 1e-1)
    dpc_result = dpc_solver_obj.solve(method="Tikhonov")

    # initial_real = dpc_result[0].real
    initial_phase = dpc_result[0].imag

    initial_phase = \
        skimage.transform.rescale(initial_phase, upsample_factor, multichannel = False, order = 0, mode = 'constant')
        
    return(initial_phase)




def shift(f,H):
    return np.real(Ft(F(f)*H))
    


def shift_add(lr_observed_stack, Np, img_coords_xm,
              img_coords_ym, led_position_xy, NA,
              wavelength, dpix_m, z_led, 
              upsample_factor,
              z_vec):
    
    # patch size
    x_patch_size, y_patch_size = img_coords_xm.shape
    
    # Maximum spatial frequency of low-resolution images set by NA 
    um_m = NA/wavelength 
    
    # FoV (object space)
    FoV = np.array([x_patch_size,y_patch_size])*dpix_m
    
    # Sampling size in Fourier plane
    du = 1./FoV 

    # Low pass filter set-up 
    m = (np.arange(0, x_patch_size, 1) - x_patch_size/2)*du[0]
    n = (np.arange(0, y_patch_size, 1) - y_patch_size/2)*du[1]
    
    # Generate a meshgrid 
    # mm: vertical
    # nn: horizontal 
    [mm,nn] = np.meshgrid(m,n, indexing='ij')
    # Find radius of each pixel from center 
    ridx = np.sqrt(mm**2+nn**2)
    
    # assume a circular pupil function, low pass filter due to finite NA
    pupil = np.zeros(ridx.shape)
    pupil[np.nonzero(ridx<um_m)] = 1.


    patch_x_center = img_coords_xm[x_patch_size//2, y_patch_size//2]
    patch_y_center = img_coords_ym[x_patch_size//2, y_patch_size//2]
    
    led_position_x = led_position_xy[:,0]
    led_position_y = led_position_xy[:,1]
    
    # angles for each LED
    tan_theta_x = (patch_x_center-led_position_x)/z_led
    tan_theta_y = (patch_y_center-led_position_y)/z_led
    
    dd = np.sqrt((led_position_x-patch_x_center)**2+(led_position_y-patch_y_center)**2+z_led**2)

    sin_theta_x = (patch_x_center-led_position_x)/dd
    sin_theta_y = (patch_y_center-led_position_y)/dd
    
    
    illumination_na_led = np.sqrt(sin_theta_x**2+sin_theta_y**2) # NA of LED
    num_bf = np.sum(illumination_na_led<= NA) # brightfield LEDs

    Nz = len(z_vec)
    
    try:
        lr_observed_stack = lr_observed_stack.astype(dtype=np.complex64)
    except AttributeError:
        lr_observed_stack = tf.cast(lr_observed_stack, dtype=tf.complex64)
        
    # umax = 1./2./dpix_m
    # du = 1./dpix_m/x_patch_size
    # dv = 1./dpix_m/y_patch_size
    # u = np.arange(-umax, umax, du[0])
    # v = np.arange(-umax, umax, du[1])
    u=m
    v=n
    [u,v] = np.meshgrid(u,v, indexing='ij')
    
    
    tot_mat = np.zeros([Nz, Np[0], Np[1]])
    initial_amplitude_mat = np.zeros([Nz, Np[0]*upsample_factor, Np[1]*upsample_factor])
    initial_phase_mat = np.zeros([Nz, Np[0]*upsample_factor, Np[1]*upsample_factor])
    

    # left_mat = np.zeros([Nz, Np[0], Np[1]])
    # right_mat = np.zeros([Nz, Np[0], Np[1]])
    # top_mat = np.zeros([Nz, Np[0], Np[1]])
    # bottom_mat = np.zeros([Nz, Np[0], Np[1]])
    # DPC_lr_mat = np.zeros([Nz, Np[0], Np[1]])
    # DPC_tb_mat = np.zeros([Nz, Np[0], Np[1]])
    
    

    for z_ind in range(Nz):
        tot = np.zeros(Np)
        

        left = np.zeros(Np)
        right = np.zeros(Np)
        top = np.zeros(Np)
        bottom = np.zeros(Np)
        
    
        for led_ind in range(len(lr_observed_stack)):
            
            if illumination_na_led[led_ind]<= NA: # only consider bright field LEDs for init conditions
                
                img = lr_observed_stack[led_ind,:,:]
                
                ## shift
                # compute shift in Fourier space for considering subpixel shift
                shift_x = z_vec[z_ind] * tan_theta_x[led_ind]
                shift_y = z_vec[z_ind] * tan_theta_y[led_ind]
                Hs = np.exp(1j*2*np.pi*(shift_x*u+shift_y*v))
                # shifted image
                img = shift(img,Hs)
                
    
                # add
                if tan_theta_x[led_ind]>0:
                    left = left + img
                elif tan_theta_x[led_ind]<=0:
                    right = right + img
        
                if tan_theta_y[led_ind]>0:
                    top = top + img
                elif tan_theta_y[led_ind]<=0:
                    bottom = bottom + img
                
                # refocused brightfield
                tot = tot + img
        
        # Average over number of images
        tot /= float(num_bf)
        left /= float(num_bf)
        right /= float(num_bf)
        top /= float(num_bf)
        bottom /= float(num_bf)
        
        tot_mat[z_ind,:,:] = tot
        
        # initial_phase = create_initial_phase(left,
        #                                       right,
        #                                       top,
        #                                       bottom,
        #                                       wavelength,
        #                                       NA,
        #                                       dpix_m,
        #                                       upsample_factor)
 
        initial_phase = create_initial_phase(right,
                                             left,
                                             bottom,
                                             top,
                                             wavelength,
                                             NA,
                                             dpix_m,
                                             upsample_factor)        
   
 
        initial_phase_mat[z_ind] = initial_phase
        
        # # computed refocused two-axis DPC
        # # Left right DPC
        # DPC_lr = (left-right)/tot
        # # Top bottom DPC
        # DPC_tb = (top-bottom)/tot
        
        # left_mat[z_ind,:,:] = left
        # right_mat[z_ind,:,:] = right
        # top_mat[z_ind,:,:] = top
        # bottom_mat[z_ind,:,:] = bottom
        # DPC_lr_mat[z_ind,:,:] = DPC_lr
        # DPC_tb_mat[z_ind,:,:] = DPC_tb

        # initial_amplitude = np.sqrt(np.maximum(np.real(lr_observed_stack[0]),0))/(upsample_factor**2) # center led only
        initial_amplitude = np.sqrt(np.maximum(tot,0))/(upsample_factor**2)
        initial_amplitude = \
            skimage.transform.rescale(initial_amplitude, upsample_factor, multichannel = False, order = 0, mode = 'constant')
        initial_amplitude_mat[z_ind] = initial_amplitude

    return(initial_amplitude_mat, initial_phase_mat, tot_mat)




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

    return H  

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


def find_Ns(img_coords_xm,
            img_coords_ym,
            led_position_xy,
            dpix_m,
            z_led,
            wavelength,
            NA,
            ):
    
    # patch size
    x_patch_size, y_patch_size = img_coords_xm.shape
    
    # Maximum spatial frequency of low-resolution images set by NA 
    um_m = NA/wavelength 
    
    # FoV (object space)
    FoV = np.array([x_patch_size,y_patch_size])*dpix_m
    
    # Sampling size in Fourier plane
    du = 1./FoV 

    # Low pass filter set-up 
    m = (np.arange(0, x_patch_size, 1) - x_patch_size/2)*du[0]
    n = (np.arange(0, y_patch_size, 1) - y_patch_size/2)*du[1]
    
    # Generate a meshgrid 
    # mm: vertical
    # nn: horizontal 
    [mm,nn] = np.meshgrid(m,n, indexing='ij')
    # Find radius of each pixel from center 
    ridx = np.sqrt(mm**2+nn**2)
    
    # assume a circular pupil function, low pass filter due to finite NA
    pupil = np.zeros(ridx.shape)
    pupil[np.nonzero(ridx<um_m)] = 1.


    patch_x_center = img_coords_xm[x_patch_size//2, y_patch_size//2]
    patch_y_center = img_coords_ym[x_patch_size//2, y_patch_size//2]
    
    led_position_x = led_position_xy[:,0]
    led_position_y = led_position_xy[:,1]
    
    # angles for each LEDs
    dd = np.sqrt((led_position_x-patch_x_center)**2+(led_position_y-patch_y_center)**2+z_led**2)
    sin_theta_x = (patch_x_center-led_position_x)/dd
    sin_theta_y = (patch_y_center-led_position_y)/dd
    
    cos_theta = z_led/dd
    
    ### corresponding spatial freq for each LEDs
    xled = sin_theta_x/wavelength
    yled = sin_theta_y/wavelength
    
    ### spatial freq index for each plane wave relative to the center
    idx_u = xled/du[0]
    idx_v = yled/du[1]
    
    illumination_na_used = np.sqrt(sin_theta_x**2+sin_theta_y**2)
    
    # number of brightfield image LEDs
    NBF = len(np.nonzero(illumination_na_used<=NA)[0])
    
    print('number of brightfield LEDs: ' + str(NBF))
    # maxium spatial frequency achievable based on the maximum illumination
    # angle from the LED array and NA of the objective
    um_p = np.max(illumination_na_used)/wavelength+um_m
    
    synthetic_NA = um_p*wavelength
    print('synthetic NA is : ' + str(synthetic_NA))
        
    # resolution achieved after freq post-processing
    dx0_p = 1./um_p/2.
    print('achievable resolution is : ' + str(dx0_p))
    
    Ns = np.zeros([len(led_position_xy),2])
    
    Ns[:,0]=idx_u
    Ns[:,1]=idx_v

    return(Ns, pupil, synthetic_NA, cos_theta)





def create_low_res_stack_multislice(obj_stack, N_obj, Ns, 
                                     P, Np, 
                                     LED_vec_i, 
                                     num_slices, H_scalar, H_scalar_f, 
                                     batch_size, change_Ns, use_window,
                                     window_2d_sqrt):
    
    '''
    Create stack of single LED images from the high resolution object stack
    '''
    low_res_stack = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    
    for ind in tf.range(batch_size):
        LED_i = tf.gather(LED_vec_i,ind)
        low_res = create_low_res_multislice(obj_stack, N_obj, Ns, \
                                            P, Np, LED_i, num_slices,\
                                            H_scalar, H_scalar_f, change_Ns, use_window,
                                            window_2d_sqrt)    
        
        low_res_stack = low_res_stack.write(low_res_stack.size(), low_res)
        # low_res = tf.expand_dims(low_res, axis=-1)

    # low_res_stack = tf.concat(low_res_stack,axis=-1) 

    return(low_res_stack.stack()) #(tf.transpose(low_res_stack.stack(),perm=[1,2,0]))

def create_low_res_multislice(obj_stack, 
                              N_obj, Ns, P, Np, LED_i, 
                              num_slices,
                              H_scalar, H_scalar_f,
                              change_Ns, 
                              use_window,
                              window_2d_sqrt):
    
    '''
    Creation of single low resolution image with a pixel shift given by Ns
    
    Ns is angle with respect to topmost slice
    
    '''
    # Ns_i = Ns[LED_i,:]
    Ns_i = tf.gather(Ns, LED_i)
    cen = (N_obj/2+Ns_i) + N_obj/2 # cen = (N_obj//2-Ns[LED_i,:]) + N_obj//2 or (N_obj/2+Ns_i) + N_obj/2

    if change_Ns or use_window:
        Psi0 = downsamp_subpixel(obj_stack[0], cen, Np, N_obj, use_window, window_2d_sqrt)
    else:
        O = F(obj_stack[0])
        Psi0 = tf.pad(O,((N_obj[0]//2,N_obj[0]//2),
                         (N_obj[1]//2,N_obj[1]//2)))
        Psi0 = downsamp_slice(Psi0, cen, N_obj)
        
    for s in range(1,num_slices):
        # scalar propagate Psi0 to next object
        Psi0 = Psi0*H_scalar
        
        #Multiply by object slice in real space
        Psi0 = F(obj_stack[s]*Ft(Psi0))
    
    # scalar propagate to the focal plane
    
    Psi0 = Psi0*H_scalar_f
    
    # Slice to dimensions of Np (downsample in real space) and filter by NA
    # print(Psi0)
    # print(P)
    Psi0 = tf.slice(Psi0, tf.cast(N_obj//2,tf.int32) - tf.cast(Np//2, tf.int32), Np)*tf.cast(P, tf.complex64)
    
    psi0 = Ft(Psi0) #low resolution field
    
    intensity_i = psi0*tf.math.conj(psi0)
    
    intensity_i = tf.cast(intensity_i, dtype=tf.float64)

    return intensity_i

def downsamp_slice(x, cen, Np): 
    
    '''
    Image shift in Fourier space due to single LED illumination
    '''

    return tf.slice(x, [tf.cast(cen[0], tf.int32)-tf.cast(Np[0]//2, tf.int32), \
                        tf.cast(cen[1], tf.int32)-tf.cast(Np[1]//2, tf.int32)], [Np[0], Np[1]])


def downsamp_subpixel(high_res_obj, cen, Np, N_obj, use_window, window_2d_sqrt):
                
    u=np.linspace(-1./2,1./2-1./N_obj[0],int(N_obj[0])) #freq coords in x
    v=np.linspace(-1./2,1./2-1./N_obj[1],int(N_obj[1])) #freq coords in y
    
    [uu,vv] = np.meshgrid(u,v,indexing='ij')        
    shift_x = tf.gather(cen,0)
    shift_y = tf.gather(cen,1)
    exponent = 2*np.pi*(shift_x*uu+shift_y*vv)
    exponent = tf.cast(exponent, dtype=tf.complex128)
    hs = tf.exp(1j*exponent)
    if use_window:
        X2 = F(window_2d_sqrt*high_res_obj*hs)
    else:
        X2 = F(high_res_obj*hs)
    # X3 = X2[int(N_obj[0]//2-Np[0]//2):int(N_obj[0]//2+Np[0]//2), \
    #         int(N_obj[1]//2-Np[1]//2):int(N_obj[1]//2+Np[1]//2)]  
    return X2


def psnr_complex(recon0,recon1):
    delta = np.max(np.abs(recon0)) - np.min(np.abs(recon0))
    mse = np.mean((np.abs(recon0-recon1))**2)
    psnr = 10*np.log10(np.abs(delta)**2/mse)
    return(psnr,mse)

def compare(recon0, recon1):

    '''
    recon0 is the reference
    '''
    
    obj_ref = np.angle(recon0)
    obj_compare = np.angle(recon1)
    ssim_recon_angle = ssim(obj_ref, obj_compare,
                      data_range=obj_ref.max() - obj_ref.min())

    obj_ref = np.abs(recon0)
    obj_compare = np.abs(recon1)
    ssim_recon_abs = ssim(obj_ref, obj_compare,
                      data_range=obj_ref.max() - obj_ref.min())

    obj_ref = (np.abs(recon0))**2
    obj_compare = (np.abs(recon1))**2
    ssim_recon_intensity = ssim(obj_ref, obj_compare,
                      data_range=obj_ref.max() - obj_ref.min())

    psnr_recon, mse_recon = psnr_complex(recon0,recon1)
    
    err_string_ssim = 'SSIM angle: {:.3f}, SSIM abs: {:.3f}, SSIM intensity: {:.3f}'
    print(err_string_ssim.format(ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity))    
    
    err_string = 'MSE: {:.8f}, PSNR: {:.3f}'
    print(err_string.format(mse_recon, psnr_recon))
    
    return(mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity)

def find_angle_offset(ground_truth, compare_obj):
    '''
    find the best angle offset
    '''
    compute_MSE = lambda angle_offset: np.mean((np.abs(ground_truth - compare_obj*np.exp(1j*angle_offset)))**2)
    results = minimize(compute_MSE, 0, method=None,  
                       bounds=[(-np.pi, np.pi)], tol=1e-3, options={'maxiter': 2000, 'disp':False})
    return(results.x)