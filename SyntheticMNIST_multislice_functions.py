#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:31:28 2021

@author: vganapa1
"""

import numpy as np
from SyntheticMNIST_functions import transform_func_vec, \
                                     F, Ft, \
                                     create_folder, \
                                     convert_uint_16
from fpm_functions import create_low_res_stack_multislice, scalar_prop_kernel                                   
from skimage.transform import resize
import tensorflow as tf
import imageio
from zernike_polynomials import get_poly_mat

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
                           synthetic_NA,
                           vary_pupil,
                           num_zernike_coeff,
                           zernike_mat,
                           change_Ns = False,
                           ):
    
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
    

    if vary_pupil:
        pupil_angle_coeff = (np.random.rand(num_zernike_coeff)-0.5)*1e-1
        pupil_angle_i = np.sum(zernike_mat*pupil_angle_coeff, axis=2)
        P = P*np.exp(1j*pupil_angle_i)  
        
    low_res_stack = create_low_res_stack_multislice(obj_stack, N_obj, Ns, \
                                                    P, Np, LED_vec[LEDs_used_boolean], \
                                                    num_slices, \
                                                    H_scalar, H_scalar_f, 
                                                    np.sum(LEDs_used_boolean), # batch_size 
                                                    change_Ns,
                                                    False, # use_window
                                                    None, # window_2d_sqrt
                                                    )

    low_res_stack = tf.transpose(low_res_stack, perm=[1,2,0]) # put num_leds last
    # print(low_res_stack.shape)
    obj_stack = np.stack(obj_stack,axis=-1)
    return low_res_stack, obj_stack



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




def find_Ns(img_coords_xm,
            img_coords_ym,
            x_patch_size,
            y_patch_size,
            led_position_xy,
            dpix_m,
            z_led,
            wavelength,
            NA,
            du,
            um_m
            ):

    patch_x_center = img_coords_xm[x_patch_size//2, y_patch_size//2]
    patch_y_center = img_coords_ym[x_patch_size//2, y_patch_size//2]
    
    led_position_x = led_position_xy[:,0]
    led_position_y = led_position_xy[:,1]
    
    # angles for each LEDs
    dd = tf.sqrt((led_position_x-patch_x_center)**2+(led_position_y-patch_y_center)**2+z_led**2)
    sin_theta_x = (patch_x_center-led_position_x)/dd
    sin_theta_y = (patch_y_center-led_position_y)/dd
    
    cos_theta = z_led/dd
    
    ### corresponding spatial freq for each LEDs
    xled = sin_theta_x/wavelength
    yled = sin_theta_y/wavelength
    
    ### spatial freq index for each plane wave relative to the center
    idx_u = xled/du[0]
    idx_v = yled/du[1]
    
    illumination_na_used = tf.sqrt(sin_theta_x**2+sin_theta_y**2)
    
    # number of brightfield image LEDs
    # NBF = len(np.nonzero(illumination_na_used<=NA)[0])
    
    # print('number of brightfield LEDs: ' + str(NBF))
    # maxium spatial frequency achievable based on the maximum illumination
    # angle from the LED array and NA of the objective
    um_p = tf.reduce_max(illumination_na_used)/wavelength+um_m
    
    synthetic_NA = um_p*wavelength
    # print('synthetic NA is : ' + str(synthetic_NA))
        
    # resolution achieved after freq post-processing
    # dx0_p = 1./um_p/2.
    # print('achievable resolution is : ' + str(dx0_p))
    
    
    # Ns = np.zeros([len(led_position_xy),2])
    
    idx_u=tf.expand_dims(idx_u, -1)
    idx_v=tf.expand_dims(idx_v, -1)

    Ns = tf.concat((idx_u,idx_v), -1)
    
    return(Ns, synthetic_NA, cos_theta)

def get_real_data_params(image_x,
                         image_y,
                         dpix_m,
                         wavelength,
                         NA,
                         zernike_poly_order,
                         x_crop_size,
                         y_crop_size,
                         upsample_factor,
                         slice_spacing,
                         f,
                         ):
    
    zernike_mat = get_poly_mat(x_crop_size, y_crop_size, image_x*dpix_m, \
                               image_y*dpix_m, wavelength, NA,
                               n_upper_bound = zernike_poly_order, show_figures = False)
        
    # coordinates in um
    img_coords_x = dpix_m*(np.arange(image_x) - image_x/2)
    img_coords_y = dpix_m*(np.arange(image_y) - image_y/2)

    img_coords_xm, img_coords_ym = np.meshgrid(img_coords_x,img_coords_y, indexing='ij')

    Np=np.array([x_crop_size, y_crop_size])
    N_obj = Np*upsample_factor

    
    dx_obj = dpix_m/upsample_factor
    dx_obj = [dx_obj,dx_obj]


    H_scalar = scalar_prop_kernel(N_obj,dx_obj,slice_spacing,wavelength)
    H_scalar_f = scalar_prop_kernel(N_obj,dx_obj,f,wavelength) # scalar prop from last plane to focal plane

    # Maximum spatial frequency of low-resolution images set by NA 
    um_m = NA/wavelength 
    
    # FoV (object space)
    FoV = np.array([x_crop_size,y_crop_size])*dpix_m
    
    # Sampling size in Fourier plane
    du = 1./FoV 
    
    # Low pass filter set-up 
    m = (np.arange(0, x_crop_size, 1) - x_crop_size/2)*du[0]
    n = (np.arange(0, y_crop_size, 1) - y_crop_size/2)*du[1]
    
    # Generate a meshgrid 
    # mm: vertical
    # nn: horizontal 
    [mm,nn] = np.meshgrid(m,n, indexing='ij')
    # Find radius of each pixel from center 
    ridx = np.sqrt(mm**2+nn**2)
    
    # assume a circular pupil function, low pass filter due to finite NA
    pupil = np.zeros(ridx.shape)
    pupil[np.nonzero(ridx<um_m)] = 1.
    return(zernike_mat,
           img_coords_xm,
           img_coords_ym,
           H_scalar,
           H_scalar_f,
           du,
           um_m,
           pupil,
           N_obj,
           Np)