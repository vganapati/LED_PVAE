#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:01:01 2021

@author: vganapa1
"""
import numpy as np
import tensorflow as tf
from helper_pattern_opt import bound_mat_tf, load_img_stack, load_img_stack_real_data
from helper_functions import get_ave_output, physical_preprocess,\
    log_prob_saturation, configure_for_performance, configure_for_performance_real_data, positive_range_base
import tensorflow_probability as tfp
import glob
from fpm_functions import create_low_res_stack_multislice
tfd = tfp.distributions

'''
# Gradient testing

with tf.GradientTape() as tape:
    x=tf.Variable(-1e-2)
    y=tf.Variable(-1e-2)
    # dist=tfd.Beta(positive_range(x),positive_range(y))
    dist=tfd.Normal(x,sqrt_reg)
    z=dist.sample()
    # t = dist.log_prob(bound_mat_tf(z))
    t = dist.log_prob(sqrt_reg*10)
grads = tape.gradient(t,x)
print(grads)

'''

def create_dataset_iter(input_path,
                        save_path,
                        restore,
                        truncate_dataset,
                        batch_size,
                        num_patterns,
                        example_num,
                        save_tag_multiplexed,
                        real_data,
                        image_x=None,
                        image_y=None,
                        full_image_x=None,
                        full_image_y=None,
                        led_position_xy=None,
                        dpix_m=None,
                        z_led=None,
                        wavelength=None,
                        NA=None,
                        img_coords_xm=None, img_coords_ym=None, # full field coordinated
                        du=None,
                        um_m=None,
                        real_mult=False,
                        multiplexed_description='',
                        ):
    
    '''
    default None parameters only needed for real data
    '''
    if restore:
        train_folders = np.load(save_path + '/train_folders.npy')
        truncate_dataset = len(train_folders) # overrides the truncate_dataset value passed in       
    else:
        train_folders = []
        data_file_path = input_path + '/training/example_*'

        if truncate_dataset == 1:
            all_folders = [sorted(glob.glob(data_file_path))[example_num]]
        else:
            if not(real_data):
                truncate_dataset = truncate_dataset - truncate_dataset % batch_size 
            all_folders = sorted(glob.glob(data_file_path))[0:truncate_dataset]
            
        img_stack_count = len(all_folders)
        print('Number of Image Stacks:')
        print(img_stack_count)
        print('Image Stack Folder Names:')
        print(all_folders)

        train_folders = np.array(all_folders)
        np.save(save_path + '/train_folders.npy', train_folders)
    
    

        # np.random.shuffle(train_folders)
    
    '''
    print(tf.data.experimental.cardinality(train_ds).numpy())
    '''
    
    num_leds = int(np.load(input_path + '/num_leds.npy'))

    if real_data:
        r_channels = None
    else:
        r_channels = len(glob.glob(input_path + '/training/example_000000' + '/reconstruction/Photo*.png'))


    all_alpha_train = np.load(input_path + '/' + save_tag_multiplexed + '/all_alpha_train' + multiplexed_description + '.npy')
    np.save(save_path + '/all_alpha_train.npy', all_alpha_train)

    if truncate_dataset == 1:
        all_alpha_train = np.expand_dims(all_alpha_train[example_num],axis=0)      

    
    train_ds = tf.data.Dataset.from_tensor_slices(train_folders)
    
    alpha_train_ds = tf.data.Dataset.from_tensor_slices(all_alpha_train)
    
    train_ds = tf.data.Dataset.zip((train_ds, alpha_train_ds))



    if real_data:
        load_img_stack2 = lambda image_path, alpha: \
            load_img_stack_real_data(image_path, num_patterns, 
                                     alpha,
                                     save_tag_multiplexed,
                                     image_x,
                                     image_y,
                                     full_image_x,
                                     full_image_y,
                                     img_coords_xm, img_coords_ym, # full field coords
                                     led_position_xy,
                                     dpix_m,
                                     z_led,
                                     wavelength,
                                     NA,
                                     du,
                                     um_m,
                                     bit_depth=16, 
                                     real_mult=real_mult,
                                     multiplexed_description=multiplexed_description,
                                     )
        # output is [image_path, alpha, im_stack_multiplexed, img_coords_xm, img_coords_ym, Ns_0, pupil, synthetic_NA, cos_theta]
    else:
        load_img_stack2 = lambda image_path, alpha: \
            load_img_stack(image_path, num_leds, num_patterns, r_channels, 
                           alpha,
                           bit_depth = 16, 
                           save_tag_multiplexed = save_tag_multiplexed,
                           )
        # output is [image_path, im_stack, im_stack_r, alpha, im_stack_multiplexed]
    
    autotune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(load_img_stack2, num_parallel_calls=autotune)

    buffer_size = np.minimum(truncate_dataset,100)
    
    if real_data:
        configure_func = configure_for_performance_real_data
    else:
        configure_func = configure_for_performance
        
    train_ds_no_shuffle = configure_func(train_ds, batch_size, autotune, shuffle=False, buffer_size = buffer_size, repeat=True)
    train_ds = configure_func(train_ds, batch_size, autotune, shuffle=True, buffer_size = buffer_size, repeat=True)


    train_ds = iter(train_ds)

    
    return(train_ds, train_ds_no_shuffle, r_channels, load_img_stack2,
           train_folders)


def find_loss_vae_unsup(alpha,
                        im_stack_multiplexed,
                        num_blocks,  
                        image_x,
                        image_y,
                        num_leds,
                        model_encode,
                        model_decode,
                        poisson_noise_multiplier,
                        sqrt_reg,
                        num_patterns,
                        batch_size,
                        prior,
                        training,
                        normalizer=None,
                        normalizer_ang=None,
                        offset=None,
                        offset_ang=None,
                        use_window=False,
                        window_2d=None,
                        window_2d_sqrt_us=None,
                        use_prior = False, # does not use conditional q(z|M)
                        kl_anneal = 1,
                        kl_multiplier=1,
                        pr_offset = 0.1,
                        num_samples = 2,
                        use_normal = False,
                        N_obj = None,
                        Ns = None,
                        pupil = None,
                        Np = None,
                        LED_vec = None,
                        LEDs_used_boolean = None,
                        num_slices = None,
                        H_scalar = None,
                        H_scalar_f = None,
                        deterministic = False,
                        use_first_skip = True,
                        anneal_std = 0,
                        eps_anneal=0,
                        img_coords_xm=None, 
                        img_coords_ym=None, 
                        cos_theta=None,
                        real_data=False,
                        exposure_time_used=None,
                        zernike_mat=None,
                        change_Ns = False,
                        vary_pupil = False,
                        ):
    
    '''
    
    Overall process is:

    R -------> M ------> z -------> R ------> M   
    
    '''
    
    positive_range = lambda x: positive_range_base(x, offset = pr_offset)

    ### M --> Z ###
    ### take im_stack_multiplexed_vec and alpha and process through model_encode ###
    
    ave_skips, total_skip_weight \
     = get_ave_output(num_blocks+1,
                      num_patterns,
                      num_patterns,
                      im_stack_multiplexed,
                      alpha,
                      batch_size,
                      image_x,
                      image_y,
                      num_leds,
                      model_encode,
                      sqrt_reg,
                      training,
                      real_data=real_data,
                      img_coords_xm=img_coords_xm, 
                      img_coords_ym=img_coords_ym, 
                      )
 

    ### create conditional latent variable distribution q(z|M) ###
    # all the skip connections are be counted as latent variables as well

    if deterministic:
        q = None
    else:
        q = []
        for i in range(num_blocks+1):
            loc, log_scale = tf.split(ave_skips[i], [ave_skips[i].shape[-1]//2, ave_skips[i].shape[-1]//2], axis=-1, num=None, name='split')
            scale = positive_range(log_scale)
            if use_normal:
                q.append(tfd.Normal(loc=loc, scale=scale+eps_anneal))
            else:
                q.append(tfd.Beta(positive_range(loc), scale))

    ### z --> R ###
    # sample all the latent variables q(z|X) and process through model_decode
    
    output_dist_vec = []
    Ns_dist_vec = []
    zernike_dist_vec = []
    cos_dist_vec = []
    pnm_dist_vec = []
    
    log_prob_M_vec = []
    for s in range(num_samples):
        if deterministic:
            q_sample = ave_skips
        else:
            if use_prior:
                q_sample = [prior[i].sample() for i in range(len(q))]
            else:
                q_sample  = [q[i].sample() for i in range(len(q))]
        
        if real_data or vary_pupil:
            if change_Ns:
                im_stack_alpha, im_stack_beta, \
                Ns_delta_mean, Ns_delta_var, \
                zernike_coeff_mean, zernike_coeff_var, \
                cos_delta_mean, cos_delta_var, pnm_delta_mean, pnm_delta_var, \
                    = model_decode((q_sample), training = training)
            else:
                im_stack_alpha, im_stack_beta, \
                zernike_coeff_mean, zernike_coeff_var, \
                    = model_decode((q_sample), training = training)
        else:
            im_stack_alpha, im_stack_beta = model_decode((q_sample), training = training)
                
        if real_data:
            output_dist = tfd.Normal(im_stack_alpha, positive_range(im_stack_beta))
            zernike_dist = tfd.Normal(zernike_coeff_mean, positive_range(zernike_coeff_var))
            if change_Ns:
                Ns_dist = tfd.Normal(Ns_delta_mean, positive_range(Ns_delta_var))
                cos_dist = tfd.Normal(cos_delta_mean, positive_range(cos_delta_var))
                pnm_dist = tfd.Normal(pnm_delta_mean, positive_range(pnm_delta_var))
            
            output_dist_vec.append(output_dist)
            zernike_dist_vec.append(zernike_dist)
            
            if change_Ns:
                Ns_dist_vec.append(Ns_dist)
                cos_dist_vec.append(cos_dist)
                pnm_dist_vec.append(pnm_dist)
            
            output_sample = output_dist.sample()
            zernike_sample = zernike_dist.sample()
            if change_Ns:
                Ns_sample = Ns_dist.sample()
                cos_sample = cos_dist.sample()
                pnm_sample = pnm_dist.sample()
            
            if change_Ns:
                Ns_new = Ns + tf.cast(Ns_sample, tf.float64)
                
                pnm_new = poisson_noise_multiplier+tf.cast(pnm_sample, tf.float64)
                pnm_new = tf.expand_dims(tf.expand_dims(tf.expand_dims(pnm_new,-1),-1),0)
                
                cos_new = cos_theta + tf.cast(cos_sample, tf.float64)
            else:
                Ns_new = Ns
                pnm_new = poisson_noise_multiplier
                cos_new = cos_theta
                

            
            log_prob_R_given_z = tf.reduce_sum(output_dist.log_prob(output_sample), axis=[1,2,3]) + \
                                 tf.reduce_sum(zernike_dist.log_prob(zernike_sample), axis=[1])

            if change_Ns:
                log_prob_R_given_z += tf.reduce_sum(Ns_dist.log_prob(Ns_sample), axis=[1,2]) + \
                    tf.reduce_sum(cos_dist.log_prob(cos_sample), axis=[1]) + \
                    tf.reduce_sum(pnm_dist.log_prob(pnm_sample), axis=[1])
            
            zernike_sample = tf.expand_dims(tf.expand_dims(zernike_sample,1),1)
            pupil_angle = tf.cast(tf.reduce_sum(zernike_mat*tf.cast(zernike_sample, tf.float64), axis=-1), tf.complex128)
            
            
            im_stack_multiplexed_dist_0, _ , _ = calculate_log_prob_M_given_R_real_data(output_sample, # batch_size x image_x x image_y x (2*num_slices)
                                                                                    tf.expand_dims(alpha,axis=0), # expand for max_steps, dims are: max_steps x batch_size x num_leds x num_patterns
                                                                                    batch_size,
                                                                                    pnm_new,
                                                                                    sqrt_reg,
                                                                                    N_obj,
                                                                                    Ns_new,
                                                                                    cos_new,
                                                                                    pupil*tf.exp(1j*pupil_angle),
                                                                                    Np,
                                                                                    num_leds,
                                                                                    num_slices,
                                                                                    H_scalar,
                                                                                    H_scalar_f,
                                                                                    exposure_time_used,
                                                                                    use_window,
                                                                                    window_2d,
                                                                                    window_2d_sqrt_us,
                                                                                    change_Ns = change_Ns,
                                                                                    anneal_std = anneal_std,
                                                                                    )
        
        else:
            # output a beta distribution, dims: batch_size x image_x x image_y x num_leds
            if use_normal:
                output_dist = tfd.Normal(positive_range(im_stack_alpha), positive_range(im_stack_beta))
            else:
                output_dist = tfd.Beta(positive_range(im_stack_alpha), positive_range(im_stack_beta))
            
            output_dist_vec.append(output_dist)
        
            output_sample = output_dist.sample()
            
            if vary_pupil:
                zernike_dist = tfd.Normal(zernike_coeff_mean, positive_range(zernike_coeff_var))
                zernike_dist_vec.append(zernike_dist)
                zernike_sample = zernike_dist.sample()
                zernike_sample_expand = tf.expand_dims(tf.expand_dims(zernike_sample,1),1)
                pupil_angle = tf.cast(tf.reduce_sum(zernike_mat*tf.cast(zernike_sample_expand, tf.float64), axis=-1), tf.complex128)
                # print(pupil_angle.shape)
                # print(pupil.shape)
                pupil_new = pupil*tf.exp(1j*pupil_angle)
            else:
                pupil_new = tf.expand_dims(pupil,0)
                pupil_new = tf.repeat(pupil_new, batch_size, axis=0)
            
            pupil_new = tf.cast(pupil_new, tf.complex64)
            if use_normal:
                log_prob_R_given_z = output_dist.log_prob(output_sample)
            else:
                log_prob_R_given_z = output_dist.log_prob(bound_mat_tf(output_sample, sqrt_reg, 1-sqrt_reg))
            
            log_prob_R_given_z = tf.reduce_sum(log_prob_R_given_z, axis=[1,2,3])
            if vary_pupil:
                log_prob_R_given_z += tf.reduce_sum(zernike_dist.log_prob(zernike_sample), axis=[1])
                                     
            im_stack_multiplexed_dist_0 = calculate_log_prob_M_given_R(output_sample, # batch_size x image_x x image_y x num_leds
                                                                       tf.expand_dims(alpha,axis=0), # expand for max_steps, dims are: max_steps x batch_size x num_leds x num_patterns
                                                                       batch_size,
                                                                       poisson_noise_multiplier,
                                                                       sqrt_reg,
                                                                       1, # max_steps
                                                                       normalizer,
                                                                       normalizer_ang,
                                                                       offset,
                                                                       offset_ang,
                                                                       N_obj = N_obj,
                                                                       Ns = Ns,
                                                                       pupil = pupil_new,
                                                                       Np = Np,
                                                                       LED_vec = LED_vec,
                                                                       LEDs_used_boolean = LEDs_used_boolean,
                                                                       num_slices = num_slices,
                                                                       H_scalar = H_scalar,
                                                                       H_scalar_f = H_scalar_f,
                                                                       anneal_std = anneal_std,
                                                                       )
            # log_prob_R_given_z = tf.reduce_sum(log_prob_R_given_z, axis=[1,2,3]) 
            
        if real_data:
            dtype = tf.float64
        else:
            dtype = tf.float32
        
        if use_window:
            log_prob_M_given_R = log_prob_saturation(tf.expand_dims(im_stack_multiplexed*tf.expand_dims(tf.expand_dims(window_2d,0),-1), 
                                                                    axis=0), 
                                                     im_stack_multiplexed_dist_0, dtype)
        else:
            log_prob_M_given_R = log_prob_saturation(tf.expand_dims(im_stack_multiplexed, axis=0), 
                                                     im_stack_multiplexed_dist_0, dtype)
        
        
        log_prob_M = tf.cast(tf.reduce_sum(tf.squeeze(log_prob_M_given_R, axis=0), axis=[1,2,3]),tf.float32)\
                     + log_prob_R_given_z
 
        log_prob_M_vec.append(log_prob_M)
    
            
    # VAE objective

    if deterministic:
        kl_divergence = 0
    else:
        if use_first_skip:
            kl_divergence = [tf.reduce_sum(tfp.distributions.kl_divergence(q[i], prior[i]),axis=[1,2,3]) for i in range(num_blocks+1)]
        else:
            kl_divergence = [tf.reduce_sum(tfp.distributions.kl_divergence(q[i], prior[i]),axis=[1,2,3]) for i in range(1,num_blocks+1)]
        kl_divergence = tf.reduce_sum(kl_divergence,axis=0)

    negloglik = -tf.reduce_mean(log_prob_M_vec,axis=0)

    loss_M_VAE = kl_anneal*kl_multiplier*kl_divergence + negloglik    

    return(loss_M_VAE, alpha, im_stack_multiplexed,
           output_dist_vec, q, q_sample, kl_divergence, -negloglik,
           Ns_dist_vec,
           zernike_dist_vec,
           cos_dist_vec,
           pnm_dist_vec,
           )


def calculate_log_prob_M_given_R_real_data(output_sample,
                                           alpha, # max_steps x batch_size x num_leds x num_patterns
                                           batch_size,
                                           poisson_noise_multiplier,
                                           sqrt_reg,
                                           N_obj,
                                           Ns, # num_leds x 2
                                           cos_theta,
                                           pupil,
                                           Np,
                                           num_leds,
                                           num_slices,
                                           H_scalar,
                                           H_scalar_f,
                                           exposure_time_used,
                                           use_window,
                                           window_2d,
                                           window_2d_sqrt_us,
                                           change_Ns = True,
                                           anneal_std = 0,
                                           max_steps = 1, # XXX assumed to be 1
                                           ):
    '''    
    im_stack dims: batch_size x image_x x image_y x num_leds
    '''
    
    LED_vec = np.arange(num_leds)
    
    if use_window:
        window_2d_sqrt_us= tf.cast(window_2d_sqrt_us, tf.complex128)
    # find im_stack from output_sample
    im_stack = []
    for b in range(batch_size):
        obj_re, obj_im = tf.split(output_sample[b],2,axis=-1)
    
        obj = tf.cast(obj_re, tf.complex128) + tf.cast(obj_im, tf.complex128)*1j
        obj = tf.transpose(obj, perm=[2,0,1])
        im_stack_i = create_low_res_stack_multislice(obj, N_obj, 
                                                     Ns[b], # num_leds x 2
                                                     pupil[b], Np, 
                                                     LED_vec,
                                                     num_slices, 
                                                     H_scalar, H_scalar_f,
                                                     num_leds, # batch_size in regards to fpm optimizer
                                                     change_Ns,
                                                     use_window,
                                                     window_2d_sqrt_us)
        im_stack.append(tf.expand_dims(im_stack_i, axis=0))
    im_stack = tf.concat(im_stack,axis=0)

    
    # adjust im_stack by the exposure
    im_stack = im_stack*tf.expand_dims(exposure_time_used,0)
    
    # adjust im_stack by cos_theta**4 dropoff
    im_stack = im_stack*tf.expand_dims(tf.expand_dims(cos_theta**4, -1),-1)
    
    # im_stack is batch_size x num_leds x image_x x image_y
    
    im_stack = tf.transpose(im_stack, perm=[0,2,3,1]) # put num_leds last
    # im_stack is now batch_size x image_x x image_y x num_leds
    
    # shape is max_steps x batch_size x image_x x image_y x num_patterns
    im_stack_multiplexed_dist_0, \
    im_stack_multiplexed_0 = physical_preprocess(im_stack, 
                                                 tf.cast(alpha, tf.float64),
                                                 poisson_noise_multiplier,
                                                 sqrt_reg,
                                                 batch_size,
                                                 max_steps,
                                                 renorm = False,
                                                 normalizer = None,
                                                 offset = None,
                                                 zero_alpha = False,
                                                 return_dist = True,
                                                 anneal_std=anneal_std,
                                                 quantize_noise = True,
                                                 )

    
    return(im_stack_multiplexed_dist_0, im_stack_multiplexed_0, im_stack)


def calculate_log_prob_M_given_R(output_sample,
                                 alpha, # max_steps x batch_size x num_leds x num_patterns
                                 batch_size,
                                 poisson_noise_multiplier,
                                 sqrt_reg,
                                 max_steps,
                                 normalizer,
                                 normalizer_ang,
                                 offset,
                                 offset_ang,
                                 N_obj = None,
                                 Ns = None,
                                 pupil = None,
                                 Np = None,
                                 LED_vec = None,
                                 LEDs_used_boolean = None,
                                 num_slices = None,
                                 H_scalar = None,
                                 H_scalar_f = None,
                                 visualize = False,
                                 anneal_std = 0,
                                 ):
    '''    
    im_stack dims: batch_size x image_x x image_y x num_leds
    '''
    

        
    # find im_stack from output_sample
    im_stack = []
    for b in range(batch_size):
        obj_re, obj_im = tf.split(output_sample[b],2,axis=-1)
        obj_re = obj_re/normalizer_ang[0] + offset_ang[0]
        obj_im = obj_im/normalizer_ang[1] + offset_ang[1]
        obj = tf.cast(obj_re, tf.complex64) + tf.cast(obj_im, tf.complex64)*1j
        obj = tf.transpose(obj, perm=[2,0,1])
        im_stack_i = create_low_res_stack_multislice(obj, N_obj, Ns, 
                                                     pupil[b], Np, LED_vec[LEDs_used_boolean],  
                                                     num_slices, 
                                                     H_scalar, H_scalar_f,
                                                     np.sum(LEDs_used_boolean),
                                                     False, # change_Ns
                                                     False, # use_window
                                                     None, # window_2d_sqrt_us
                                                     )
        im_stack.append(tf.expand_dims(im_stack_i, axis=0))
    im_stack = tf.concat(im_stack,axis=0)
    im_stack = tf.cast(im_stack, dtype = tf.float32)
    im_stack = tf.transpose(im_stack, perm=[0,2,3,1]) # put num_leds last
    
    # normalize and offset this im_stack
    im_stack = (im_stack - offset)*normalizer

        
    # shape is max_steps x batch_size x image_x x image_y x num_patterns
    im_stack_multiplexed_dist_0, \
    im_stack_multiplexed_0 = physical_preprocess(im_stack, 
                                                 alpha,
                                                 tf.cast(poisson_noise_multiplier, tf.float32),
                                                 sqrt_reg,
                                                 batch_size,
                                                 max_steps,
                                                 renorm = True,
                                                 normalizer = normalizer,
                                                 offset = offset,
                                                 zero_alpha = False,
                                                 return_dist = True,
                                                 anneal_std=anneal_std,
                                                 quantize_noise = True,
                                                 dtype = tf.float32
                                                 )
    if visualize:
        im_stack_multiplexed_0 = physical_preprocess(im_stack, 
                                                     alpha,
                                                     poisson_noise_multiplier,
                                                     sqrt_reg,
                                                     batch_size,
                                                     max_steps,
                                                     renorm = True,
                                                     normalizer = normalizer,
                                                     offset = offset,
                                                     zero_alpha = False,
                                                     return_dist = False,
                                                     dtype = tf.float32
                                                     )
            
        return(im_stack_multiplexed_0,im_stack)
    
    return(im_stack_multiplexed_dist_0)


