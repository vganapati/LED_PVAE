#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:01:01 2021

@author: vganapa1
"""
import numpy as np
import tensorflow as tf
from helper_pattern_opt import run_episode, process_alpha, \
    bound_mat_tf, load_img_stack
from helper_functions import get_ave_output, physical_preprocess,\
    log_prob_saturation, configure_for_performance, positive_range_base
import tensorflow_probability as tfp
import glob
from SyntheticMNIST_multislice_functions import create_low_res_stack_multislice
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

def create_dataset_iter(input_path_vec,
                        save_path,
                        restore,
                        reconstruct,
                        carone,
                        truncate_dataset,
                        batch_size,
                        num_patterns,
                        unsupervised,
                        choose_patterns,
                        example_num,
                        restore_patt,
                        dirichlet_multiplier,
                        save_tag_multiplexed = None,
                        test = False,
                        ):
    if restore:
        if not(unsupervised):
            val_folders = np.load(save_path + '/val_folders.npy')
        else:
            val_folders = []
        train_folders = np.load(save_path + '/train_folders.npy')
        
        truncate_dataset = len(val_folders) + len(train_folders)
        
        val_size = len(val_folders)
        
    else:
        val_folders = []
        train_folders = []
    
        if carone:
            input_path_vec = [0]
            
        for ind, path_i in enumerate(input_path_vec):
            
            
            if carone:
                data_file_path = '/data2/CaroneLabCells/TrainingDataset/Images_SampleNumber00*_RegionNumber00*/Images/Stack0001'
            else:
                data_file_path = path_i + '/training/example_*'

            if truncate_dataset < 2: # batch size should also be 1
                all_folders = [sorted(glob.glob(data_file_path))[example_num]]
            else:
                # XXX allow a vector of truncate_dataset values
                truncate_dataset = truncate_dataset - truncate_dataset % batch_size # truncate_dataset is PER path_i
                all_folders = sorted(glob.glob(data_file_path))[0:truncate_dataset]
                
            img_stack_count = len(all_folders)
            print('Number of Image Stacks:')
            print(img_stack_count)
            print('Image Stack Folder Names:')
            print(all_folders)
    
            if unsupervised or test:
                val_size = 0
            else:
                val_size = int(img_stack_count*0.1)
                val_size = val_size - (val_size % batch_size)
                val_size = np.maximum(val_size, batch_size)
            
            if ind==0:
                val_folders = all_folders[0:val_size]
                train_folders = all_folders[val_size:]
            else:
                val_folders = val_folders + all_folders[0:val_size]
                train_folders = train_folders + all_folders[val_size:]            
        
        train_folders = np.array(train_folders)
        val_folders = np.array(val_folders)
        if not(test):
            np.save(save_path + '/val_folders.npy', val_folders)
            np.save(save_path + '/train_folders.npy', train_folders)
    
    
    # if not(test):
    #     # shuffle train_folders and val_folders
    #     np.random.shuffle(val_folders)
    #     np.random.shuffle(train_folders)
    
    
    '''
    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())
    '''
    
    if restore:
        num_leds = int(np.load(save_path + '/num_leds.npy'))
    else:
        num_leds = len(glob.glob(all_folders[0] + '/Photo*.png'))
        if not(test):
            np.save(save_path + '/num_leds.npy', num_leds)
    
    if reconstruct:
        r_channels = len(glob.glob(input_path_vec[0] + '/training/example_000000' + '/reconstruction/Photo*.png'))
    else:
        r_channels = num_leds

    if unsupervised and choose_patterns:
        if save_tag_multiplexed is not None and not(restore):
            all_alpha_train = np.load(input_path_vec[0] + '/' + save_tag_multiplexed + '/all_alpha_train.npy')
            if truncate_dataset == 1:
                all_alpha_train = np.expand_dims(all_alpha_train[example_num],axis=0)
            np.save(save_path + '/all_alpha_train.npy', all_alpha_train)
        elif restore and (restore_patt is None):
            all_alpha_train = np.load(save_path + '/all_alpha_train.npy')
        elif (restore_patt is not(None)) and truncate_dataset==1:
            all_alpha_train = np.expand_dims(np.load(restore_patt + '/all_alpha_train.npy')[example_num],axis=0)
        elif (restore_patt is not(None)):
            all_alpha_train = np.load(restore_patt + '/all_alpha_train.npy')
        else:
            # Choose alpha patterns ahead of time

            all_alpha_train = tfd.Dirichlet(dirichlet_multiplier*np.ones([num_patterns,num_leds])).sample(len(train_folders))
            all_alpha_train = tf.cast(tf.transpose(all_alpha_train, perm=[0,2,1]), dtype=tf.float32)
            
            np.save(save_path + '/all_alpha_train.npy', all_alpha_train)
        
        train_ds = tf.data.Dataset.from_tensor_slices(train_folders)
        
        alpha_train_ds = tf.data.Dataset.from_tensor_slices(all_alpha_train)
        
        train_ds = tf.data.Dataset.zip((train_ds, alpha_train_ds))

        load_img_stack2 = lambda folder_name, alpha: load_img_stack(folder_name, num_leds, num_patterns, r_channels, 
                                                                    bit_depth = 16, reconstruct = reconstruct,
                                                                    unsupervised = unsupervised, choose_patterns=choose_patterns,
                                                                    alpha=alpha, save_tag_multiplexed=save_tag_multiplexed,
                                                                    )
        val_ds = None
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(train_folders)
        val_ds = tf.data.Dataset.from_tensor_slices(val_folders)
    
        load_img_stack2 = lambda folder_name: load_img_stack(folder_name, num_leds, num_patterns, r_channels, 
                                                             bit_depth = 16, reconstruct = reconstruct,
                                                             unsupervised = unsupervised, choose_patterns=choose_patterns,
                                                             alpha=None, dirichlet_multiplier = dirichlet_multiplier)
    
    autotune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(load_img_stack2, num_parallel_calls=autotune)
    
    if not(unsupervised) and not(test): # no validation dataset for unsupervised option
        val_ds = val_ds.map(load_img_stack2, num_parallel_calls=autotune)
    
    
    buffer_size = np.minimum(truncate_dataset,100)
    train_ds_no_shuffle = configure_for_performance(train_ds, batch_size, autotune, shuffle=False, buffer_size = buffer_size, repeat=True)
    train_ds = configure_for_performance(train_ds, batch_size, autotune, shuffle=True, buffer_size = buffer_size, repeat=True)

    if not(unsupervised) and not(test):
        val_ds = configure_for_performance(val_ds, batch_size, autotune, shuffle = False, repeat=True)
    

    train_ds = iter(train_ds)
    if not(unsupervised) and not(test):
        val_ds = iter(val_ds)
    
    return(train_ds, train_ds_no_shuffle, val_ds, num_leds, r_channels, load_img_stack2,
           train_folders, val_folders)


def find_loss_vae(im_stack,
                  im_stack_r,
                  image_x,
                  image_y,
                  image_x_r,
                  image_y_r,
                  skips_pixel_x_pi,
                  skips_pixel_y_pi, 
                  skips_pixel_z_pi,
                  skips_pixel_z,
                  skips_pixel_z_R,
                  num_blocks,  
                  num_leds,
                  use_model_encode_pi,
                  model_encode_pi,
                  model_encode,
                  model_encode_R,
                  model_pi,
                  model_decode,
                  pi_iter,
                  poisson_noise_multiplier,
                  sqrt_reg,
                  num_patterns,
                  max_steps,
                  batch_size,
                  alpha_0,
                  first_step_patterns,
                  prior,
                  training,
                  skip_connect_ind,
                  normalizer,
                  offset,
                  use_prior = False, # does not use im_stack_r
                  eps_anneal = 0,
                  kl_anneal = 1,
                  kl_multiplier=1,
                  pr_offset = 0.1,
                  num_samples = 10,
                  use_normal = False,
                  deterministic=False,
                  ):
    
    '''
    
    Overall process is:
                       R & M
                         |
                         |
                         v
    R -------> M ------> z -------> R ------> M   
    
    '''
    positive_range = lambda x: positive_range_base(x, offset = pr_offset)
    
    ### R --> M ###
    ### pick paramters for measurement with model_encode_pi and model_pi ###
    
    if use_model_encode_pi:
        alpha_vec, im_stack_multiplexed_vec, state_pi_vec = \
            run_episode(im_stack,
                        image_x,
                        image_y,
                        skips_pixel_x_pi,
                        skips_pixel_y_pi, 
                        skips_pixel_z_pi,
                        num_blocks,  
                        num_leds,
                        model_encode_pi,
                        model_pi,
                        pi_iter,
                        poisson_noise_multiplier,
                        sqrt_reg,
                        num_patterns,
                        max_steps,
                        batch_size,
                        alpha_0,
                        first_step_patterns,
                        training,
                        skip_connect_ind,
                        normalizer,
                        offset,
                        )
    else:
        alpha_vec, im_stack_multiplexed_vec, state_pi_vec = \
        run_episode(im_stack,
                    image_x,
                    image_y,
                    skips_pixel_x_pi,
                    skips_pixel_y_pi, 
                    skips_pixel_z_pi,
                    num_blocks,  
                    num_leds,
                    model_encode, #model_encode_pi
                    model_pi,
                    pi_iter,
                    poisson_noise_multiplier,
                    sqrt_reg,
                    num_patterns,
                    max_steps,
                    batch_size,
                    alpha_0,
                    first_step_patterns,
                    training,
                    skip_connect_ind,
                    normalizer,
                    offset,
                    )
    
    # process alpha_vec with time_fraction_vec and exposure_time_vec
    alpha_scaled_vec = process_alpha(alpha_vec, sqrt_reg)

    # reshape im_stack_multiplexed_vec and alpha_scaled_vec
    total_num_patterns = num_patterns*max_steps
    alpha_scaled = tf.transpose(alpha_scaled_vec, perm=[1,2,3,0])
    alpha_scaled = tf.reshape(alpha_scaled, shape = [batch_size, num_leds, total_num_patterns])
    im_stack_multiplexed = tf.transpose(im_stack_multiplexed_vec, perm=[1,2,3,4,0])   
    im_stack_multiplexed = tf.reshape(im_stack_multiplexed, shape = [batch_size, image_x, image_y, total_num_patterns])
    
    # im_stack_multiplexed_vec = tf.ones_like(im_stack_multiplexed_vec)
    
    # remove patterns that are zeroed out due to activation of first_step_patterns, FEATURE DEACTIVATED
    # total_num_patterns = first_step_patterns + num_patterns*(max_steps-1), FEATURE DEACTIVATED


    ### M --> Z ###
    ### take im_stack_multiplexed_vec and alpha_scaled_vec, and process through model_encode ###
    
    ave_skips, total_skip_weight \
     = get_ave_output(num_blocks+1,
                      total_num_patterns,
                      total_num_patterns,
                      im_stack_multiplexed,
                      alpha_scaled,
                      batch_size,
                      image_x,
                      image_y,
                      num_leds,
                      model_encode,
                      sqrt_reg,
                      training,
                      )
 
    ### R & M --> Z ###
    ### take im_stack_multiplexed_vec and alpha_scaled_vec and im_stack_R, and process through model_encode ###
    ave_skips_r, total_skip_weight_r \
     = get_ave_output(num_blocks+1,
                      total_num_patterns,
                      total_num_patterns,
                      im_stack_multiplexed,
                      alpha_scaled,
                      batch_size,
                      image_x,
                      image_y,
                      num_leds,
                      model_encode_R,
                      sqrt_reg,
                      training,
                      append_r = True,
                      im_stack_r = im_stack_r,
                      image_x_r = image_x_r,
                      image_y_r = image_y_r,
                      )
    
    ### Create q(z|M,R) ###
    # appends to the middle skip connection in M --> R
    

    loc, log_scale = tf.split(ave_skips_r[-1], 
                              [ave_skips_r[-1].shape[-1]//2, 
                               ave_skips_r[-1].shape[-1]//2], axis=-1, num=None, name='split')
    scale = positive_range(log_scale)
    if use_normal:
        q = tfd.Normal(loc=loc, scale=scale+eps_anneal)
    else:
        q = tfd.Beta(positive_range(loc),scale)

    ### z --> R ###
         
    # sample the latent variables q(z|M,R), append to output of model_encode, and process through model_decode

    if use_prior:
        q_sample = prior.sample(num_samples)
    else:
        q_sample  = q.sample(num_samples)
    
    if deterministic:
        num_samples = 1
    
    output_dist_vec = []
    log_prob_R_vec = []
    for s in range(num_samples):
        q_sample_i = q_sample[s]
        
        if deterministic:
            ave_skips_i = ave_skips[-1]
        else:
            ave_skips_i = tf.concat((ave_skips[-1], q_sample_i),axis=-1)
        
        im_stack_alpha, im_stack_beta = model_decode((ave_skips[:-1] + [ave_skips_i]), 
                                                     training = training)
        
        # output a beta distribution, dims: batch_size x image_x x image_y x num_leds
        if use_normal:
            output_dist = tfd.Normal(im_stack_alpha, positive_range(im_stack_beta))
        else:
            output_dist = tfd.Beta(positive_range(im_stack_alpha), positive_range(im_stack_beta))
        output_dist_vec.append(output_dist)
    
        log_prob_R = output_dist.log_prob(bound_mat_tf(im_stack_r, sqrt_reg, 1-sqrt_reg))
        log_prob_R = tf.reduce_sum(log_prob_R, axis=[1,2,3])        
        log_prob_R_vec.append(log_prob_R)
        
    # cVAE objective

    if deterministic:
        kl_divergence = 0
    else:
        kl_divergence = tf.reduce_sum(tfp.distributions.kl_divergence(q, prior),axis=[1,2,3])
    
    negloglik = -tf.reduce_mean(log_prob_R_vec,axis=0)

    loss_M_VAE = kl_anneal*kl_multiplier*kl_divergence + negloglik    
    
    return(loss_M_VAE, alpha_vec, im_stack_multiplexed_vec,
           output_dist_vec, q, q_sample, kl_divergence, -negloglik)



def find_loss_vae_unsup(im_stack,
                        im_stack_r,
                        alpha,
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
                        reconstruct,
                        normalizer,
                        normalizer_ang,
                        offset,
                        offset_ang,
                        use_prior = False, # does not use conditional q(z|M)
                        eps_anneal = 0,
                        kl_anneal = 1,
                        kl_multiplier=1,
                        pr_offset = 0.1,
                        num_samples = 2,
                        use_normal = False,
                        N_obj = None,
                        Ns = None,
                        P = None,
                        Np = None,
                        LED_vec = None,
                        LEDs_used_boolean = None,
                        num_slices = None,
                        H_scalar = None,
                        H_scalar_f = None,
                        deterministic = False,
                        use_first_skip = True,
                        anneal_std = 0,
                        im_stack_multiplexed = None,
                        ):
    
    '''
    
    Overall process is:

    R -------> M ------> z -------> R ------> M   
    
    '''
    
    positive_range = lambda x: positive_range_base(x, offset = pr_offset)
    
    ### R --> M ###

    if im_stack_multiplexed is None:    
        # shape is max_steps x batch_size x image_x x image_y x num_patterns
        im_stack_multiplexed = physical_preprocess(im_stack, 
                                                   tf.expand_dims(alpha, axis=0), # add max_steps dim
                                                   poisson_noise_multiplier,
                                                   sqrt_reg,
                                                   batch_size,
                                                   1, # max_steps
                                                   renorm = True,
                                                   normalizer = normalizer,
                                                   offset = offset,
                                                   zero_alpha = False,
                                                   return_dist = False,
                                                   set_seed = False,
                                                   )
    
        # remove max_steps dim
        im_stack_multiplexed = tf.squeeze(im_stack_multiplexed, axis=0)


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
    log_prob_M_vec = []
    for s in range(num_samples):
        if deterministic:
            q_sample = ave_skips
        else:
            if use_prior:
                q_sample = [prior[i].sample() for i in range(len(q))]
            else:
                q_sample  = [q[i].sample() for i in range(len(q))]
        
        im_stack_alpha, im_stack_beta = model_decode((q_sample), training = training)
        
        # output a beta distribution, dims: batch_size x image_x x image_y x num_leds
        if use_normal:
            output_dist = tfd.Normal(positive_range(im_stack_alpha), positive_range(im_stack_beta))
        else:
            output_dist = tfd.Beta(positive_range(im_stack_alpha), positive_range(im_stack_beta))
        output_dist_vec.append(output_dist)
        
        
        output_sample = output_dist.sample()
        
        if use_normal:
            log_prob_R_given_z = output_dist.log_prob(output_sample)
        else:
            log_prob_R_given_z = output_dist.log_prob(bound_mat_tf(output_sample, sqrt_reg, 1-sqrt_reg))
            
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
                                                                   reconstruct = reconstruct,
                                                                   N_obj = N_obj,
                                                                   Ns = Ns,
                                                                   P = P,
                                                                   Np = Np,
                                                                   LED_vec = LED_vec,
                                                                   LEDs_used_boolean = LEDs_used_boolean,
                                                                   num_slices = num_slices,
                                                                   H_scalar = H_scalar,
                                                                   H_scalar_f = H_scalar_f,
                                                                   anneal_std = anneal_std,
                                                                   )
        log_prob_M_given_R = log_prob_saturation(tf.expand_dims(im_stack_multiplexed, axis=0), 
                                                 im_stack_multiplexed_dist_0)
        
        log_prob_M = tf.reduce_sum(tf.squeeze(log_prob_M_given_R, axis=0), axis=[1,2,3])\
                     + tf.reduce_sum(log_prob_R_given_z, axis=[1,2,3]) 
 
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
           output_dist_vec, q, q_sample, kl_divergence, -negloglik)



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
                                 reconstruct = False,
                                 N_obj = None,
                                 Ns = None,
                                 P = None,
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
    
    if reconstruct:
        
        # find im_stack from output_sample
        im_stack = []
        for b in range(batch_size):
            obj_re, obj_im = tf.split(output_sample[b],2,axis=-1)
            obj_re = obj_re/normalizer_ang[0] + offset_ang[0]
            obj_im = obj_im/normalizer_ang[1] + offset_ang[1]
            obj = tf.cast(obj_re, tf.complex64) + tf.cast(obj_im, tf.complex64)*1j
            obj = tf.transpose(obj, perm=[2,0,1])
            im_stack_i = create_low_res_stack_multislice(obj, N_obj, Ns, 
                                                         P, Np, LED_vec, 
                                                         LEDs_used_boolean, 
                                                         num_slices, 
                                                         H_scalar, H_scalar_f)
            im_stack.append(tf.expand_dims(im_stack_i, axis=0))
        im_stack = tf.concat(im_stack,axis=0)
        im_stack = tf.cast(im_stack, dtype = tf.float32)
        
        # normalize and offset this im_stack
        im_stack = (im_stack - offset)*normalizer
    else:
        im_stack = output_sample
        
    # shape is max_steps x batch_size x image_x x image_y x num_patterns
    im_stack_multiplexed_dist_0, \
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
                                                 return_dist = True,
                                                 anneal_std=anneal_std,
                                                 quantize_noise = True,
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
                                                     )
            
        return(im_stack_multiplexed_0,im_stack)
    
    return(im_stack_multiplexed_dist_0)


