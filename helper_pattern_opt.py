#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:12:01 2021

@author: vganapa1
"""

import sys
import tensorflow as tf
import tensorflow_probability as tfp
from helper_functions import physical_preprocess, \
                             get_ave_output, positive_range_base
import numpy as np
from SyntheticMNIST_multislice_functions import find_Ns

tfd = tfp.distributions


def bound_mat_tf(mat, lb = 0+np.finfo(np.float32).eps.item(), ub = 1-np.finfo(np.float32).eps.item()):
    return(tf.clip_by_value(mat, lb, ub))

def clamp(x):
    mask_0=x<0
    mask_1=x>1
    return(x*(1-mask_0-mask_1) +1*(mask_1))

def load_multiplexed(num_patterns,
                     image_path,
                     save_tag_multiplexed,
                     bit_depth,
                     exposure=1,
                     real_mult=False,
                     dtype=tf.float32,
                     multiplexed_description='',
                     background_subtraction = False):
    im_stack_multiplexed = []

    for p in range(num_patterns):
        if real_mult:
            file_path = image_path + '/Multiplex' + multiplexed_description + '/{0:04}.png'.format(p)
        else:
            file_path = image_path + '/' + 'multiplexed' + '/' +\
                save_tag_multiplexed + '/' + 'mult_image_' + str(p) + '.png'
        im = tf.io.read_file(file_path)
        im = tf.io.decode_image(
                im, channels=0, dtype=tf.dtypes.uint16)
        
        if background_subtraction:
            file_path = image_path + '/Multiplex' + multiplexed_description + '/Reference.png'
            black_img = tf.io.read_file(file_path)
            black_img = tf.io.decode_image(
                    black_img, channels=0, dtype=tf.dtypes.uint16)
            Ibk_0 = tf.reduce_mean(black_img)
            im = tf.maximum(im-Ibk_0, 0)
            
        if real_mult:
            # rotate
            im = tf.image.rot90(im,3)
            
        im = tf.cast(im, dtype)
        im = im/float(2**bit_depth-1)/exposure
        im_stack_multiplexed.append(im)

    im_stack_multiplexed = tf.concat(im_stack_multiplexed,axis=-1)  
    

    return(im_stack_multiplexed)
    
def load_img_stack(image_path, num_leds, num_patterns, r_channels, 
                   alpha, # pattern chosen ahead of time
                   bit_depth, 
                   save_tag_multiplexed=None,
                   ):
        
    ### Load in a stack of images ###
    
    im_stack = []
    
    for ind in range(num_leds):
        # num_str = str(ind)
        # file_path = image_path + '/Photo' + '0'*(4-len(num_str)) + num_str + '.png'
        file_path = image_path + '/Photo{:04d}.png'.format(ind) 
        im = tf.io.read_file(file_path)
        im = tf.io.decode_image(
                im, channels=0, dtype=tf.dtypes.uint16)
        
        im = tf.cast(im, tf.float32)
        im = im/float(2**bit_depth-1)
        im_stack.append(im)

    im_stack = tf.concat(im_stack,axis=-1)
 
    if save_tag_multiplexed is not None:
        im_stack_multiplexed = \
        load_multiplexed(num_patterns,
                             image_path,
                             save_tag_multiplexed,
                             bit_depth,
                             real_mult=False,
                             dtype=tf.float32)

    im_stack_r = []
    
    for ind in range(r_channels):
        num_str = str(ind)
        file_path = image_path + '/reconstruction/Photo' + '0'*(4-len(num_str)) + num_str + '.png'
        im = tf.io.read_file(file_path)
        im = tf.io.decode_image(
                im, channels=0, dtype=tf.dtypes.uint16)
        
        im = tf.cast(im, tf.float32)
        im = im/float(2**bit_depth-1)
        im_stack_r.append(im)

    im_stack_r = tf.concat(im_stack_r,axis=-1)        
    

    if save_tag_multiplexed is not None:
        return (image_path, im_stack, im_stack_r, alpha, im_stack_multiplexed)
    else:
        return (image_path, im_stack, im_stack_r, alpha)


def load_img_stack_real_data(image_path, num_patterns, 
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
                             bit_depth,
                             force_x_corner=None,
                             force_y_corner=None,
                             real_mult=False,
                             multiplexed_description='',
                             exposure=1,
                             ):
    
    # randomly choose a patch
    if force_x_corner is not None:
        x_corner = force_x_corner
    else:
        x_corner = tf.random.uniform([1],minval=0, maxval=full_image_x-image_x, dtype=tf.int32)[0]
    
    if force_y_corner is not None:
        y_corner = force_y_corner
    else:
        y_corner = tf.random.uniform([1],minval=0, maxval=full_image_y-image_y, dtype=tf.int32)[0]
    
    im_stack_multiplexed = \
    load_multiplexed(num_patterns,
                     image_path,
                     save_tag_multiplexed,
                     bit_depth,
                     real_mult=real_mult,
                     dtype=tf.float32,
                     multiplexed_description = multiplexed_description,
                     exposure=exposure,
                     )


    im_stack_multiplexed = im_stack_multiplexed[x_corner:x_corner+image_x,
                                                y_corner:y_corner+image_y,
                                                :]
    # im_stack_multiplexed = tf.slice(im_stack_multiplexed, [x_corner,y_corner,0],[image_x, image_y, num_patterns])

    
    # patch coords
    # img_coords_xm = img_coords_xm[x_corner:x_corner + image_x, y_corner:y_corner + image_y]
    # img_coords_ym = img_coords_ym[x_corner:x_corner + image_x, y_corner:y_corner + image_y]
    img_coords_xm = tf.slice(img_coords_xm, [x_corner,y_corner],[image_x, image_y])
    img_coords_ym = tf.slice(img_coords_ym, [x_corner,y_corner],[image_x, image_y])
    
    
    
    Ns_0, synthetic_NA, cos_theta_0 = find_Ns(img_coords_xm,
                                              img_coords_ym,
                                              image_x,
                                              image_y,
                                              led_position_xy,
                                              dpix_m,
                                              z_led,
                                              wavelength,
                                              NA,
                                              du,
                                              um_m)  
        
    return(image_path, alpha, im_stack_multiplexed, img_coords_xm, img_coords_ym, Ns_0, synthetic_NA, cos_theta_0)



def find_loss(im_stack,
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
              time_fraction_0,
              total_exposure_time,
              first_step_patterns,
              training,
              ):

    
    alpha_vec, time_fraction_vec, im_stack_multiplexed_vec, state_pi_vec, exposure_time_vec = \
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
                time_fraction_0,
                total_exposure_time,
                first_step_patterns,
                training,
                )
    
    loss, all_loss = calculate_loss(im_stack, 
                                    alpha_vec, 
                                    time_fraction_vec,
                                    exposure_time_vec,
                                    batch_size,
                                    poisson_noise_multiplier,
                                    sqrt_reg,
                                    image_x,
                                    image_y,
                                    max_steps,
                                    patch_x = 16,
                                    patch_y = 16,
                                    training=training,
                                    )
            
    return loss, alpha_vec, time_fraction_vec, im_stack_multiplexed_vec, state_pi_vec, exposure_time_vec
    
# XXX This function is incorrect.
# Each example should have an im_stack_multiplexed_dist that is recalculated with the pertainent alpha
def calculate_loss(im_stack, 
                   alpha_vec, 
                   time_fraction_vec,
                   exposure_time_vec,
                   batch_size,
                   poisson_noise_multiplier,
                   sqrt_reg,
                   image_x,
                   image_y,
                   max_steps,
                   patch_x,
                   patch_y, 
                   training,
                   ):   



    alpha_all = process_alpha(alpha_vec, sqrt_reg)*tf.expand_dims(process_time(time_fraction_vec), axis=2)*tf.expand_dims(exposure_time_vec, axis=2)
    im_stack_multiplexed_dist, \
    im_stack_multiplexed = physical_preprocess(im_stack, 
                                               alpha_all,
                                               poisson_noise_multiplier,
                                               sqrt_reg,
                                               batch_size,
                                               max_steps,
                                               zero_alpha = False,
                                               return_dist = True,
                                               )
    
    '''
    # check log probabilities when sampled from distribution   
    tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(im_stack_multiplexed_dist.log_prob(im_stack_multiplexed_dist.sample()), axis = -1), axis=-1), axis=-1), axis=0)

    # check log probabilities when using the truncated values
    tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(im_stack_multiplexed_dist.log_prob(im_stack_multiplexed), axis = -1), axis=-1), axis=-1), axis=0)
    
    For valid performance, above 2 should be similar
    '''

    # likelihoods
    # See https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorToeplitz for construction of Toeplitz matrix
    
    '''
    A = |a b c d|
        |e a b c|
        |f e a b|
        |g f e a|
        
    col is [a e f g]
    row is [a b c d]
    
    In this case if we have 4 examples [a b c d], we want:

    A = |a b c d|
        |d a b c|
        |c d a b|
        |b c d a|        
    
    Thus:
    
    col is [a d c b]
    row is [a b c d]
    '''
    
    col = tf.concat((tf.expand_dims(im_stack_multiplexed[:,0],axis=1),tf.reverse(im_stack_multiplexed,[-1])[:,:-1]),axis=1)
    col = tf.transpose(col, perm=[0,2,3,4,1]) # put batch dimension last
    
    row = im_stack_multiplexed
    row = tf.transpose(row, perm=[0,2,3,4,1]) # put batch dimension last
    
    im_stack_multiplexed_sample_mat = tf.linalg.LinearOperatorToeplitz(
        col, row, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None,
        is_square=None, name='LinearOperatorToeplitz'
        )

    im_stack_multiplexed_sample_mat_1 = tf.transpose(im_stack_multiplexed_sample_mat.to_dense(), perm=[4,0,5,1,2,3])


    # all probs is batch_size x max_steps x batch_size x image_x x image_y x num_leds
    # batch_size x batch_size is in the form of A shown above
    all_probs = im_stack_multiplexed_dist.log_prob(im_stack_multiplexed_sample_mat_1)  # likelihood of each sample
    

    '''
    Check that the first row of this matrix matches the log probabilities of im_stack_multiplexed
    tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(all_probs,axis=-1), axis=-1), axis=-1), axis=1)
    '''
    
    # zero pad to make xyz dimensions divisable by patch size
    remain_x = patch_x - image_x % patch_x
    remain_y = patch_y - image_y % patch_y
    
    all_probs_mat = tf.reduce_sum(all_probs, axis=1) # sum over max_steps dimension
    all_probs_mat = tf.reduce_sum(all_probs_mat, axis=-1)  # sum over num_patterns_dimension
    
    # pad with 0s (since we are manipulating log probabilities)
    all_probs_mat = tf.pad(all_probs_mat, [[0,0],[0,0], [0,remain_x],[0,remain_y]],  mode='CONSTANT', constant_values=0)
    
    image_x_1 = image_x + remain_x
    image_y_1 = image_y + remain_y
    
    all_probs_mat = tf.reshape(all_probs_mat, [batch_size, batch_size,image_x_1//patch_x,patch_x,image_y_1])
    all_probs_mat = tf.reshape(all_probs_mat, [batch_size, batch_size, image_x_1//patch_x,patch_x,image_y_1//patch_y,patch_y])
    
    all_probs_mat = tf.reduce_sum(tf.reduce_sum(all_probs_mat,axis=-1),axis=-2)

    all_probs_mat_diff = all_probs_mat[1:]-all_probs_mat[0]
    # tf.print(all_probs_mat_diff)
    all_loss = -tf.math.log(1/(batch_size)) - tf.math.log1p(tf.reduce_sum(tf.exp(tf.clip_by_value(all_probs_mat_diff, -1e10, 88)),axis=0)) # Equation 9 of foster_2021_deep

    
    L = tf.reduce_mean(all_loss) # average over all examples
    
    loss = -L # want to maximize L
    return(loss, -all_loss)


def run_episode(im_stack,
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
                ):
    

    alpha_vec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    im_stack_multiplexed_vec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state_pi_vec = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    
    # initial state obtained with alpha_0
    
    alpha_i = tf.repeat(tf.expand_dims(alpha_0, axis=0), batch_size, axis=0)
    state_pi_input = tf.zeros([batch_size, skips_pixel_x_pi[skip_connect_ind], skips_pixel_y_pi[skip_connect_ind], skips_pixel_z_pi[skip_connect_ind]//2], tf.float32)
    prior_pi_weight = tf.zeros([batch_size, skips_pixel_x_pi[skip_connect_ind], skips_pixel_y_pi[skip_connect_ind], skips_pixel_z_pi[skip_connect_ind]//2], tf.float32)
        

    for t in tf.range(0,max_steps):
        # tf.print(t)
        # sys.exit()
        
        alpha_vec = alpha_vec.write(t, alpha_i)
        state_pi_vec = state_pi_vec.write(t, state_pi_input)
        
        # adjust number of patterns if t == 0 to first_step_patterns
        num_patterns_adjusted = tf.cond(t==0, true_fn=lambda: first_step_patterns, 
                                        false_fn=lambda: num_patterns)
        
        alpha_scaled = process_alpha(alpha_i, sqrt_reg)
        im_stack_multiplexed = physical_preprocess(im_stack, 
                                                   tf.expand_dims(alpha_scaled, axis = 0),
                                                   poisson_noise_multiplier,
                                                   sqrt_reg,
                                                   batch_size,
                                                   1, # max_steps
                                                   renorm = True,
                                                   normalizer = normalizer,
                                                   offset = offset,
                                                   zero_alpha = False
                                                   )
        im_stack_multiplexed = tf.squeeze(im_stack_multiplexed, axis = 0)

        im_stack_multiplexed_vec = im_stack_multiplexed_vec.write(t, im_stack_multiplexed)

        current_skips_pi, current_skip_weight_pi \
         = get_ave_output(num_blocks+1,
                          num_patterns,
                          num_patterns_adjusted,
                          im_stack_multiplexed,
                          alpha_scaled,
                          batch_size,
                          image_x,
                          image_y,
                          num_leds,
                          model_encode_pi,
                          sqrt_reg,
                          training,
                          )

        total_pi_weight = current_skip_weight_pi[skip_connect_ind] + prior_pi_weight
         
        state_pi_input = (state_pi_input*prior_pi_weight + current_skips_pi[skip_connect_ind]*current_skip_weight_pi[skip_connect_ind])/(total_pi_weight + sqrt_reg)

        prior_pi_weight = total_pi_weight # update prior pi weight



        if pi_iter:
            alpha_i = model_pi((state_pi_input,
                                tf.repeat(tf.expand_dims(t, axis=0),batch_size, axis=0)), \
                               training = training)
        else:
            alpha_i = model_pi((state_pi_input), \
                               training = training)
        


    alpha_vec = alpha_vec.stack()
    im_stack_multiplexed_vec = im_stack_multiplexed_vec.stack()
    state_pi_vec = state_pi_vec.stack()
           
    return(alpha_vec, im_stack_multiplexed_vec, \
           state_pi_vec)
        
def process_alpha(alpha, reg):
    # input can be alpha_vec: max_steps x batch_size x num_leds x num_patterns
    
    # input can be alpha_i: batch_size x num_leds x num_patterns
    
    # return(tf.sigmoid(alpha))
    

    alpha = positive_range_base(alpha)
    
    alpha_sum = tf.expand_dims(tf.reduce_sum(alpha, axis = -2), axis = -2)
    
    # normalize alpha
    
    alpha = alpha / (alpha_sum + reg)
    return(alpha)


def process_time(time):
    # input can be time_fraction_vec: max_steps x batch_size x num_patterns
        
    # input can be time_fraction_i: batch_size x num_patterns
    
    return(tf.sigmoid(time))