#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:14:23 2020

@author: vganapa1
"""

import os
import numpy as np
from scipy import signal
import tensorflow as tf
from tensorflow.keras import layers
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_window(image_x, image_y, window_func = signal.windows.hann):
    window_1d_x = np.expand_dims(window_func(image_x),1)
    window_1d_y = np.expand_dims(window_func(image_y),1)
    window_2d = window_1d_x @ window_1d_y.T
    return(window_2d)

def evaluate_patch(base_command, x_corner, y_corner, subfolder_name):
    command_addition = ' --xcorner ' + str(x_corner) + \
                       ' --ycorner '+ str(y_corner)
    
    full_command = base_command + command_addition
    print(full_command)
    os.system(full_command)
    
    # read in patch
    hr_computed = np.load(subfolder_name + '/computed_obj_i.npy')
    # read in window
    window_2d = np.load(subfolder_name + '/window_2d_i.npy')    
    return(hr_computed, window_2d)
    

def merge_patches_func(upsample_factor,
                       x_size,
                       y_size,
                       num_patches_x,
                       num_patches_y,
                       overlap_x,
                       overlap_y,
                       num_slices,
                       start_x_corner,
                       start_y_corner,
                       evaluate_patch_func, 
                       visualize_window,
                       reg,
                       ):
    
    total_length_x = upsample_factor*(x_size*num_patches_x - overlap_x*(num_patches_x-1))
    total_length_y = upsample_factor*(y_size*num_patches_y - overlap_y*(num_patches_y-1))
    
    full_field = np.zeros([num_slices, total_length_x,total_length_y], dtype=np.complex128)
    full_field_window = np.zeros([total_length_x,total_length_y], dtype=np.complex128)
    
    x_corner = start_x_corner
    
    count = -1
    for patch_x in range(num_patches_x):
        y_corner = start_y_corner
        for patch_y in range(num_patches_y):
            count += 1
            
            hr_computed, window_2d = evaluate_patch_func(x_corner, y_corner)
            
            # place in the full reconstruction
            full_field[:,x_corner:x_corner+x_size, y_corner:y_corner+y_size] += hr_computed*visualize_window
            full_field_window[x_corner:x_corner+x_size, y_corner:y_corner+y_size] += visualize_window
    
            # update coordinates
            y_corner = y_corner + y_size - overlap_y
            
        x_corner = x_corner + x_size - overlap_x
    full_field = full_field / (full_field_window+reg)
    return(full_field, full_field_window)

def positive_range_base(x, offset = np.finfo(np.float32).eps.item()):
    x -= 1
    mask = x<0
    return((tf.exp(tf.clip_by_value(x, -1e10, 10))+offset)*tf.cast(mask, tf.float32) + (x+1)*(1-tf.cast(mask, tf.float32)))

def find_I_m_dist(im_stack, # batch x x_dim x y_dim x num_leds
                  alpha_sample, # max_steps x batch x num_leds x num_patterns
                  poisson_noise_multiplier,
                  sqrt_reg,
                  batch_size,
                  max_steps,
                  anneal_std = 0,
                  quantize_noise = False,
                  quantize_bit_depth = 16,
                  dtype=tf.float64,
                  ):

    # max_steps is 0th dimension of alpha_sample 
    
    multiplexed_mean_all = []
    for j in range(max_steps):
        multiplexed_mean = []
        for i in range(batch_size):
            im_stack_i = im_stack[i]
            alpha_i = alpha_sample[j,i]
            # process im_stack_i by alpha_i and exposure_i
            multiplexed_mean_i = tf.linalg.matmul(im_stack_i,alpha_i)
            multiplexed_mean_i = tf.expand_dims(multiplexed_mean_i,axis=0)
            multiplexed_mean.append(multiplexed_mean_i)
        multiplexed_mean_all.append(tf.expand_dims(tf.concat(multiplexed_mean, axis=0),axis=0))
    multiplexed_mean_all = tf.concat(multiplexed_mean_all, axis=0)
    
    # add Poisson-like noise
    if quantize_noise:
        '''
        print('LINE 58')
        print(multiplexed_mean_all.dtype)
        print(anneal_std.dtype)
        print(poisson_noise_multiplier.dtype)
        '''
        im_stack_multiplexed_dist = tfd.Normal(loc = multiplexed_mean_all, \
                                               scale = tf.cast(anneal_std, dtype) + sqrt_reg\
                                                   + tf.sqrt(sqrt_reg+multiplexed_mean_all/poisson_noise_multiplier+(1/(2**quantize_bit_depth-1)**2)/12))
    else:
        im_stack_multiplexed_dist = tfd.Normal(loc = multiplexed_mean_all, \
                                               scale = anneal_std + sqrt_reg + tf.sqrt(sqrt_reg+multiplexed_mean_all/poisson_noise_multiplier))
    
    return(im_stack_multiplexed_dist)

def saturation_masks(im_stack_multiplexed):
    indices = tf.where(im_stack_multiplexed<=0)
    len_indices = indices.shape[0]
    if len_indices:
        mask_below_zero = tf.sparse.to_dense(tf.sparse.SparseTensor(indices, \
                                                                     tf.ones([len_indices]), 
                                                                    im_stack_multiplexed.shape))
    else:
        mask_below_zero = tf.zeros_like(im_stack_multiplexed)
        
    indices = tf.where(im_stack_multiplexed>=1)
    len_indices = indices.shape[0]
    if len_indices:
        mask_above_one = tf.sparse.to_dense(tf.sparse.SparseTensor(indices, \
                                                                    tf.ones([len_indices]), 
                                                                    im_stack_multiplexed.shape))
    else:
        mask_above_one = tf.zeros_like(im_stack_multiplexed)
        
    return(mask_below_zero, mask_above_one)

def log_prob_saturation(im_stack_multiplexed, im_stack_multiplexed_dist, dtype = tf.float64):
    im_stack_multiplexed = tf.cast(im_stack_multiplexed, dtype)
    mask_below_zero, mask_above_one = saturation_masks(im_stack_multiplexed)
    # bound between some factor of standard deviations
    im_stack_multiplexed_clipped = tf.clip_by_value(im_stack_multiplexed,
                                                    im_stack_multiplexed_dist.scale*(-1e3),
                                                    im_stack_multiplexed_dist.scale*(1e3))
    nominal_probs = im_stack_multiplexed_dist.log_prob(im_stack_multiplexed_clipped)
    probs_below_zero = im_stack_multiplexed_dist.log_cdf(tf.zeros_like(im_stack_multiplexed))
    probs_above_one = tf.math.log1p(-im_stack_multiplexed_dist.cdf(tf.ones_like(im_stack_multiplexed)))
    
    all_probs = nominal_probs*(1-mask_below_zero-mask_above_one) + probs_below_zero*mask_below_zero + \
        probs_above_one*mask_above_one
    return(all_probs)


def log_prob_sat_quantize(im_stack_multiplexed, im_stack_multiplexed_dist, 
                          sqrt_reg = np.finfo(np.float32).eps.item(), bit_depth = 16):
    mask_below_zero, mask_above_one = saturation_masks(im_stack_multiplexed)
    # bound between some factor of standard deviations
    im_stack_multiplexed_clipped = tf.clip_by_value(im_stack_multiplexed,im_stack_multiplexed_dist.scale*(-1e3),im_stack_multiplexed_dist.scale*(1e3))
    # nominal_probs = im_stack_multiplexed_dist.log_prob(im_stack_multiplexed_clipped)
    probs_below_zero = im_stack_multiplexed_dist.log_cdf(tf.zeros_like(im_stack_multiplexed))
    probs_above_one = tf.math.log1p(sqrt_reg-im_stack_multiplexed_dist.cdf(tf.ones_like(im_stack_multiplexed)))
    
    probs_below_values = im_stack_multiplexed_dist.log_cdf(im_stack_multiplexed - 1/(2**bit_depth-1)/2)
    probs_above_values = im_stack_multiplexed_dist.log_cdf(im_stack_multiplexed + 1/(2**bit_depth-1)/2)

    nominal_probs = -probs_above_values + tf.math.log1p(sqrt_reg-tf.math.exp(probs_below_values-probs_above_values))
    
    all_probs = nominal_probs*(1-mask_below_zero-mask_above_one) + probs_below_zero*mask_below_zero + \
        probs_above_one*mask_above_one
    return(all_probs)

def physical_preprocess(im_stack, 
                        alpha_sample,
                        poisson_noise_multiplier,
                        sqrt_reg,
                        batch_size,
                        max_steps,
                        renorm, # True if need to remove normalization and offset
                        normalizer=None,
                        offset=None,
                        zero_alpha=False,
                        return_dist = False,
                        anneal_std=0,
                        set_seed=False,
                        quantize_noise=False,
                        dtype=tf.float64,
                        ):

    if renorm:
        # remove offset from im_stack, keep normalizer
        im_stack = (im_stack/normalizer + offset)*normalizer
    
    im_stack_multiplexed_dist = find_I_m_dist(im_stack,
                                              alpha_sample,
                                              poisson_noise_multiplier,
                                              sqrt_reg,
                                              batch_size,
                                              max_steps,
                                              anneal_std = anneal_std,
                                              quantize_noise = quantize_noise,
                                              dtype=dtype)

    if set_seed:
        tf.random.set_seed(1234)
        im_stack_multiplexed = im_stack_multiplexed_dist.sample(seed=10)
    else:
        im_stack_multiplexed = im_stack_multiplexed_dist.sample()

  


    if zero_alpha:
        im_stack_multiplexed = tf.zeros_like(im_stack_multiplexed)
    
    if return_dist:
        return(im_stack_multiplexed_dist, im_stack_multiplexed) 
    else:
        # apply image saturation
        im_stack_multiplexed = tf.maximum(im_stack_multiplexed,0)
        im_stack_multiplexed = tf.minimum(im_stack_multiplexed,1)  
        # im_stack_multiplexed = tfp.math.smootherstep(im_stack_multiplexed)
        
        return(im_stack_multiplexed)

def create_prob_dist(alpha_loc, alpha_log_scale, sqrt_reg, dirichlet, fixed, offset_dirichlet, deterministic):
    

    if dirichlet:
        if fixed:
            a_T = tf.transpose(alpha_loc)
        else:
            a_T = tf.transpose(alpha_loc, perm=[0,2,1])
        if deterministic:    
            a_T = tf.exp(a_T)/tf.expand_dims((tf.reduce_sum(tf.exp(a_T), axis=-1)+sqrt_reg),axis=-1)
            action_dist = tfd.Deterministic(a_T)
        else:
            action_dist = tfd.Dirichlet(sqrt_reg + offset_dirichlet + tf.exp(tf.clip_by_value(a_T, -1e10, 10)))
    else:
        # action_dist =  tfp.distributions.Normal(alpha_loc, .1, \
        #                                         validate_args=False, allow_nan_stats=True, name='Normal')      
        
        action_dist =  tfp.distributions.Normal(alpha_loc, sqrt_reg + tf.exp(tf.clip_by_value(alpha_log_scale, -1e10, 10)), \
                                                validate_args=False, allow_nan_stats=True, name='Normal')
    

    return action_dist

def configure_for_performance(ds, batch_size, 
                              autotune, 
                              shuffle = True,
                              buffer_size = 100,
                              repeat = True):
      
#    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size)        

    if repeat:
        ds = ds.repeat()
        
    ds = ds.prefetch(buffer_size=autotune)
    return ds

def configure_for_performance_real_data(ds, batch_size, 
                              autotune, 
                              shuffle = True,
                              buffer_size = 100,
                              repeat = True):
      
#    ds = ds.cache()

    if repeat:
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size)        

    ds = ds.prefetch(buffer_size=autotune)
    return ds



def trim_lit_coord(LitCoord):
    row_min = np.min(np.nonzero(LitCoord)[0])
    row_max = np.max(np.nonzero(LitCoord)[0])
    
    col_min = np.min(np.nonzero(LitCoord)[1])
    col_max = np.max(np.nonzero(LitCoord)[1])

    LitCoord = LitCoord[row_min:row_max+1, col_min:col_max+1]
    return LitCoord


def create_alpha_mat(alpha, 
                     batch_size,
                     num_leds,
                     num_patterns,
                     LitCoord2,
                     ):

    alpha_mat_tot = []
    for j in range(batch_size):
        alpha_mat = []
        for i in range(num_patterns):
            alpha_mat_i = tf.sparse.SparseTensor(tf.where(LitCoord2==1), \
                                                 tf.squeeze(tf.slice(alpha,[j,0,i],[1,num_leds,1])), \
                                                 LitCoord2.shape)
            alpha_mat_i = tf.sparse.to_dense(alpha_mat_i)
            alpha_mat_i = tf.expand_dims(alpha_mat_i,axis=-1)
            alpha_mat.append(alpha_mat_i)

        alpha_mat = tf.concat((alpha_mat), axis=-1)
        alpha_mat_tot.append(tf.expand_dims(alpha_mat,axis=0))
        
    alpha_mat_tot = tf.concat(alpha_mat_tot,axis=0)
    
    return alpha_mat_tot


def reformat_alpha(alpha_mat, 
                   batch_size,
                   num_leds,
                   num_patterns,
                   LitCoord2,
                   num_leds_length,
                   ):
    alpha = [] # batch_size, num_leds, num_patterns
    alpha_coords = tf.where(LitCoord2==1)
    for j in range(batch_size):
        alpha_batch_i = []
        for i in range(num_patterns):
            # alpha_mat_i = alpha_mat[j,:,:,i]
            alpha_mat_i = tf.squeeze(tf.squeeze(tf.slice(alpha_mat,[j,0,0,i],[1,num_leds_length,num_leds_length,1]),axis=0), axis=-1)
            alpha_pattern_0 = tf.zeros([num_leds])
            alpha_pattern = tf.expand_dims(alpha_pattern_0 + tf.gather_nd(alpha_mat_i,alpha_coords), axis = -1)
            alpha_batch_i.append(alpha_pattern)
        alpha.append(tf.expand_dims(tf.concat(alpha_batch_i,axis=1), axis=0))
    alpha = tf.concat(alpha, axis=0)
    return(alpha)
                

class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, initializer = 'glorot_uniform'):
      super(InstanceNormalization, self).__init__()
      self.epsilon = epsilon
      self.initializer = initializer

    def build(self, input_shape):
      self.scale = self.add_weight(
          name='scale',
          shape=input_shape[-1:],
          initializer=self.initializer,
          trainable=True)

      self.offset = self.add_weight(
          name='offset',
          shape=input_shape[-1:],
          initializer='zeros',
          trainable=True)

    def call(self, x):
      mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
      inv = tf.math.rsqrt(variance + self.epsilon)
      normalized = (x - mean) * inv
      return self.scale * normalized + self.offset
      

def average_skips(skips, num_skips, num_patterns, num_patterns_adjusted):
    ave_skips = []
    for ii in range(num_skips):        
        ave_skips_ii = [tf.expand_dims(skips[i][ii],axis=0) for i in range(num_patterns)]

        ave_skips_ii = tf.math.reduce_sum(tf.concat(ave_skips_ii, axis=0), axis=0)
        ave_skips.append(ave_skips_ii)  
    
    ave_skips = [ave_skips[i]/float(num_patterns_adjusted) for i in range(num_skips)]
    return ave_skips


def weighted_ave(num_blocks,
                 num_patterns,
                 skip_val_vec,
                 skip_weight_vec,
                 sqrt_reg,
                 ):
    
    total_skip_vec = []
    total_skip_weight_vec = []
    
    for ii in range(num_blocks):    

        for jj in range(num_patterns):
            skip_weight = positive_range_base(skip_weight_vec[jj][ii],)
            # skip_weight = tf.clip_by_value(skip_weight_vec[jj][ii], 0, 1e10) # make weight nonnegative
            # skip_weight = skip_weight_vec[jj][ii]
            
            if jj==0:
                total_skip = skip_val_vec[jj][ii]*skip_weight
                total_skip_weight = skip_weight
            else:
                total_skip += skip_val_vec[jj][ii]*skip_weight
                total_skip_weight += skip_weight
                
        total_skip_vec.append(total_skip)
        total_skip_weight_vec.append(total_skip_weight)
        
    ave_skip_vec = [total_skip_vec[i]/(total_skip_weight_vec[i]+sqrt_reg) for i in range(num_blocks)]   

    return ave_skip_vec, total_skip_weight_vec




def get_ave_output(num_blocks,
                   num_patterns,
                   num_patterns_adjusted, # zeros out patterns more than num_patterns_adjusted
                   im_stack_multiplexed,
                   alpha_sample,
                   batch_size,
                   image_x,
                   image_y,
                   num_leds,
                   model_encode,
                   sqrt_reg,
                   training,
                   real_data=False,
                   img_coords_xm=None, 
                   img_coords_ym=None,
                   ):
    
    skip_val_vec = []
    skip_weight_vec = []
    
    for t in range(0, num_patterns):
        zero_multiplier = tf.cond(tf.less(t-1,(num_patterns_adjusted-1)), lambda: 1.0, lambda: 0.0)
        
        im_stack_multiplexed_i = tf.slice(im_stack_multiplexed,[0,0,0,t],[batch_size,image_x,image_y,1])
    
        alpha_sample_i = tf.slice(alpha_sample,[0,0,t],[batch_size,num_leds,1])
        alpha_sample_i = tf.squeeze(alpha_sample_i,axis=-1)
        
        if real_data:
            model_inputs = (im_stack_multiplexed_i, alpha_sample_i, img_coords_xm, img_coords_ym)
        else:
            model_inputs = (im_stack_multiplexed_i, alpha_sample_i)
            
        skips_val, skips_weight = model_encode(model_inputs, training=training)
       
        skips_weight = [skip*zero_multiplier for skip in skips_weight] 
      
        skip_val_vec.append(skips_val)
        skip_weight_vec.append(skips_weight)
        


    ave_skip_vec, total_skip_weight_vec = \
    weighted_ave(num_blocks,
                 num_patterns,
                 skip_val_vec,
                 skip_weight_vec,
                 sqrt_reg,
                 )
    
    return ave_skip_vec, total_skip_weight_vec

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