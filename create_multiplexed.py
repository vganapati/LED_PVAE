#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:26:21 2022

@author: vganapa1
"""

import sys
import numpy as np
import tensorflow as tf
import glob
from helper_pattern_opt import load_img_stack
from helper_functions import configure_for_performance, physical_preprocess
import tensorflow_probability as tfp
from SyntheticMNIST_functions import create_folder, convert_uint_16
import imageio

tfd = tfp.distributions

import argparse

### COMMAND LINE ARGS ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--input_path', action='store', help='dataset to process')

parser.add_argument('--save_tag', action='store', help='output is saved in input_path/save_tag')

parser.add_argument('--save_tag_alpha', action='store', default=None,
                    help='use the alpha saved in input_path/save_tag_alpha')

parser.add_argument('-p', type=int, action='store', dest='num_patterns', \
                help='number of illumination patterns per sample')

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier', 
                    help='poisson noise multiplier, higher value means higher SNR', default = None) #(2**16-1)*0.41
    
parser.add_argument('--dm', type=float, action='store', dest='dirichlet_multiplier', 
                        help='dirichlet_multiplier', default = 1)

parser.add_argument('--uniform', action='store_true', dest='uniform_pattern', 
                    help='only use uniform patterns') 

parser.add_argument('--single', action='store_true', dest='single_pattern', 
                    help='use a single set of patterns for all examples') 

parser.add_argument('--real_data', action='store_true', dest='real_data', 
                    help='uses real data for the image stacks') 

args = parser.parse_args()



### INPUTS ###

input_path = args.input_path #'dataset_foam_singleslice_nosat9_r3' # dataset to process
poisson_noise_multiplier = args.poisson_noise_multiplier # 26869.35
dirichlet_multiplier = args.dirichlet_multiplier #0.1
num_patterns = args.num_patterns
save_tag = args.save_tag #'pnm2e4_dm01_p1' # output is saved in input_path/save_tag
save_tag_alpha = args.save_tag_alpha
single_pattern = args.single_pattern # only used if save_tag_alpha is None
uniform_pattern = args.uniform_pattern # only used if save_tag_alpha is None, takes precendence over single_pattern
real_data = args.real_data # create emulated multiplexed images from real data

exposure_mult = 1 # exposure for the multiplexed images

### LOAD VALUES ###

buffer_size = 10
sqrt_reg = np.finfo(np.float32).eps.item()
batch_size = 1 # must be == 1

if real_data:
    exposure_time_used = np.load(input_path + '/exposure_time_used.npy')  
else:
    normalizer = np.load(input_path + '/normalizer.npy')
    normalizer_ang = np.load(input_path + '/normalizer_ang.npy')
    
    offset = np.load(input_path + '/offset.npy')
    offset_ang = np.load(input_path + '/offset_ang.npy')

data_file_path = input_path + '/training/example_*'
train_folders = sorted(glob.glob(data_file_path))

if real_data:
    num_leds = np.load(input_path + '/num_leds.npy')  
else:
    num_leds = len(glob.glob(train_folders[0] + '/Photo*.png'))


if real_data:
    r_channels = None
else:
    r_channels = len(glob.glob(input_path + '/training/example_000000' + '/reconstruction/Photo*.png'))


### SET AUTOTUNE ###

autotune = tf.data.experimental.AUTOTUNE

### Create the LED patterns (alpha) ###

if save_tag_alpha is None:
    if uniform_pattern: # only use uniform patterns
        all_alpha_train = tf.ones([len(train_folders),num_leds,num_patterns], dtype=tf.float32)/num_leds
    elif single_pattern: # use a single set of patterns for all examples
        all_alpha_train = tfd.Dirichlet(dirichlet_multiplier*np.ones([num_patterns,num_leds])).sample(1)
        all_alpha_train = tf.cast(tf.transpose(all_alpha_train, perm=[0,2,1]), dtype=tf.float32)
        all_alpha_train = tf.repeat(all_alpha_train, len(train_folders),axis=0)
    else:
        all_alpha_train = tfd.Dirichlet(dirichlet_multiplier*np.ones([num_patterns,num_leds])).sample(len(train_folders))
        all_alpha_train = tf.cast(tf.transpose(all_alpha_train, perm=[0,2,1]), dtype=tf.float32)
else:
    all_alpha_train = np.load(input_path + '/' + save_tag_alpha + '/all_alpha_train.npy')
    
create_folder(input_path + '/' + save_tag)
np.save(input_path + '/' + save_tag + '/all_alpha_train.npy', all_alpha_train)

### MAKE TRAINING DATASET ###

train_ds = tf.data.Dataset.from_tensor_slices(train_folders)
alpha_train_ds = tf.data.Dataset.from_tensor_slices(all_alpha_train)
train_ds = tf.data.Dataset.zip((train_ds, alpha_train_ds))

if real_data:
    pass
else:
    load_img_stack2 = lambda folder_name, alpha: load_img_stack(folder_name, num_leds, num_patterns, r_channels, alpha,
                                                                bit_depth = 16,
                                                                )                
                    
                    
    train_ds = train_ds.map(load_img_stack2, num_parallel_calls=autotune)
    train_ds = configure_for_performance(train_ds, 
                                         batch_size,
                                         autotune, shuffle = False, buffer_size = buffer_size, repeat=False)




for object_i in train_ds:
    
    if real_data:
        path, alpha = object_i
        im_stack = np.load(path.numpy().decode('utf-8') + '/im_stack.npy')
        im_stack *= exposure_time_used
        im_stack_expand = np.expand_dims(im_stack, -1)
        alpha_expand = np.expand_dims(np.expand_dims(alpha,1),1)
        # im_stack is now num_leds x image_x x image_y x num_patterns
        im_stack_multiplexed = np.sum(im_stack_expand*alpha_expand, axis=0)/exposure_mult # image_x x image_y x num_patterns
        path = tf.expand_dims(path,0) # add a batch dimension
        
    else:
        path, im_stack, im_stack_r, alpha = object_i
        
        ### Find multiplexed images ###
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
        
        # remove batch dimension
        im_stack_multiplexed = tf.squeeze(im_stack_multiplexed, axis=0)
        
        im_stack_multiplexed = im_stack_multiplexed.numpy()
    
    im_stack_multiplexed_u16 = \
        convert_uint_16(im_stack_multiplexed, 
                        1, #normalizer, 
                        0, #offset, 
                        False, # add_poisson_noise
                        None, #poisson_noise_multiplier
                        )

    for p in range(num_patterns):
        sub_folder_reconstruction_name = '{}/{}/{}'.format(path[0].numpy().decode('UTF-8'), 'multiplexed',save_tag)
        create_folder(sub_folder_reconstruction_name)
        file_name = str('{}/{}_{}{}'.format(sub_folder_reconstruction_name,'mult_image',p,'.png'))
        print(file_name)
        imageio.imwrite(file_name, im_stack_multiplexed_u16[:,:,p])
        
        
        