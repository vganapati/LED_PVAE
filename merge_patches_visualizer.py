#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:23:31 2022

@author: vganapa1
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from helper_functions import compare_low_res, create_window
import argparse
from visualizer_functions import show_alpha_scatter

### COMMAND LINE ARGS ###


parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--iterative', action='store_true', dest='iterative', 
                    help='analyze solution from iterative optimization')

parser.add_argument('--verbose', action='store_true', dest='verbose', 
                    help='print low res patch figures')

parser.add_argument('--input_path', action='store', help='path(s) to overall folder containing training data',
                    default = 'dataset_frog_blood_v3')

parser.add_argument('--save_tag_recons', action='store', help='only for iterative option, path for results is /reconstruction_save_name', \
                    default = '_merge_patch_test')

parser.add_argument('--save_path', action='store', help='path where neural network output is saved', default = 'refactor_test')
    
parser.add_argument('--obj', type=int, action='store', dest='obj_ind', \
                        help='obj number to reconstruct', default = 0)
    
parser.add_argument('--xcrop', type=int, action='store', dest='x_crop_size', \
                    help='patch size to consider in reconstruction', default = 512)
        
parser.add_argument('--ycrop', type=int, action='store', dest='y_crop_size', \
                    help='patch size to consider in reconstruction', default = 512)    

parser.add_argument('--md', dest = 'multiplexed_description',
                    action='store', help='description of multiplex type', default = '') # _Dirichlet or _Random

parser.add_argument('--use_window', action='store_true', dest='use_window', 
                    help='uses a windowing function for real data') 

parser.add_argument('--da', action='store_true', dest='download_all', 
                    help='downloads all patches, otherwise just downloads the full field reconstruction') 


args = parser.parse_args()


### END COMMAND LINE ARGS ###

### INPUTS ###
input_path = args.input_path
obj_ind = args.obj_ind
iterative = args.iterative
verbose = args.verbose
multiplexed_description = args.multiplexed_description

dataset_type = 'training'
visualize_trim = 64

x_crop_size = args.x_crop_size
y_crop_size = args.y_crop_size
    
download_all = args.download_all

if iterative:
    use_window = False
    exposure = 1
else: # for neural network output
    if multiplexed_description == '_Random':
        exposure = 50
    elif multiplexed_description == '_Dirichlet':
        exposure = 100
    
    use_window = args.use_window 


if use_window:
    window_2d = create_window(x_crop_size, y_crop_size)
else:
    window_2d = np.ones([x_crop_size, y_crop_size])
    



folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)

if iterative:
    # Iterative solution
    save_tag_recons = args.save_tag_recons
    subfolder_name = folder_name + '/reconstruction' + save_tag_recons
    filepath = subfolder_name + '/full_field.npy'
else:
    # Neural network solution
    save_path = args.save_path
    filepath = save_path + '/full_field_example_' + str(obj_ind) + '.npy'
    subfolder_name = folder_name + '/reconstruction_' + save_path

### END OF INPUTS ###

num_leds = int(np.load(input_path + '/num_leds.npy'))
full_field = np.load(filepath)
num_slices = full_field.shape[0]


for ss in range(num_slices):
    plt.figure()
    plt.title('slice ' + str(ss) + ' amplitude')
    plt.imshow(np.abs(full_field[ss,visualize_trim:-visualize_trim,
                                  visualize_trim:-visualize_trim]))
    plt.colorbar()
    # plt.savefig(subfolder_name + '/full_field_amp.png')
    
    plt.figure()
    plt.title('slice ' + str(ss) + ' angle')
    plt.imshow(np.angle(full_field[ss,visualize_trim:-visualize_trim,
                                    visualize_trim:-visualize_trim]))
    plt.colorbar()
    # plt.savefig(subfolder_name + '/full_field_angle.png')
    
# calculate merit, only if download_all is passed

if download_all:
    all_patch_folders = glob.glob(subfolder_name + '/x_corner_*_y_corner_*')
    
    mse_vec = []
    ssim_vec = []
    psnr_vec = []
    
    for folder in all_patch_folders:
        lr_calculated = np.load(folder + '/lr_calc_stack_final.npy')/exposure # num_leds x x_size x y_size
        
        if iterative and multiplexed_description == '_Dirichlet':
            lr_calculated = lr_calculated/2
        
        lr_observed = window_2d*np.load(folder + '/lr_observed_stack.npy')/exposure # num_leds x x_size x y_size
        
        if verbose:
            led_ind = num_leds//2
            plt.figure()
            plt.title('lr_calculated')
            plt.imshow(lr_calculated[led_ind,:,:])
            plt.colorbar()
            
            plt.figure()
            plt.title('lr_observed')
            plt.imshow(lr_observed[led_ind,:,:])
            plt.colorbar()
    
        mse_vec_patch_i = []
        ssim_vec_patch_i = []
        psnr_vec_patch_i = []
        
        for led in range(num_leds):
            mse, ssim, psnr = compare_low_res(lr_calculated[led,:,:], lr_observed[led,:,:])
            mse_vec_patch_i.append(mse)
            ssim_vec_patch_i.append(ssim)
            psnr_vec_patch_i.append(psnr)
    
        mse_vec_patch_i = np.stack(mse_vec_patch_i)
        ssim_vec_patch_i = np.stack(ssim_vec_patch_i)
        psnr_vec_patch_i = np.stack(psnr_vec_patch_i)   
        
        mse_vec.append(mse_vec_patch_i)
        ssim_vec.append(ssim_vec_patch_i)
        psnr_vec.append(psnr_vec_patch_i)
        
    mse_vec = np.stack(mse_vec) # patches x leds
    ssim_vec = np.stack(ssim_vec) # patches x leds
    psnr_vec = np.stack(psnr_vec) # patches x leds
    
    print('average MSE is: ', np.mean(mse_vec))
    print('average SSIM is: ', np.mean(ssim_vec))
    print('average PSNR is: ', np.mean(psnr_vec))
    
    led_position_xy = np.load(input_path + '/led_position_xy.npy')
    mse_per_led = np.mean(mse_vec,axis=0)
    show_alpha_scatter(led_position_xy, mse_per_led, 
                       None)