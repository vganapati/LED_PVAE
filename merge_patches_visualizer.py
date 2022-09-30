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
    plt.show()
    # plt.savefig(subfolder_name + '/full_field_amp.png')
    
    plt.figure()
    plt.title('slice ' + str(ss) + ' angle')
    plt.imshow(np.angle(full_field[ss,visualize_trim:-visualize_trim,
                                    visualize_trim:-visualize_trim]))
    plt.colorbar()
    plt.show()
    # plt.savefig(subfolder_name + '/full_field_angle.png')
    