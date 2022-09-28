#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:40:08 2021

@author: vganapa1
"""

import numpy as np
import time
from scipy import signal
from helper_functions import create_window, evaluate_patch, merge_patches_func
import argparse

### COMMAND LINE ARGS ###

parser = argparse.ArgumentParser(description='Get command line args')


parser.add_argument('--input_path', action='store', help='path(s) to overall folder containing training data',
                    default = 'dataset_frog_blood_v3')

parser.add_argument('--save_tag_recons', action='store', help='reconstruction_save_name', \
                    default = '_multiplexed_p1')
    
parser.add_argument('-p', type=int, 
                    action='store', dest='num_patterns',
                    help='num_patterns when reconstructing with multiplexed', 
                    default = 1)

parser.add_argument('-b', type=int, 
                    action='store', dest='batch_size',
                    help='batch_size for reconstruction with all single LEDs', 
                    default = 85)

parser.add_argument('--alr', type=float, action='store', dest='adam_learning_rate', 
                        help='learning rate for adam optimizer', default = 1e-2)
    
parser.add_argument('--md', dest = 'multiplexed_description',
                    action='store', help='description of multiplex type', default = '_Dirichlet') # _Dirichlet or _Random

parser.add_argument('--use_mult', action='store_true', dest='use_mult', 
                    help='multiplexed iterative reconstruction') 

parser.add_argument('-i', type=int, action='store', dest='num_iter', \
                        help='number of training iterations', default = 10000)

parser.add_argument('--obj', type=int, action='store', dest='obj_ind', \
                        help='obj number to reconstruct', default = 0)

parser.add_argument('--uf', type=int, action='store', dest='upsample_factor', \
                    help='High resolution object pixels = collected image pixels * upsample_factor', default = 1)
        
parser.add_argument('--num_slices', type=int, action='store', dest='num_slices', \
                    help='num z slices', default = 1)
    
parser.add_argument('--x_corner', type=int, 
                    action='store', dest='start_x_corner',
                    default = 0)

parser.add_argument('--y_corner', type=int, 
                    action='store', dest='start_y_corner',
                    default = 0)

parser.add_argument('--x_size', type=int, 
                    action='store', dest='x_size',
                    default = 512)

parser.add_argument('--y_size', type=int, 
                    action='store', dest='y_size',
                    default = 512)

parser.add_argument('--overlap_x', type=int, 
                    action='store', dest='overlap_x',
                    default = 256)

parser.add_argument('--overlap_y', type=int, 
                    action='store', dest='overlap_y',
                    default = 256)

parser.add_argument('--patches_x', type=int, 
                    action='store', dest='num_patches_x',
                    default = 7)

parser.add_argument('--patches_y', type=int, 
                    action='store', dest='num_patches_y',
                    default = 7)

args = parser.parse_args()


'''
# parameters for testing
--x_corner 0 --y_corner 0 --x_size 256 --y_size 256 --overlap_x 128 --overlap_y 128 --patches_x 2 --patches_y 2
'''


    
### INPUTS ###

input_path = args.input_path
dataset_type = 'training'
obj_ind = args.obj_ind
upsample_factor = args.upsample_factor
num_slices = args.num_slices
reg = np.finfo(np.float32).eps.item()
save_tag_recons = args.save_tag_recons
num_patterns = args.num_patterns # for multiplexed
batch_size = args.batch_size # for single LED stack
adam_learning_rate = args.adam_learning_rate



start_x_corner = args.start_x_corner
start_y_corner = args.start_y_corner
x_size = args.x_size
y_size = args.y_size
overlap_x = args.overlap_x
overlap_y = args.overlap_y
num_patches_x = args.num_patches_x
num_patches_y = args.num_patches_y


use_mult = args.use_mult
multiplexed_description = args.multiplexed_description
num_iter = args.num_iter


### END OF INPUTS ###


folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)
subfolder_name = folder_name + '/reconstruction' + save_tag_recons
print(subfolder_name)

if use_mult:
    # multiplexed iterative
    base_command = 'python fpm_optimizer_v2.py --input_path ' + input_path + ' --obj_ind ' + str(obj_ind) + \
                    ' -i ' + str(num_iter) + ' -p ' + str(num_patterns) + ' --save_tag real_multiplexed ' + \
                    '--real_data --alr ' + str(adam_learning_rate) + ' --uf ' + str(upsample_factor) + ' --real_mult ' + \
                    '--mult --num_slices ' + str(num_slices) +' --slice_spacing 0 --md ' + multiplexed_description + ' --ones --xcrop ' \
                    + str(x_size) + ' --ycrop ' + str(y_size) + ' --window --save_tag_recons ' + save_tag_recons
else:
    # All LEDs, full stack iterative
    base_command = 'python fpm_optimizer_v2.py --input_path ' + input_path + ' --obj_ind ' + str(obj_ind) + \
                    ' -i ' + str(num_iter) + ' -b ' + str(batch_size) + ' ' + \
                    '--real_data --alr ' + str(adam_learning_rate) + ' --uf ' + str(upsample_factor) + \
                    ' --num_slices ' + str(num_slices) +' --slice_spacing 0 --ones --xcrop ' \
                    + str(x_size) + ' --ycrop ' + str(y_size) + ' --window --save_tag_recons ' + save_tag_recons
                
visualize_window = create_window(x_size,y_size,signal.windows.bartlett)
# visualize_window = create_window(x_size, y_size)

evaluate_patch_func = lambda x_corner, y_corner: evaluate_patch(base_command, x_corner, y_corner, subfolder_name)

start_time = time.time()

full_field, full_field_window = \
    merge_patches_func(upsample_factor,
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
                       )

end_time = time.time()

print('total time (minutes): ' + str((end_time-start_time)/60))


np.save(subfolder_name + '/full_field.npy', full_field)
np.save(subfolder_name + '/full_field_window.npy', full_field_window)


print('saved in ' + subfolder_name)
