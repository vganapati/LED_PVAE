#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:23:40 2022

@author: vganapa1
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from helper_functions import find_angle_offset, psnr_complex
import argparse
from skimage.metrics import mean_squared_error
from visualizer_functions import make_cutout_fig
from SyntheticMNIST_functions import create_folder

### COMMAND LINE ARGS ###

parser = argparse.ArgumentParser(description='Get command line args')



parser.add_argument('--input_path', action='store', help='path to overall folder containing training data',
                    default = 'dataset_frog_blood_v3')


parser.add_argument('--input_path_vec', action='store', help='path(s) to overall folder(s) containing training and test data', 
                     nargs='+')

parser.add_argument('--save_tag_recons', action='store', help="path(s) for iterative results is /reconstruction_'save_tag_recons'", nargs='+')

parser.add_argument('--save_path', action='store', help='path(s) where neural network output is saved', nargs='+')

parser.add_argument('--md', dest = 'multiplexed_description',
                    action='store', help='description(s) of multiplex type', nargs='+') # _Dirichlet or _Random

parser.add_argument('--obj', type=int, action='store', dest='obj_ind', \
                        help='obj number to reconstruct', default = 0)
    
args = parser.parse_args()


### END COMMAND LINE ARGS ###

### INPUTS ###
input_path = args.input_path
obj_ind = args.obj_ind
save_tag_recons = args.save_tag_recons # for iterative path
save_path = args.save_path # for neural network solution path
multiplexed_description = args.multiplexed_description

dataset_type = 'training'
visualize_trim = 64
num_leds = int(np.load(input_path + '/num_leds.npy'))

# Inputs for patch cutout
start_x_corner_inner = 480
start_y_corner_inner = 480
image_x_inner = 512
image_y_inner = 512

start_corner = np.array([start_x_corner_inner,start_y_corner_inner])
size = np.array([image_x_inner,image_y_inner])

### END OF INPUTS ###

create_folder('exp_data_figures')

### FUNCTIONS ###

def visualize_all_slices(full_field, num_slices, title, reference,
                         vmin=None, vmax=None,
                         vmin_ang=None, vmax_ang=None):
    MSE_vec = []
    for ss in range(num_slices):
        fig_name = 'slice_' + str(ss) + '_amplitude_' + title
        
        full_field_trimmed = full_field[ss,visualize_trim:-visualize_trim,
                                      visualize_trim:-visualize_trim]
        reference_trimmed = reference[ss,visualize_trim:-visualize_trim,
                                      visualize_trim:-visualize_trim]
        
        angle_offset = find_angle_offset(reference_trimmed, full_field_trimmed)
        print('angle_offset is: ' + str(angle_offset))
        
        
        
        full_field_trimmed = full_field_trimmed*np.exp(1j*angle_offset)
        
        psnr, mse = psnr_complex(reference_trimmed,full_field_trimmed)
        MSE = np.mean((np.abs(reference_trimmed-full_field_trimmed))**2)
        MSE_vec.append(MSE)
        print('PSNR is: ' + str(psnr))
        print('MSE is: ' + str(MSE))
        
        print('max of amplitude is: ' + str(np.max(np.abs(full_field_trimmed))))
        print('min of amplitude is: ' + str(np.min(np.abs(full_field_trimmed))))
        
        print('max of angle is: ' + str(np.max(np.angle(full_field_trimmed))))
        print('min of angle is: ' + str(np.min(np.angle(full_field_trimmed))))

        plt.figure()
        plt.imshow(np.abs(full_field_trimmed),
                   vmin=vmin, vmax=vmax, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fig_name + '.png', pad_inches=0, dpi=1200)
        plt.colorbar()
        plt.savefig(fig_name + '_cb.png', pad_inches=0, dpi=1200)
        plt.title(fig_name)
        
        img=np.abs(full_field_trimmed)
        if vmin_ang is None:
            vmin_i = np.min(img)
        else:
            vmin_i = vmin_ang

        if vmax_ang is None:
            vmax_i = np.max(img)
        else:
            vmax_i = vmax_ang
        make_cutout_fig(np.abs(full_field_trimmed), start_corner, 
                        size, vmin_i, vmax_i, 'exp_data_figures', fig_name + '_patch')
        
        fig_name = 'slice_' + str(ss) + '_angle_' + title
        plt.figure()
        plt.imshow(np.angle(full_field_trimmed),
                   vmin=vmin_ang, vmax=vmax_ang, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fig_name + '.png', pad_inches=0, dpi=1200)
        plt.colorbar()
        plt.savefig(fig_name + '_cb.png', pad_inches=0, dpi=1200)
        plt.title(fig_name)
        
        img=np.angle(full_field_trimmed)
        if vmin_ang is None:
            vmin_i = np.min(img)
        else:
            vmin_i = vmin_ang

        if vmax_ang is None:
            vmax_i = np.max(img)
        else:
            vmax_i = vmax_ang
        make_cutout_fig(img, start_corner, 
                        size, vmin_i, vmax_i, 'exp_data_figures', fig_name + '_patch')
        
    MSE_vec = np.stack(MSE_vec)
    return(MSE_vec)
### END OF FUNCTIONS ###


folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)


MSE_mat = []

# Iterative solutions

for ind,save_tag_recons_0 in enumerate(save_tag_recons):
    subfolder_name = folder_name + '/reconstruction' + save_tag_recons_0
    filepath_iter = subfolder_name + '/full_field.npy'
    print(filepath_iter)
    full_field_iter = np.load(filepath_iter)
    num_slices = full_field_iter.shape[0]
    if ind==0:
        reference = full_field_iter
    MSE_vec = visualize_all_slices(full_field_iter, num_slices, save_tag_recons_0, reference)
    MSE_mat.append(MSE_vec)
    
# Neural network solutions

for ind,save_path_0 in enumerate(save_path):

    if multiplexed_description[ind] == '_Random':
        exposure = 50
    elif multiplexed_description[ind] == '_Dirichlet':
        exposure = 100
        

    filepath_nn = save_path_0 + '/full_field_example_' + str(obj_ind) + '.npy'
    print(filepath_nn)
    full_field_nn = np.load(filepath_nn)
    
    full_field_nn = full_field_nn/np.sqrt(exposure)
    MSE_vec = visualize_all_slices(full_field_nn, num_slices, save_path_0, reference)
    MSE_mat.append(MSE_vec)

# reference to the zeroth iterative solution


# calculate merit

# optimization to calibrate constant phase factor which can't be measured

# visualize_all_slices(full_field_iter-full_field_nn, num_slices)



