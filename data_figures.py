#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:22:47 2022

@author: vganapa1

Creates figures of LED patterns and corresponding images for single and multiplexed
"""

import numpy as np
import matplotlib.pyplot as plt
from SyntheticMNIST_functions import create_folder
import imageio
import sys

# Inputs

dataset_path = 'dataset_frog_blood_v3'
object_path = 'training/example_000000'
mult_object_inds = [0,1,2,3,4,5,6,7]

multiplex_description = '_Dirichlet'
start_x_corner = 768
start_y_corner = 768
image_x = 512
image_y = 512
mult_ind = 0
slice_ind=0
cmap = 'gray' # 'gray'

neural_net_save_path = 'frog_mult6_v3_100k'

# Load values

led_position_xy = np.load(dataset_path + '/led_position_xy.npy')
num_leds = led_position_xy.shape[0]

# Create folder to save output

save_folder = dataset_path + '/' + object_path + '/formatted_figs'
create_folder(save_folder)

# Single images

im_stack = np.load(dataset_path + '/' + object_path + '/im_stack.npy')


# Plot reconstruction

def plot_recon(reconstruction_patch,save_folder,tag,slice_ind):
    amplitude_patch = np.abs(reconstruction_patch)
    phase_patch = np.angle(reconstruction_patch)

    plt.figure()
    plt.imshow(amplitude_patch, cmap=cmap)
    plt.axis('off')
    plt.savefig(save_folder + '/' + tag + '_recon_patch_amp_slice_' + str(slice_ind), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.colorbar()

    plt.figure()
    plt.imshow(phase_patch, cmap=cmap) # cmap='twilight_shifted', cmap='twilight'
    plt.axis('off')
    plt.savefig(save_folder + '/' + tag + '_recon_patch_phase_slice_' + str(slice_ind), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.colorbar()

# Reconstruction from iterative for object_path
iterative_reconstruction = np.load(dataset_path + '/' + object_path + '/reconstruction/full_field.npy')[slice_ind]
reconstruction_patch = iterative_reconstruction[start_x_corner:start_x_corner+image_x, start_y_corner:start_y_corner+image_y]
tag='iterative'
plot_recon(reconstruction_patch,save_folder,tag,slice_ind)
np.save('iter_reconstruction_patch.npy',reconstruction_patch)


# Reconstruction from neural network
nn_reconstruction = np.load(neural_net_save_path + '/full_field_restore_None.npy')[slice_ind]
# nn_reconstruction = np.load(save_path + '/full_field_example_' + str(example_num) + '.npy')[slice_ind]
reconstruction_patch = nn_reconstruction[start_x_corner:start_x_corner+image_x, start_y_corner:start_y_corner+image_y]
tag='neural_net'
plot_recon(reconstruction_patch,save_folder,tag,slice_ind)

np.save('nn_reconstruction_patch.npy',reconstruction_patch)


# XXX choose the phase reference to make them as close as possible


# function to make figures for each alpha and im_patch of interest
def make_figs(im_patch, alpha, vmax, tag, save_folder,ind):
    print('figures saved in: ' + save_folder)
    plt.figure()
    plt.imshow(im_patch, vmin=0, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.savefig(save_folder + '/' + tag + '_patch_im_' + str(ind), dpi=300, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.scatter(led_position_xy[:,0], led_position_xy[:,1],c=alpha, s=100, cmap='Greens', edgecolors= "black", 
                vmin=np.min(alpha), vmax=1.5)
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')
    plt.axis('off')
    plt.savefig(save_folder + '/' + tag + '_alpha_im_' + str(ind), dpi=300, bbox_inches='tight', pad_inches=0)

# Iterate through single images and create corresponding illumination pattern

for ind,im in enumerate(im_stack):
    im_patch = im[start_x_corner:start_x_corner+image_x, start_y_corner:start_y_corner+image_y]
    if ind==0:
        vmax = np.max(im_patch)
    plt.figure()
    
    alpha = np.zeros(num_leds)
    alpha[ind] = 1
    
    tag = 'single_led'
    make_figs(im_patch, alpha, vmax, tag, save_folder, ind)



# Multiplexed patterns
alpha_multiplexed = np.load(dataset_path + '/real_multiplexed/all_alpha_train' + multiplex_description + '.npy')

# Iterate through objects and output the multiplexed pattern and image patch
vmax = 65535
for ind, object_ind in enumerate(mult_object_inds):
    object_path = 'training/example_{:06d}'.format(object_ind)   
      
    # Multiplexed images
    file_multiplex = dataset_path + '/' + object_path + '/Multiplex' + \
                                           multiplex_description + '/{:04d}.png'.format(mult_ind)
    im = imageio.imread(file_multiplex)
    im_patch = np.rot90(im[start_x_corner:start_x_corner+image_x, start_y_corner:start_y_corner+image_y],3)
    alpha = alpha_multiplexed[object_ind,:,mult_ind]

    
    tag = 'multiplexed_' + str(mult_ind)
    save_folder = dataset_path + '/' + object_path + '/formatted_figs'
    create_folder(save_folder)
    make_figs(im_patch, alpha, vmax, tag, save_folder, 0)
    


# Reconstructions from neural network