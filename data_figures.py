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
from visualizer_functions import make_cutout_fig
import imageio
import sys

##############
### INPUTS ###

dataset_path = 'dataset_frog_blood_v3'
object_path = 'training/example_000000'
mult_object_inds = [0,1,2,3,4,5,6,7]

multiplex_description = '_Dirichlet'
start_x_corner = 64
start_y_corner = 64
image_x = 2048-64*2
image_y = 2048-64*2

start_x_corner_inner = 480
start_y_corner_inner = 480
image_x_inner = 512
image_y_inner = 512

mult_ind = 0
slice_ind=0
cmap = 'gray'

neural_net_save_path = 'frog_blood_pvae'

### END OF INPUTS ###
#####################

# Load values

led_position_xy = np.load(dataset_path + '/led_position_xy.npy')
num_leds = led_position_xy.shape[0]

# Create folder to save output

save_folder = dataset_path + '/' + object_path + '/formatted_figs'
create_folder(save_folder)

# Single images

im_stack = np.load(dataset_path + '/' + object_path + '/im_stack.npy')



# function to make figures for each alpha and im_patch of interest
def make_figs(im_patch, alpha, vmax, tag, save_folder,ind):
    print('figures saved in: ' + save_folder)
    plt.figure()
    plt.imshow(im_patch, vmin=0, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.savefig(save_folder + '/' + tag + '_patch_im_' + str(ind), dpi=300, bbox_inches='tight', pad_inches=0)

    img = im_patch
    start_corner = np.array([start_x_corner_inner,start_y_corner_inner])
    size = np.array([image_x_inner,image_y_inner])
    make_cutout_fig(img, start_corner, 
                    size, 0, vmax, save_folder, tag + '_patch_im_' + str(ind))

    plt.figure()
    plt.scatter(led_position_xy[:,0], led_position_xy[:,1],c=alpha, s=100, cmap='Greens', edgecolors= "black", 
                vmin=np.min(alpha), vmax=1)
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