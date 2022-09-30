#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:20:48 2022

@author: vganapa1
"""
import sys
import numpy as np
from final_visualize_v2_helper import get_low_res, plotter, sub_plotter, make_video, alpha_scatter
from helper_functions import create_alpha_mat, trim_lit_coord
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

##############
### INPUTS ###
input_data='dataset_MNIST_multislice_v2' # 'dataset_foam_v2_pac1' or 'dataset_MNIST_multislice_v2'
noise_level=3 # poisson_noise_multiplier is 10^noise_level 
single = False # single means that a single set of the SAME illumination patterns are used for all objects
obj_ind = 0 # object index of the dataset

vmin_abs = 0
vmax_abs = 1.4

vmin_ang = -np.pi/2
vmax_ang = np.pi/2

vmin_alpha = 0
vmax_alpha = 1

create_video = True

### END of INPUTS ###
#####################

### Load values ###

normalizer = float(np.load(input_data + '/normalizer.npy'))
offset = float(np.load(input_data + '/offset.npy')) # not applied to multiplexed

LitCoord = np.load(input_data + '/LitCoord.npy')
LitCoord2 = trim_lit_coord(LitCoord)

dx_obj = np.load(input_data + '/dx_obj.npy')

### load z coordinates ###

f = float(np.load(input_data + "/f.npy")) # f is distance from the last slice to the focal plane
slice_spacing = float(np.load(input_data + "/slice_spacing.npy"))
num_slices = int(np.load(input_data + "/num_slices.npy"))

z_vec = np.arange(0,slice_spacing*num_slices,slice_spacing)+f



alpha, im_stack, im_stack_multiplexed, obj, image_path, save_tag_mult = \
get_low_res(input_data,
            noise_level,
            single,
            obj_ind,
            normalizer,
            offset)

im_stack = im_stack*normalizer
im_stack_multiplexed = im_stack_multiplexed*normalizer

save_folder = image_path + '/multiplexed/' + save_tag_mult
num_leds = alpha.shape[0]
num_patterns = alpha.shape[-1] # can override
# num_patterns = 1 # override num_patterns

alpha = np.expand_dims(alpha,0) # add a batch dimension

alpha_mat = \
create_alpha_mat(alpha, # alpha_sample 
                 1, # batch_size_per_gpu
                 num_leds,
                 num_patterns,
                 LitCoord2)


alpha_mat = np.squeeze(alpha_mat,0) # remove batch dimension



# high-res object 

for s in range(obj.shape[-1]):

    plotter(np.abs(obj[:,:,s]), 'obj_abs_slice' + str(s), 'obj_abs_slice' + str(s), vmin_abs, vmax_abs, save_folder)
    plotter(np.angle(obj[:,:,s]), 'obj_ang_slice' + str(s), 'obj_ang_slice' + str(s), vmin_ang, vmax_ang, save_folder)


# multiplexed alpha

# low-res multiplexed

filenames = []
for patt in range(num_patterns):
    
    alpha_scatter(alpha_mat[:,:,patt], LitCoord2, 'alpha_mat_patt' + str(patt), save_folder)

    plotter(im_stack_multiplexed[:,:,patt], 'im_multiplexed_patt' + str(patt), 'im_multiplexed_patt' + str(patt), 
            0, 1, save_folder)
    
    full_save_name =\
    sub_plotter([alpha_mat[:,:,patt], im_stack_multiplexed[:,:,patt]], 
                'patt' + str(patt), 'patt' + str(patt), 
                [vmin_alpha, 0], 
                [vmax_alpha/2, 1], 
                save_folder, cmap='gray', grid=False)
    filenames.append(full_save_name)

make_video(filenames, 
           'multiplexed',
           remove_files=False, loop=1, fps=1)



# single alpha

# low-res single
filenames = []
for led_ind in range(num_leds):
    alpha_single = np.zeros([1,num_leds,1])
    alpha_single[0,led_ind,0] = 1
    
    alpha_single_mat = \
    create_alpha_mat(alpha_single, # alpha_sample 
                     1, # batch_size_per_gpu
                     num_leds,
                     1, # num_patterns
                     LitCoord2)
    
    alpha_single_mat = np.squeeze(alpha_single_mat,0) # remove batch dimension

    
    alpha_scatter(alpha_single_mat[:,:,0], LitCoord2, 'alpha_single_led' + str(led_ind), save_folder)
    plotter(im_stack[:,:,led_ind], 'im_stack_led' + str(led_ind), 'im_stack_led' + str(led_ind), 
            0, 1, save_folder)

    full_save_name =\
    sub_plotter([alpha_single_mat[:,:,0], im_stack[:,:,led_ind]], 
                'single_led' + str(led_ind), 'single_led' + str(led_ind), 
                [0, 0], 
                [1, 1], 
                save_folder, cmap='gray', grid=False)
    filenames.append(full_save_name)


if create_video:
    make_video(filenames, 
               'single_led',
               remove_files=False, loop=1, fps=5)


# create a colorbar
plt.figure(figsize=[10,10])
plt.imshow([[0,1]],cmap='gray')
cbar = plt.colorbar()
cbar.ax.tick_params(size=0)
cbar.set_ticks([])
plt.savefig(save_folder + '/colorbar.png', bbox_inches='tight',dpi=600)


# uncomment to format into latex table

'''
def format_mat(mat,width='1in',prefix = os.getcwd()):
    for i in range(len(mat)): # number of rows
        for j in range(len(mat[0])): # number of columns
            mat[i][j] = '\includegraphics[width=' + width + ']{'+prefix+'/'+mat[i][j]+'}'
        
    return(mat)

mat1 = [[save_folder + '/obj_abs_slice0.png', save_folder + '/obj_ang_slice0.png'],
        [save_folder + '/obj_abs_slice1.png', save_folder + '/obj_ang_slice1.png']]

mat2 = [[save_folder + '/alpha_single_led0.png', save_folder + '/im_stack_led0.png'],
        [save_folder + '/alpha_single_led1.png', save_folder + '/im_stack_led1.png'],
        [save_folder + '/alpha_single_led2.png', save_folder + '/im_stack_led2.png'],
        [save_folder + '/alpha_single_led3.png', save_folder + '/im_stack_led3.png']]

mat3 = [[save_folder + '/alpha_mat_patt0.png', save_folder + '/im_multiplexed_patt0.png'],
        [save_folder + '/alpha_mat_patt1.png', save_folder + '/im_multiplexed_patt1.png'],
        [save_folder + '/alpha_mat_patt2.png', save_folder + '/im_multiplexed_patt2.png'],
        [save_folder + '/alpha_mat_patt3.png', save_folder + '/im_multiplexed_patt3.png']]

mat1=format_mat(mat1)
mat2=format_mat(mat2, width = '0.5in')
mat3=format_mat(mat3, width = '0.5in')



filename = 'table2.tex'
tablefmt="latex_raw"

table1 = tabulate(mat1, tablefmt=tablefmt, floatfmt=".2f")
print(tabulate(mat1, floatfmt=".2f"))

table2 = tabulate(mat2, tablefmt=tablefmt, floatfmt=".2f")
print(tabulate(mat2, floatfmt=".2f"))

table3 = tabulate(mat3, tablefmt=tablefmt, floatfmt=".2f")
print(tabulate(mat3, floatfmt=".2f"))

f = open(filename, "w")
f.write(table1)
f.write(table2)
f.write(table3)
f.close()
'''


'''
Side by side tables:
    
1)
obj abs | obj angle
------------------
obj abs | obj angle
------------------

2)
alpha sin | low res img 
------------------
alpha sin | low res img 
------------------
alpha sin | low res img
------------------
alpha sin | low res img

3)
alpha mult | low res img 
------------------
alpha mult | low res img
------------------
alpha mult | low res img
------------------
alpha mult | low res img

'''


