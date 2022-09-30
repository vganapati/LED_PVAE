#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:31:00 2022

@author: vganapa1
"""
import numpy as np
from final_visualize_v2_helper import visualize_all, make_table

#####################
### INPUTS ###
input_data='dataset_foam_v2_pac1' #'dataset_foam_v2_pac1' or 'dataset_MNIST_multislice_v2'
save_tag='foam_pac1' #'foam_pac1' or 'mnist'
single = False # single means that a single set of the SAME illumination patterns are used for all objects

# items that can be varied
noise_level_vec=[2,3,4,5] # poisson noise multiplier is 10^noise_level
num_examples_vec=[0,1,2,3,4] # neural network training examples is 10^num_examples
num_patterns_vec=[1,2,3,4]
compare_val_ind_vec=[0,1,2,3,4] # mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity

# variable to vary in the table
vary_vec = num_patterns_vec # num_examples_vec
vary_name = 'num_patterns_vec' # 'num_examples_vec'

# index of the vectors that is fixed if the variable is not varied
noise_level_ind=0
num_examples_ind=3
num_patterns_ind=0
compare_val_ind=1

# Average over for text tables
obj_ind_vec = [0,1,2,3,4,5,6,7,8,9]
slice_ind_vec = [0] # [0] for single slice (2D) or [0,1] for 2 slices (3D)

obj_ind_visualize = 0 # for graphics tables
slice_ind_visualize = 0 # for graphics tables

### Can be varied for image tables
visualize_func = np.angle # np.abs or np.angle

# np.angle
vmin = -np.pi/2
vmax = np.pi/2

# # np.abs # uncomment to change the colorbar range for figures
# vmin = 0
# vmax = 1.4


# change inputs to match what was used in training/optimization of dataset
batch_size_opt = 5 # batch size in the iterative optimization
num_iter = 10000 # number of iterations in the iterative optimization
num_slices = 1 # 2 for 'dataset_MNIST_multislice_v2', 1 for 'dataset_foam_v2_pac1'
t2_reg = 1e-2 # regularization lambda
adam_learning_rate = 1e-3 # learning rate of the iterative solve
dataset_type = 'training'
batch_size = 10 # batch_size for the neural network training

### END of INPUTS ###
#####################

if compare_val_ind==0:
    reduce_func = np.argmin
else:
    reduce_func = np.argmax # np.min for metrics that minimize, np.max for metrics that maximize


compare_value_mat_sum = None
compare_visualize_mat = []
compare_visualize_mat_diff = []

for slice_ind in slice_ind_vec:
    for obj_ind in obj_ind_vec:
        compare_value_mat = []
        
        if (slice_ind == slice_ind_visualize) and (obj_ind == obj_ind_visualize):
            plt_flag=True
        else:
            plt_flag=False
                
        for i in range(len(vary_vec)):

            print(slice_ind)
            print(obj_ind)
            print(i)
            if vary_name == 'noise_level_vec':
                noise_level = noise_level_vec[i]
            else:
                noise_level = noise_level_vec[noise_level_ind]
        
            if vary_name == 'num_examples_vec':
                num_examples = num_examples_vec[i]
            else:
                num_examples = num_examples_vec[num_examples_ind]
            
            if vary_name == 'num_patterns_vec':
                num_patterns = num_patterns_vec[i]
            else:
                num_patterns = num_patterns_vec[num_patterns_ind]
            
            if vary_name == 'compare_val_ind_vec':
                compare_val_ind=compare_val_ind_vec[i]
        
            compare_value_vec, name_vec, \
                save_name_vec_fullpath, \
                    save_name_vec_fullpath_diff = visualize_all(obj_ind,
                                                                slice_ind,
                                                                compare_val_ind,
                                                                input_data,
                                                                save_tag,
                                                                noise_level,
                                                                num_examples,
                                                                num_patterns,
                                                                adam_learning_rate,
                                                                batch_size_opt,
                                                                num_iter,
                                                                t2_reg,
                                                                dataset_type,
                                                                visualize_func,
                                                                num_slices, # total number of slices
                                                                batch_size, # batch size of the neural network training
                                                                plt_flag=plt_flag,
                                                                single=single,
                                                                example_num_i = obj_ind,
                                                                vmin=vmin,
                                                                vmax=vmax,
                                                                )
            
            compare_value_vec_trimmed = [compare_value_vec[0], 
                                         compare_value_vec[1:4][reduce_func(compare_value_vec[1:4])], 
                                         compare_value_vec[4:7][reduce_func(compare_value_vec[4:7])]]
            compare_value_mat.append(compare_value_vec_trimmed)
            if (slice_ind == slice_ind_visualize) and (obj_ind == obj_ind_visualize):
                save_name_vec_fullpath_trimmed = save_name_vec_fullpath[0:2] + \
                                                  [save_name_vec_fullpath[2:5][reduce_func(compare_value_vec[1:4])], 
                                                  save_name_vec_fullpath[5:8][reduce_func(compare_value_vec[4:7])]]
                
                compare_visualize_mat.append(save_name_vec_fullpath_trimmed)
                
                save_name_vec_fullpath_diff_trimmed = save_name_vec_fullpath_diff[0:2] + \
                                                       [save_name_vec_fullpath_diff[2:5][reduce_func(compare_value_vec[1:4])], 
                                                       save_name_vec_fullpath_diff[5:8][reduce_func(compare_value_vec[4:7])]]
                compare_visualize_mat_diff.append(save_name_vec_fullpath_diff_trimmed)
                
        compare_value_mat = np.stack(compare_value_mat) # vary vec x reconstruction algorithm
        

        if compare_value_mat_sum is None:
            compare_value_mat_sum = compare_value_mat
        else:
            compare_value_mat_sum += compare_value_mat


compare_visualize_mat = np.stack(compare_visualize_mat)
compare_visualize_mat_diff = np.stack(compare_visualize_mat_diff)

compare_value_mat_ave = compare_value_mat_sum/(len(obj_ind_vec)*len(slice_ind_vec))

make_table(compare_value_mat_ave, vary_name, vary_vec, filename = 'table.tex')




if (obj_ind_visualize is not None) and (slice_ind_visualize is not None):
    
    make_table(compare_visualize_mat, vary_name, vary_vec, filename = 'visualize_table.tex', tablefmt="latex_raw")
    
    make_table(compare_visualize_mat_diff, vary_name, vary_vec, filename = 'visualize_diff_table.tex', tablefmt="latex_raw")
    

