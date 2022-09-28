#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:31:00 2022

@author: vganapa1
"""
import numpy as np
import sys
from final_visualize_v2_helper import visualize_all, line_plotter

### INPUTS ###
input_data='dataset_MNIST_multislice_v2' #'dataset_cells_vae', 'dataset_foam_v2', 'dataset_MNIST_multislice_v2', 'dataset_foam_v2_pac1', 'dataset_foam_v2_pac2'
save_tag='mnist' # 'cells','foam2', 'mnist', 'foam_pac1', 'foam_pac2'
num_slices = 2
single=False # single means that a single set of the SAME illumination patterns are used for all objects

# items that can be varied
noise_level_vec=[2,3,4,5]
num_examples_vec=[0,1,2,3,4] # neural network training examples is 10^num_examples
num_patterns_vec=[1,2,3,4]
compare_val_ind_vec=[0,1,2,3,4] # mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity

vary_vec_0 = num_patterns_vec # x-axis
vary_name_0 = 'num_patterns_vec'

vary_vec_1 = noise_level_vec # marked by legend
vary_name_1 = 'noise_level_vec'

# only use the value for the quantity not included in vary_vec_0 or vary_vec_1
noise_level_ind=2
num_examples_ind=4
num_patterns_ind=0

compare_val_ind=1
compare_val_names = ['mse_recon', 'psnr_recon', 'ssim_recon_angle', 
                     'ssim_recon_abs', 'ssim_recon_intensity']
compare_val_name = compare_val_names[compare_val_ind]

if compare_val_ind==0:
    reduce_fun = np.argmin
else:
    reduce_func = np.argmax # np.min for metrics that minimize, np.max for metrics that maximize

# Do not change the following
# Average over for text tables
obj_ind_vec = [0,1,2,3,4,5,6,7,8,9]
slice_ind_vec = [0,1]

plt_flag = False
obj_ind_visualize = None # for graphics tables
slice_ind_visualize = None # for graphics tables

visualize_func = np.angle # unused in this script

######## inputs that change with different input_data datasets
batch_size_opt = 5 # batch size in the iterative optimization
num_iter = 10000 # number of iterations in the iterative optimization
########

t2_reg = 1e-2 # regularization lambda
adam_learning_rate = 1e-3 # learning rate of the iterative solve
dataset_type = 'training'
batch_size = 10 # batch_size for the neural network training


### END of INPUTS ###

compare_value_mat_sum = np.zeros([len(vary_vec_0), len(vary_vec_1),3]) # 3 denotes NN, multiplexed iterative, single iterative



for slice_ind in slice_ind_vec:
    for obj_ind in obj_ind_vec:
        compare_value_mat = []
                
        for i in range(len(vary_vec_0)):
            print(i)
            if vary_name_0 == 'noise_level_vec':
                noise_level = noise_level_vec[i]
            else:
                noise_level = noise_level_vec[noise_level_ind]
        
            if vary_name_0 == 'num_examples_vec':
                num_examples = num_examples_vec[i]
            else:
                num_examples = num_examples_vec[num_examples_ind]
            
            if vary_name_0 == 'num_patterns_vec':
                num_patterns = num_patterns_vec[i]
            else:
                num_patterns = num_patterns_vec[num_patterns_ind]
            
            if vary_name_0 == 'compare_val_ind_vec':
                compare_val_ind=compare_val_ind_vec[i]
                    
            for j in range(len(vary_vec_1)):

                print(j)
                if vary_name_1 == 'noise_level_vec':
                    noise_level = noise_level_vec[j]
            
                if vary_name_1 == 'num_examples_vec':
                    num_examples = num_examples_vec[j]
                
                if vary_name_1 == 'num_patterns_vec':
                    num_patterns = num_patterns_vec[j]
                
                if vary_name_1 == 'compare_val_ind_vec':
                    compare_val_ind=compare_val_ind_vec[j]
            
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
                                                                    )
                
                compare_value_vec_trimmed = [compare_value_vec[0], 
                                             compare_value_vec[1:4][reduce_func(compare_value_vec[1:4])], 
                                             compare_value_vec[4:7][reduce_func(compare_value_vec[4:7])]]
                
                compare_value_mat_sum[i,j,:] += compare_value_vec_trimmed

# compare_value_mat_ave is len(vary_vec_0) x len(vary_vec_1) x 3  
compare_value_mat_ave = compare_value_mat_sum/(len(obj_ind_vec)*len(slice_ind_vec))

fig_save_name = vary_name_0 + '_' + vary_name_1 + '_line_plot.png'
save_folder = input_data
line_plotter(compare_value_mat_ave, 
             vary_vec_0,
             vary_name_0,
             vary_vec_1,
             vary_name_1,
             compare_val_name,
             fig_save_name,
             save_folder)
