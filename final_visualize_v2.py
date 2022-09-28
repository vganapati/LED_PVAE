#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:31:00 2022

@author: vganapa1
"""
import numpy as np
from final_visualize_v2_helper import visualize_all

### INPUTS ###
input_data= 'dataset_MNIST_multislice_v2' #'dataset_foam_v2_pac1' # 'dataset_cells_vae', 'dataset_foam_v2_pac1', 'dataset_foam_v2', 'dataset_MNIST_multislice_v2'
save_tag='mnist' #'foam_pac1' # 'cells', 'foam_pac1', 'foam_pac2', 'foam2', 'mnist'
noise_level=3
num_examples=3 # neural network training examples is 10^num_examples
num_patterns=1
plt_flag = True
single = False # single means that a single set of the SAME illumination patterns are used for all objects
example_num_i = None # example_num_i is ONLY used when num_examples==0 (i.e. deep prior)
force_save_path = None # 'mnist_noise_5_ex_4_p_4_pnm3_2'

######## inputs that change with different input_data datasets
batch_size_opt = 5 # batch size in the iterative optimization
num_iter = 10000 # number of iterations in the iterative optimization
num_slices = 2
########

t2_reg = 1e-2 # regularization lambda
adam_learning_rate = 1e-3 # learning rate of the iterative solve
dataset_type = 'training'
batch_size = 10 # batch_size for the neural network training
obj_ind = 0
slice_ind = 0
visualize_func = np.angle
compare_val_ind = 1 # mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity


# # visualize_func = np.abs
# vmin = 0
# vmax = 1.4

# visualize_func = np.angle
vmin = -np.pi/2
vmax = np.pi/2

### END of INPUTS ###

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
                                                    force_save_path = force_save_path,
                                                    example_num_i = example_num_i,
                                                    vmin=vmin,
                                                    vmax=vmax,
                                                    )

print(compare_value_vec)