#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:46:17 2022

@author: vganapa1
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import compare, find_angle_offset
from helper_pattern_opt import load_multiplexed
from SyntheticMNIST_functions import create_folder
from tabulate import tabulate
from mpl_toolkits import mplot3d
import imageio
import os

def make_video(filenames, 
               gifname,
               remove_files=False, loop=0, fps=30):
    
    # Build GIF
    with imageio.get_writer(gifname + '.gif', mode='I', loop=loop, fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    if remove_files:
        # Remove files
        for filename in set(filenames):
            os.remove(filename)


def make_table(mat, vary_name, vary_vec, filename = 'table.tex', tablefmt="latex"):
    headers = [vary_name,
               'neural network',
               'all leds',
               'mult']
    if mat.shape[1]==4:
        headers.insert(1,'ground truth')
    table = tabulate(mat, tablefmt=tablefmt, floatfmt=".2f", headers = headers, showindex=vary_vec)
    print(tabulate(mat, floatfmt=".2f", headers = headers, showindex=vary_vec))

    f = open(filename, "w")
    f.write(table)
    f.close()
                      

        
def line_plotter(compare_value_mat_ave, 
                 vary_vec_0,
                 vary_name_0,
                 vary_vec_1,
                 vary_name_1,
                 compare_val_name,
                 fig_save_name, save_folder):
    
    '''
    compare_value_mat_ave is len(vary_vec_0) x len(vary_vec_1) x 3
    '''
    fig = plt.figure()

    for i in range(len(vary_vec_1)):
        for j in range(3):
            if j==0:
                color='C0.-'
            if j==1:
                color='C1.-'
            if j==2:
                color='C3.-'
            plt.plot(vary_vec_0,compare_value_mat_ave[:,i,j], color, 
                     label=vary_name_1 + '_' + str(vary_vec_1[i]),
                     linewidth=2)
            
    fig.axes[0].set_xticks(vary_vec_0)
    plt.tick_params(which='both',      # both major and minor ticks are affected
                    bottom=True,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    left=True,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False) # labels along the bottom edge are off    


    full_save_name_0 = save_folder + '/' + fig_save_name + '.png'
    plt.savefig(full_save_name_0, bbox_inches='tight',dpi=600)
    plt.title(vary_name_1)
    plt.xlabel(vary_name_0)
    plt.ylabel(compare_val_name)

    for i in range(len(vary_vec_1)):
        for j in range(3):
            plt.annotate(vary_vec_1[i],(vary_vec_0[-1]+0.05,compare_value_mat_ave[:,i,j][-1]))
            
    plt.tick_params(which='both',      # both major and minor ticks are affected
                    bottom=True,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    left=True,
                    labelleft=True,
                    labelright=False,
                    labelbottom=True) # labels along the bottom edge are off    


    full_save_name_1 = save_folder + '/' + fig_save_name + '_labelled.png'
    plt.savefig(full_save_name_1, bbox_inches='tight',dpi=600)
    
    return(full_save_name_0, full_save_name_1)

def sub_plotter_scatter(img, scatter_position, title, fig_save_name, vmin, vmax, save_folder, 
                cmap='gray',
                ):
    
    fig, ax = plt.subplots(1, 2, 
                   figsize = (5, 10))
    
    ax[0].axis('square')
    ax[0].set_xlim((-50, 50))
    ax[0].set_ylim((-50, 50))


    # plt.subplot(1, len(img_vec), ind+1)
    ax[0].scatter(scatter_position[0], scatter_position[1],c='g')
    # ax[0].axis('equal')
    # ax[0].set_aspect('auto')
    # ax[ind].set_xlim((-50, 50))
    # ax[ind].set_ylim((-50, 50))
    
    ax[0].tick_params(axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    labelright=False,
                    left=False,
                    labelleft=False,
                    labelbottom=False) # labels along the bottom edge are off
    ax[0].tick_params(axis='y',          # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    left=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False) # labels along the bottom edge are off
    
    # plt.subplot(1, len(img_vec), ind+1)
    ax[1].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].tick_params(axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    labelright=False,
                    left=False,
                    labelleft=False,
                    labelbottom=False) # labels along the bottom edge are off
    ax[1].tick_params(axis='y',          # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    right=False,
                    left=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False) # labels along the bottom edge are off
    
    full_save_name = save_folder + '/' + fig_save_name + '.png'
    plt.savefig(full_save_name, bbox_inches='tight',dpi=600)
    # plt.title(title)
    # plt.colorbar()
    return(full_save_name)

def sub_plotter(img_vec, title, fig_save_name, vmin_vec, vmax_vec, save_folder, 
                cmap='gray', grid=False,
                ):
    
    fig, ax = plt.subplots(1, len(img_vec), 
                   figsize = (5, 10))
    
    for ind, img in enumerate(img_vec):
        
        # plt.subplot(1, len(img_vec), ind+1)
        ax[ind].imshow(img, cmap=cmap, vmin=vmin_vec[ind], vmax=vmax_vec[ind])
        ax[ind].tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        right=False,
                        labelright=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False) # labels along the bottom edge are off
        ax[ind].tick_params(axis='y',          # changes apply to the y-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        right=False,
                        left=False,
                        labelleft=False,
                        labelright=False,
                        labelbottom=False) # labels along the bottom edge are off
        
        if grid:
            ax = plt.gca();
    
            # Major ticks
            ax.set_xticks(np.arange(1, img.shape[0], 1))
            ax.set_yticks(np.arange(1, img.shape[1], 1))
            
            # Minor ticks
            ax.set_xticks(np.arange(0.5, img.shape[0], 1), minor=True)
            ax.set_yticks(np.arange(0.5, img.shape[1], 1), minor=True)
            
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    full_save_name = save_folder + '/' + fig_save_name + '.png'
    plt.savefig(full_save_name, bbox_inches='tight',dpi=600)
    # plt.title(title)
    # plt.colorbar()
    return(full_save_name)



def plotter(img, title, fig_save_name, vmin, vmax, save_folder, cmap='gray', grid=False):
    plt.figure()
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.tick_params(axis='x',          # changes apply to the x-axis
    #                 which='both',      # both major and minor ticks are affected
    #                 bottom=False,      # ticks along the bottom edge are off
    #                 top=False,         # ticks along the top edge are off
    #                 right=False,
    #                 labelright=False,
    #                 left=False,
    #                 labelleft=False,
    #                 labelbottom=False) # labels along the bottom edge are off
    # plt.tick_params(axis='y',          # changes apply to the y-axis
    #                 which='both',      # both major and minor ticks are affected
    #                 bottom=False,      # ticks along the bottom edge are off
    #                 top=False,         # ticks along the top edge are off
    #                 right=False,
    #                 left=False,
    #                 labelleft=False,
    #                 labelright=False,
    #                 labelbottom=False) # labels along the bottom edge are off
    plt.axis('off')
    if grid:
        ax = plt.gca();

        # Major ticks
        ax.set_xticks(np.arange(1, img.shape[0], 1))
        ax.set_yticks(np.arange(1, img.shape[1], 1))
        
        # Minor ticks
        ax.set_xticks(np.arange(0.5, img.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(0.5, img.shape[1], 1), minor=True)
        
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    full_save_name = save_folder + '/' + fig_save_name + '.png'
    plt.savefig(full_save_name, bbox_inches='tight',dpi=600, pad_inches=0)
    plt.title(title)
    plt.colorbar()
    return(full_save_name)


def alpha_scatter(alpha, LitCoord2, fig_save_name, save_folder):
    
    led_position_x, led_position_y = np.meshgrid(np.arange(alpha.shape[0]), np.arange(alpha.shape[1]),
                                                 indexing='ij')

    LitCoord_reshape = np.reshape(LitCoord2, -1)
    plt.figure()
    plt.scatter(np.reshape(led_position_x,-1)[LitCoord_reshape==1], 
                np.reshape(led_position_y,-1)[LitCoord_reshape==1],
                c=np.reshape(alpha,-1)[LitCoord_reshape==1], 
                s=200, cmap='Greens', edgecolors= "black", 
                vmin=np.min(alpha), vmax=np.max(alpha))
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')
    plt.axis('off')

    full_save_name = save_folder + '/' + fig_save_name + '.png'
    plt.savefig(full_save_name, bbox_inches='tight',dpi=300, pad_inches=0)
    print('alpha saved as: ' + full_save_name)
    return(full_save_name)

def get_low_res(input_data,
                noise_level,
                single,
                obj_ind,
                normalizer,
                offset):
    
    if single:
        save_tag_mult='pnm1e'+ str(noise_level) + '_single_dm01_p4'
    else:
        save_tag_mult='pnm1e'+ str(noise_level) + '_dm01_p4'
        
    alpha = np.load(input_data + '/' + save_tag_mult + '/all_alpha_train.npy')[obj_ind]
    image_path = '{}/training/example_{:06d}'.\
                         format(input_data, obj_ind)
                         
    actual_obj = np.load(image_path + '/obj_stack.npy')                        
                         
    im_stack = np.load('{}/im_stack.npy'.format(image_path))

    im_stack_multiplexed = \
    load_multiplexed(alpha.shape[-1], # num_patterns
                     image_path,
                     save_tag_mult,
                     16, # bit_depth
                     )
    
    # remove normalization and offset
    im_stack = im_stack/normalizer
    im_stack = im_stack + offset
    
    # im_stack_multiplexed not affected by offset
    im_stack_multiplexed = im_stack_multiplexed/normalizer
    
    return(alpha, im_stack, im_stack_multiplexed, actual_obj, image_path, save_tag_mult)


def visualize_all(obj_ind,
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
                  plt_flag=False,
                  single=False,
                  force_save_path = None,
                  example_num_i = None,
                  vmin = None,
                  vmax = None,
                  ):
    
    if force_save_path is not None:
        save_path = force_save_path
    else:
        if single:
            save_path=save_tag + '_single_noise_' +\
                str(noise_level) + '_ex_' + str(num_examples) + '_p_' + str(num_patterns) # neural network path for final_train
        else:
            save_path=save_tag + '_noise_' +\
                str(noise_level) + '_ex_' + str(num_examples) + '_p_' + str(num_patterns) # neural network path for final_train
    
    if num_examples==0 and (example_num_i is not None):
        save_path += '_ex' + str(example_num_i) 
        obj_ind = example_num_i
        flag_0 = True
    else:
        flag_0 = False

    save_folder = save_path + '/final_figures' # save output figures here if plt_flag==True
    print('save_folder is: ' + save_folder)
    create_folder(save_folder)
    
    if single:
        save_tag_mult='pnm1e'+ str(noise_level) + '_single_dm01_p4'
    else:
        save_tag_mult='pnm1e'+ str(noise_level) + '_dm01_p4'

    noise_level_full=10.**noise_level

    #single led
    save_name_single_led_l1 = 'all_leds_iter_' + str(num_iter) + '_' + 'l1' + '_' + str(t2_reg) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(batch_size_opt)
        
    save_name_single_led_l2 = 'all_leds_iter_' + str(num_iter) + '_' + 'l2' + '_' + str(t2_reg) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(batch_size_opt)
        
    save_name_single_led_no_reg = 'all_leds_iter_' + str(num_iter) + '_' + 'l2' + '_' + str(0.0) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(batch_size_opt)
        
    # multiplexed
    save_name_mult_l1 = 'mult_iter_' + str(num_iter) + '_' + 'l1' + '_' + str(t2_reg) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(np.minimum(batch_size_opt,num_patterns)) +\
        '_' + save_tag_mult + '_p_' + str(num_patterns)   
        
    save_name_mult_l2 = 'mult_iter_' + str(num_iter) + '_' + 'l2' + '_' + str(t2_reg) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(np.minimum(batch_size_opt,num_patterns)) +\
        '_' + save_tag_mult + '_p_' + str(num_patterns)   
        
    save_name_mult_no_reg = 'mult_iter_' + str(num_iter) + '_' + 'l2' + '_' + str(0.0) +\
        '_pnm_' + str(noise_level_full) + '_lr_' + str(adam_learning_rate) + '_b_' + str(np.minimum(batch_size_opt,num_patterns)) +\
        '_' + save_tag_mult + '_p_' + str(num_patterns)   
        
    save_name_vec = [None,save_name_single_led_l1, save_name_single_led_l2, save_name_single_led_no_reg,\
                     save_name_mult_l1, save_name_mult_l2, save_name_mult_no_reg]
    name_vec = ['neural_network','single_led_l1', 'single_led_l2', 'single_led_no_reg',
                'mult_l1','mult_l2','mult_no_reg']
    save_name_vec_fullpath = []
    save_name_vec_fullpath_diff = [] # l1 error
    ### GET FOLDER NAME WHERE DATA IS SAVED ###
    object_name = '{}/example_{:06d}'.format(dataset_type, obj_ind)
    subfolder_name = input_data + '/' + object_name

    actual_obj = np.load(subfolder_name + '/obj_stack.npy')


    ### PLOT REFERENCE ###
    
    ref_obj = actual_obj[:,:,slice_ind]
    ref = visualize_func(actual_obj[:,:,slice_ind])
    if plt_flag:
        save_name_i = plotter(ref, 'Ground Truth', 'Ground_Truth_' + str(obj_ind), vmin, vmax, save_folder)
        save_name_vec_fullpath.append('\includegraphics[width=1in]{'+save_name_i+'}')
        save_name_vec_fullpath_diff.append('None')
        
    # vmin = np.min(ref)
    # vmax = np.max(ref)

    # vmin_abs = 0
    # vmax_abs = 1.4
    
    # vmin_ang = -np.pi/2
    # vmax_ang = np.pi/2

    compare_value_vec = [] # neural network, then the iterative solution values
    ### PLOT NEURAL NETWORK OUTPUT ###
    
    if flag_0: # deep prior
        ind_batched = 0
        batch_ind = 0
    else:
        ind_batched = obj_ind//batch_size
        batch_ind = obj_ind % batch_size
        
    try:
        all_filtered_obj = np.load(save_path + '/' + dataset_type + '/all_filtered_obj' 
                                           + str(ind_batched) +'.npy')
    except FileNotFoundError:
        all_filtered_obj = np.load(save_path + '/all_filtered_obj' 
                                           + str(ind_batched) +'.npy')
    # entropy_vec = np.load(save_path + '/' + dataset_type + '/entropy_vec' 
    #                                    + str(ind_batched) +'.npy')
    all_filtered_obj = all_filtered_obj[batch_ind,:,:,:]
    # entropy = entropy_vec[batch_ind]
    # print('Differential entropy is: ' + str(entropy))

    compare_values_all_nn = []
    for s in range(num_slices):
        angle_offset = find_angle_offset(actual_obj[:,:,s], all_filtered_obj[:,:,s])
        
        all_filtered_obj[:,:,s] = all_filtered_obj[:,:,s]*np.exp(1j*angle_offset)
        
        # output is mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity
        compare_values = \
            compare(actual_obj[:,:,s], all_filtered_obj[:,:,s])
        compare_values_all_nn.append(compare_values)
        
    compare_values_all_nn = np.stack(compare_values_all_nn)

    compare_value_vec.append(compare_values_all_nn[slice_ind,compare_val_ind])
    
    if plt_flag:
        save_name_i = plotter(visualize_func(all_filtered_obj[:,:,slice_ind]), 
                name_vec[0], name_vec[0] +'_' + str(obj_ind), vmin, vmax, save_folder)
        save_name_j = plotter(np.abs(ref_obj - all_filtered_obj[:,:,slice_ind]), 
                name_vec[0] + ' Error',  name_vec[0] + '_Output_Error' + '_' + str(obj_ind), 
                None, None, save_folder)
        
        save_name_vec_fullpath.append('\includegraphics[width=1in]{'+save_name_i+'}')
        save_name_vec_fullpath_diff.append('\includegraphics[width=1in]{'+save_name_j+'}')

    # sys.exit()
    ### PLOT COMPUTED ###

    for ind, save_name in enumerate(save_name_vec[1:]):
        computed_obj = np.load(subfolder_name + '/reconstruction/' + save_name + '_computed_obj.npy')
        computed_obj = np.transpose(computed_obj, axes=[1,2,0]) # put slice_ind last
        compare_values_all = np.load(subfolder_name + '/reconstruction/' + save_name + '_compare_values_all.npy')
        
        compare_value_vec.append(compare_values_all[slice_ind,compare_val_ind])

        if plt_flag:
            save_name_i = plotter(visualize_func(computed_obj[:,:,slice_ind]), 
                    name_vec[ind+1], name_vec[ind+1] +'_' + str(obj_ind), vmin, vmax, save_folder)
            save_name_j = plotter(np.abs(ref_obj - computed_obj[:,:,slice_ind]), 
                    name_vec[ind+1] + ' Error',  name_vec[ind+1] + '_Output_Error' +'_' + str(obj_ind), 
                    None, None, save_folder)
                    
            save_name_vec_fullpath.append('\includegraphics[width=1in]{'+save_name_i+'}')
            save_name_vec_fullpath_diff.append('\includegraphics[width=1in]{'+save_name_j+'}')
    
    return(compare_value_vec, name_vec, save_name_vec_fullpath, save_name_vec_fullpath_diff)
