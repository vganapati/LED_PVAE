#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:53:05 2020

@author: vganapa1
"""

import numpy as np
from helper_functions import create_alpha_mat

try:
    import matplotlib.pyplot as plt
    plt_flag = True
except ImportError:
    plt_flag = False


def analyze_multiopt(training_opts_vec, dataset_type, worst_case = False):
    results = []
    for training_opt in training_opts_vec:
        merit_vec = np.load(training_opt + '/MSE_final_' + dataset_type + '.npy')
        if worst_case:
            merit_vec = np.sort(merit_vec)
            result = np.mean(merit_vec[-1000:])
        else:
            result = np.mean(merit_vec)
        results.append(result)
    return results

def optical_element_transform(optical_element, LitCoord2):    
    LED_mat = np.zeros(LitCoord2.shape)
    LED_mat[np.nonzero(LitCoord2)]=optical_element
    return LED_mat  

def show_fig_alpha(save_path, 
                   alpha,
                   LitCoord,
                   oe_ind = 0):
        

    LED_mat = optical_element_transform(alpha.numpy()[:,oe_ind], LitCoord)
    
    if plt_flag:
        plt.figure()
        plt.title('LED mat ' + str(oe_ind))
        LED_mat[np.nonzero(LED_mat==0)] = np.nan
        plt.imshow(LED_mat)
        plt.colorbar()
        plt.savefig(save_path + '/LED_mat_' + str(oe_ind) + '.png')


def show_fig_alpha2(alpha,
                    LitCoord,
                   ):
        
    LED_mat = optical_element_transform(alpha.numpy(), LitCoord)
    if plt_flag:
            plt.figure()
            plt.title('LED mat')
            LED_mat[np.nonzero(LED_mat==0)] = np.nan
            plt.imshow(LED_mat)
            plt.colorbar()
        
def make_box(pixels_x,
             pixels_y,
             pixel_x_start, # patch inclusive of start value
             pixel_x_stop, # patch exclusive of stop value
             pixel_y_start,
             pixel_y_stop,
             line_thickness = 2):
    
    box = np.zeros([pixels_x,pixels_y,4])
    box[pixel_x_start-line_thickness:pixel_x_stop+line_thickness,\
        pixel_y_start-line_thickness:pixel_y_start, 3] = 1
    box[pixel_x_start-line_thickness:pixel_x_stop+line_thickness,\
        pixel_y_stop:pixel_y_stop+line_thickness, 3] = 1
    box[pixel_x_start-line_thickness:pixel_x_start,\
        pixel_y_start-line_thickness:pixel_y_stop+line_thickness, 3] = 1
    box[pixel_x_stop:pixel_x_stop+line_thickness,\
        pixel_y_start-line_thickness:pixel_y_stop+line_thickness, 3] = 1
    
    return box


def show_figs_pixel_map(save_path,
                        im_stack,
                        LitCoord2,
                        pixel_x_start, # patch inclusive of start value
                        pixel_x_stop, # patch exclusive of stop value
                        pixel_y_start,
                        pixel_y_stop,
                        name_tag,
                        data_folder,
                        example_num,
                        iter_ind,
                        img_ind,
                        batch_ind = 0):

    
    pixels_x = im_stack.shape[1]
    pixels_y = im_stack.shape[2]
    
    box = make_box(pixels_x,
                   pixels_y,
                   pixel_x_start, # patch inclusive of start value
                   pixel_x_stop, # patch exclusive of stop value
                   pixel_y_start,
                   pixel_y_stop)
    
    if plt_flag:
        plt.figure()
        plt.title('Pixel Patch')
        plt.imshow(np.sum(im_stack[batch_ind,:,:,:], axis=-1))
        plt.colorbar()
        plt.imshow(box)
        plt.savefig(save_path + '/pixel_patch_' + name_tag + '_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png') 
    
    im_stack_patch = \
    im_stack[batch_ind, \
             pixel_x_start:pixel_x_stop, \
             pixel_y_start:pixel_y_stop, \
             :]
    
    
    im_stack_patch = np.squeeze(im_stack_patch)
    im_stack_patch = np.mean(im_stack_patch, axis = 0)
    im_stack_patch = np.mean(im_stack_patch, axis = 0)
    pixel_mat = optical_element_transform(im_stack_patch, LitCoord2)
    pixel_mat[np.nonzero(pixel_mat==0)] = np.nan
    
    if plt_flag:
        plt.figure()
        plt.title('Pixel patch average value')
        plt.imshow(pixel_mat)
        plt.colorbar()
        plt.savefig(save_path + '/pixel_mat_' + name_tag + '_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')    
    
def show_figs2(save_path, 
               iter_vec,
               train_loss_vec,
               ):
    
    # train_loss_vec = np.array(train_loss_vec)[:,-1]
    # val_loss_vec = np.array(val_loss_vec)[:,-1]
    
    if plt_flag:
        
        plt.figure()
        plt.title('Training loss')
        plt.plot(train_loss_vec)
        plt.savefig(save_path + '/train_loss_vec.png')
    



def show_fig_alpha_loc_probs(save_path, 
                             alpha_loc_probs_vec,
                             example_num,
                             pattern_ind,
                             data_folder,
                             batch_ind=0):
    
    
    plt.figure()
    plt.title('alpha_loc_probs')
    
    for ii in range(alpha_loc_probs_vec.shape[0]):
        plt.plot(alpha_loc_probs_vec[ii,batch_ind,pattern_ind,:],label=ii)
    plt.legend()
    plt.savefig(save_path + '/alpha_loc_probs' + \
                    data_folder + '_example_' + str(example_num) + '_pattern_' + str(pattern_ind) + '.png')  
        

def show_figs_a(save_path, 
                a,
                batch_size_per_gpu,
                data_folder,
                example_num,
                iter_ind,
                pattern_ind,
                num_leds,
                num_patterns,
                LitCoord2,
                batch_ind=0):
        
    a_mat = \
    create_alpha_mat(a, 
                     batch_size_per_gpu,
                     num_leds,
                     num_patterns,
                     LitCoord2)
    a_mat = a_mat[batch_ind,:,:,pattern_ind]
    a_mat = a_mat.numpy()
    a_mat[np.nonzero(a_mat==0)] = np.nan
    if plt_flag:
        plt.figure()
        plt.title('a_mat')
        plt.imshow(a_mat)
        plt.colorbar()
        plt.savefig(save_path + '/a_mat' + \
                    data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_pattern_' + str(pattern_ind) + '.png')  
        
        

def show_alpha_scatter(led_position_xy, alpha, im_stack_multiplexed):
    plt.figure()
    plt.title('LED Illumination Pattern')
    plt.scatter(led_position_xy[:,0], led_position_xy[:,1],c=alpha, s=100, cmap='Greens', edgecolors= "black", 
                vmin=np.min(alpha), vmax=1.5*np.max(alpha))
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')
    # plt.axis('off')

    plt.figure()
    plt.title('im_stack_multiplexed')
    plt.imshow(im_stack_multiplexed)
    plt.colorbar()

def show_figs_alpha(save_path, 
                    alpha_sample,
                    batch_size_per_gpu,
                    im_stack_multiplexed,
                    data_folder,
                    example_num,
                    iter_ind,
                    pattern_ind,
                    num_leds,
                    num_patterns,
                    LitCoord2,
                    batch_ind=0):
        
    alpha_mat = \
    create_alpha_mat(alpha_sample, 
                     batch_size_per_gpu,
                     num_leds,
                     num_patterns,
                     LitCoord2)
    alpha_mat = alpha_mat[batch_ind,:,:,pattern_ind]
    alpha_mat = alpha_mat.numpy()
    alpha_mat[np.nonzero(alpha_mat==0)] = np.nan
    im_stack_multiplexed = im_stack_multiplexed[batch_ind,:,:,pattern_ind]  

    if plt_flag:
        plt.figure()
        plt.title('alpha_mat')
        plt.imshow(alpha_mat)
        plt.colorbar()
        plt.savefig(save_path + '/alpha_mat' + \
                    data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_pattern_' + str(pattern_ind) + '.png')  
        
        plt.figure()
        plt.title('im_stack_multiplexed')
        plt.imshow(im_stack_multiplexed)
        plt.colorbar()
        plt.savefig(save_path + '/im_stack_multiplexed_' + \
                    data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_pattern_' + str(pattern_ind) + '.png')  
            

def show_figs_input_output(save_path, 
                           data_folder,
                           im_stack,
                           output,
                           batch_ind,
                           img_ind):

    input_fig = im_stack[batch_ind,:,:,img_ind]
    output_fig = output[batch_ind,:,:,img_ind]

    vmin = np.min(input_fig)
    vmax = np.max(input_fig)
    
    if plt_flag:
        plt.figure()
        plt.title('input fig')
        plt.imshow(input_fig, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.savefig(save_path + '/input_fig_' + \
                    data_folder + '_img_ind_' + str(img_ind) + '_batch_ind_' + str(batch_ind) + '.png')  
        
        plt.figure()
        plt.title('output fig')
        plt.imshow(output_fig, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.savefig(save_path + '/output_fig_' + \
                    data_folder + '_img_ind_' + str(img_ind) + '_batch_ind_' + str(batch_ind) + '.png')  
            
        
            
        
def show_figs(save_path, 
              im_stack,
              im_stack_mean,
              im_stack_var,
              data_folder,
              example_num,
              iter_ind,
              img_ind,
              batch_ind = 0):

        if plt_flag:
            x,y = im_stack_mean[batch_ind,:,:,0].shape
            x = int(0.75*x)
            y = int(0.75*y)
            plt.figure()
            plt.title('Pixel comparison 1')
            plt.plot(im_stack_mean[batch_ind,x,y,:])
            plt.plot(im_stack[batch_ind,x,y,:])
            plt.savefig(save_path + '/line_compare0_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')
            
    
            x,y = im_stack_mean[batch_ind,:,:,0].shape
            x = int(0.25*x)
            y = int(0.25*y)  
            plt.figure()
            plt.title('Pixel comparison 2')
            plt.plot(im_stack_mean[batch_ind,x,y,:],label='computed')
            plt.plot(im_stack[batch_ind, x,y,:], label = 'actual')
            plt.legend()
            plt.savefig(save_path + '/line_compare1_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')
            
            actual_guess = np.concatenate((im_stack[batch_ind,:,:,img_ind],im_stack_mean[batch_ind,:,:,img_ind]),axis=1)
            plt.figure(figsize=[10,10])
            plt.title('Actual and guess comparison')
            plt.imshow(actual_guess, vmin=0, vmax=1) #vmax=np.max(im_stack[batch_ind,:,:,img_ind])
            plt.savefig(save_path + '/actual_guess_comparison_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')

                
            plt.figure()
            plt.title('im_stack actual')
            plt.imshow(im_stack[batch_ind,:,:,img_ind])
            plt.colorbar()
            plt.savefig(save_path + '/im_stack_actual_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')  
                
                
            plt.figure()
            plt.title('im_stack_mean')
            plt.imshow(im_stack_mean[batch_ind,:,:,img_ind])
            plt.colorbar()
            plt.savefig(save_path + '/guess_mean_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')  
            
            plt.figure()
            plt.title('im_stack_var')
            plt.imshow(im_stack_var[batch_ind,:,:,img_ind])
            plt.colorbar()
            plt.savefig(save_path + '/guess_var_' + \
                        data_folder + '_example_' + str(example_num) + '_iter_' + str(iter_ind) + '_LED_' + str(img_ind) + '.png')    
