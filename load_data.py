#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:06:56 2021

@author: vganapa1
"""

import numpy as np
import sys
import argparse
import imageio
import matplotlib.pyplot as plt
import glob
from SyntheticMNIST_functions import create_folder

sys.path.append('../AdaptiveFourierML')
from final_visualize_v2_helper import sub_plotter_scatter, make_video
### Command Line Inputs ################

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--input_path', action='store', help='path for input dataset', \
                    default = 'dataset_frog_blood')
    
parser.add_argument('--mag', type=float, action='store', dest='mag', 
                    help='microscope magnification')

args = parser.parse_args()

################################


### Inputs ################

input_path = args.input_path

show_figures = True
create_animation = False # create an animation of all the low res images
multiplex_descriptions = ['_Random','_Dirichlet']
full_stack_inds = [0,1] # examples for which we have the full single image stack as well
num_patterns = 4 # number of patterns in each multiplex collection

background_removal = False

# Wavelength of illumination 
wavelength = 0.525 # [um] 

# Magnification of the system 
mag = args.mag

'''
https://www.edmundoptics.com/f/nikon-cfi-plan-fluor-objectives/14859/
10X	0.30 NA
20X	0.50 NA
40X	0.75 NA
'''

####
# Numerical Aperture of Objective
if mag<45 and mag>35:
    NA = 0.75 # 40x objective
elif mag<25 and mag>17:
    NA = 0.50 # 20x objective
elif mag<12 and mag>6:
    NA = 0.30 # 10x objective

# Size of pixel on sensor plane 
dpix_c = 6.5

z_led = 11.5e4 # um
                
image_x = 2048 # num pixels in the i direction
image_y = 2048 # num pixels in the j direction

# LEDs used
all_LED_num = np.arange(85)
#all_LED_num = np.concatenate((np.arange(0,118), np.arange(119,257))) # removes dead LEDs

LED_num = np.arange(85) # which of the LEDs in all_LED_num to use for further processing
num_leds = len(LED_num)

# # illumination patterns

# alpha_mat = np.zeros([num_leds, num_patterns]) # num_leds x num_patterns

# alpha_mat = np.identity(num_leds,dtype=np.float64)

# LED key
#0 - circle 0
#1-8 - circle 1
#9-24 - circle 2
#25-48 - circle 3
#49-84 - circle 4
#85-132 - circle 5
#133-188 - circle 6
#189-256 - circle 7

##########################

# Load the array parameters
led_position_list_cart = np.load(input_path + '/led_position_list_cart.npy')
circle_sizes = [1,8,16,24,36,48,56,68] # number of leds in each circle
num_circles = len(circle_sizes)
led_spacing = 6.2 # mm

if show_figures:
    plt.figure()
    plt.scatter(led_position_list_cart[:, 0], led_position_list_cart[:, 1],c=np.arange(257), cmap='gray')
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')



# reorder LEDs

led_position_xy = led_position_list_cart[:,0:2]

find_ind = lambda led_xy: np.argmin(np.sqrt(np.sum((led_position_xy-led_xy)**2,axis=1)))


center_ind = find_ind([0,0])
LED_order = [center_ind]

for i in range(1,num_circles):
    circ_rad = i*led_spacing
    num_leds_circle = circle_sizes[i]
    topmost_led_xy = [0,led_spacing*i]
    LED_order.append(find_ind(topmost_led_xy))
    circ_angle = 2*np.pi/circle_sizes[i]
    for j in range(1,circle_sizes[i]):
        angle_j = circ_angle*j + np.pi/2
        x_ij = circ_rad*np.cos(angle_j)
        y_ij = circ_rad*np.sin(angle_j)
        LED_order.append(find_ind([x_ij,y_ij]))
    
    
    
led_position_xy = led_position_xy[LED_order]
if show_figures:
    plt.figure()
    plt.scatter(led_position_xy[:, 0], led_position_xy[:, 1],c=np.arange(257), cmap='gray')
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')   


# switch to micron spacing
led_position_xy = led_position_xy*1e3

led_position_xy = led_position_xy[all_LED_num] # experiment only collects data for the LEDs in LED_num


    
##########################

# Get folder names for every object

data_file_path = input_path + '/training/example_*'
exposure_time = np.loadtxt(input_path + '/optimalExposureRound.txt')

all_folders = sorted(glob.glob(data_file_path))


exposure_time_used = exposure_time[all_LED_num]
exposure_time_used = np.expand_dims(np.expand_dims(exposure_time_used, axis=-1),axis=-1)
    
# Number of pixels of low-resolution images
dpix_m = dpix_c/mag 
    


# Save universal quantities
np.save(input_path + '/exposure_time_used.npy', exposure_time_used[LED_num])  

np.save(input_path + '/LED_num.npy', LED_num)
np.save(input_path + '/num_leds.npy', num_leds) # number of leds in LED_num
np.save(input_path + '/all_LED_num.npy', all_LED_num)

np.save(input_path + '/z_led.npy', z_led)
np.save(input_path + '/wavelength.npy', wavelength)

np.save(input_path + '/dpix_c.npy', dpix_c)
np.save(input_path + '/mag.npy', mag)    
np.save(input_path + '/NA.npy', NA)

# spacing in the low-res image
np.save(input_path + '/dpix_m.npy', dpix_m)

# size of low res image
np.save(input_path + '/image_x.npy', image_x)
np.save(input_path + '/image_y.npy', image_y)


#

all_alpha_train = np.zeros([len(multiplex_descriptions),len(all_folders), num_leds, num_patterns]) #tf.ones([len(train_folders),num_leds,num_patterns], dtype=tf.float32)/num_leds

for obj_ind, folder in enumerate(all_folders):
    
    if obj_ind in full_stack_inds:
        # Read in reference
        
        if background_removal:
            black_img =imageio.imread(folder + '/Reference.png')
            Ibk_0 = np.mean(black_img)/float(2**16-1)/exposure_time[0]
    
    
        # Read in images
        
        img_stack = np.zeros([len(all_LED_num),2048,2048])
    
        for ind,led in enumerate(all_LED_num):
                
            img = imageio.imread(folder + '/Photo{:04d}.png'.format(led) )
            img = np.rot90(img,3)/float(2**16-1)/exposure_time_used[ind] # rotate to be facing viewer and normalize to 1
            
            if background_removal:
                img_stack[ind,:,:] = np.maximum(img-Ibk_0, 0)
            else:
                img_stack[ind,:,:] = img    
        
        
        if show_figures:
            
            plt.figure()
            plt.title('First image in the single LED stack')
            plt.imshow(img_stack[0,:,:])
            
    
        np.save(folder + '/im_stack.npy', img_stack[LED_num])
        

    # Load multiplexed 
    for mult_ind, multiplex_description in enumerate(multiplex_descriptions):
        with open(folder + '/Multiplex' + multiplex_description +'/LED_lists.txt') as f:
            lines = f.readlines() # all the LED patterns for this particular example
        
        alpha_mat = []
        for line in lines:
            print(line)
            line1 = line[2:-1] # remove l. in the front and carriage return at end
            leds = np.array(line1.split('.'))
            leds = leds.astype(np.int32)
            alpha = np.zeros(num_leds)
            alpha[leds] = 1
            alpha_mat.append(alpha)
        alpha_mat = np.stack(alpha_mat,axis=-1)
        all_alpha_train[mult_ind,obj_ind,:,:] = alpha_mat
        
        '''
        # Subtract black image from multiplexed
        all_files_multiplex = sorted(glob.glob(folder + '/Multiplex/*.png'))
    
        Ibk_mult = ?
        for file_mult in all_files_multiplex:
            img_mult = imageio.imread(file_mult)
            img_mult -= Ibk_mult
            # save as uint16 png
        '''
    
create_folder(input_path + '/real_multiplexed')
for mult_ind,multiplex_description in enumerate(multiplex_descriptions):
    np.save(input_path + '/real_multiplexed/all_alpha_train' + multiplex_description + '.npy', all_alpha_train[mult_ind])


# Create animation on the last stack loaded in
if create_animation:
    filenames=[]
    for led_i in LED_num:
        print(led_i)
        fig_save_name = 'real_led_' + str(led_i)
        out_filepath =\
            sub_plotter_scatter(img_stack[led_i]*np.squeeze(exposure_time_used[led_i]), 
                                led_position_xy[led_i]/1e3, fig_save_name, fig_save_name, 
                                0, 1, input_path, 
                                cmap='gray',
                                )
            
        filenames.append(out_filepath)
    make_video(filenames, 
               'real_data',
               remove_files=True, loop=1, fps=10)


# Find the center of the LED array by looking for the brightest point by a weighted average
# This is completed for the last image stack read in
# XXX replace by a stack with no slide

# Following uses average intensities of img_stack
#Make XY, and Z arrays and list
Z = np.mean(np.mean(img_stack,axis=-1), axis=-1)


LED_center_x = np.sum(led_position_xy[:,0]*Z)/np.sum(Z)
LED_center_y = np.sum(led_position_xy[:,1]*Z)/np.sum(Z)

print(LED_center_x)
print(LED_center_y)


if show_figures:
    plt.figure()
    plt.scatter(led_position_xy[:,0], led_position_xy[:,1],c=Z, cmap='rainbow')
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.axis('square')

# center the LED array, relative to the images

led_position_xy[:,0] -= LED_center_x
led_position_xy[:,1] -= LED_center_y



'''
### average for blank img_stack
im_stack_ave = np.mean(np.mean(img_stack[:,:,500:],axis=-1), axis=-1)
'''

'''
img_stack -= blank_img_stack
img_stack += np.expand_dims(np.expand_dims(im_stack_ave, axis=-1), axis=-1)


img_stack = np.maximum(img_stack,0)

'''

# Save universal quantities
np.save(input_path + '/led_position_xy.npy', led_position_xy[LED_num])

