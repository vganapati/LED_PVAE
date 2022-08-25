#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:49:24 2021

@author: vganapa1
"""


import sys
from zernike_polynomials import get_poly_mat
import numpy as np
import matplotlib.pyplot as plt
import argparse

from create_dataset_functions import find_fpm_params, \
                                     create_folder, \
                                     get_paddings, \
                                     find_normalizer, \
                                     NAfilter
                                     
from SyntheticMNIST_multislice_functions import scalar_prop_kernel, \
                                                process_img_multislice, \
                                                process_dataset_multislice, \
                                                create_img_stack
import tensorflow as tf

from tensorflow.keras import datasets
import tensorflow_datasets as tfds





### COMMAND LINE ARGS ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--save_path', action='store', help='path to save output')


parser.add_argument('--num_train', type=int, action='store', dest='num_train', \
                    help='number of training examples to create', default = 1000)
    
parser.add_argument('--rad', type=float, action='store', dest='LED_radius', \
                        help='radius of LEDs used on square grid LED array', default = 4)
    
parser.add_argument('--Nx', type=int, action='store', dest='Nx', \
                    help='N_obj[0], size of the object in the row direction', default = 256)
    
parser.add_argument('--Ny', type=int, action='store', dest='Ny', \
                    help='N_obj[1], size of the object in the column direction', default = 256)
        
parser.add_argument('--dti', type=int, action='store', dest='dataset_type_ind', \
                    help='dataset type, either 0 or 1. Types are: mnist or foam]', default = 0)

parser.add_argument('--ns', type=int, action='store', dest='num_slices', \
                    help='number of z slices', default = 1)
    
parser.add_argument('--uf', type=int, action='store', dest='upsample_factor', \
                    help='upsamples the camera sensor pixel size for object reconstruction by this factor', default = 2)
    
parser.add_argument('--ss', type=float, action='store', dest='slice_spacing', \
                        help='spacing between z slices in microns', default = 10)
    
parser.add_argument('--dp', type=float, action='store', dest='dpix_c', \
                        help='actual size of sensor pixels in microns', default = 6.5)
    
parser.add_argument('-f', type=float, action='store', dest='f', \
                        help='f is distance from the last slice to the focal plane', default = -10)
    
parser.add_argument('-w', type=float, action='store', dest='wavelength', \
                        help='wavelength of illumination in microns', default = 0.518)
    
parser.add_argument('--na', type=float, action='store', dest='NA', \
                        help='numerical aperture of microscope objective', default = 0.5)
    
parser.add_argument('--mag', type=float, action='store', dest='mag', \
                        help='magnification of the system ', default = 20.0)
    
parser.add_argument('--dxlx', type=float, action='store', dest='ds_led_x', \
                        help='[um] spacing between neighboring LEDs in x direction', default = 4e3)

parser.add_argument('--dxly', type=float, action='store', dest='ds_led_y', \
                        help='[um] spacing between neighboring LEDs in y direction', default = 4e3)
        
parser.add_argument('--zl', type=float, action='store', dest='z_led', \
                        help='z distance from LED array to specimen [um]', default = 69.5e3)

args = parser.parse_args()




### INPUTS ###
 
reduce_max_factor = 0.9 # scaling to prevent saturation when converting to uint16

N_obj = np.array([args.Nx, args.Ny])  # Size of the high resolution objects # [66,66]
dir_name = 'dataset_' + args.save_path

truncate_dataset = args.truncate_dataset # set to True to only process part of input dataset 
truncate_number_train = args.truncate_number_train # 60000 mnist, 50000 for cifar
truncate_number_test = args.truncate_number_test # 10000
random_transform = args.random_transform 
min_padding = 0
LED_radius = args.LED_radius #5 or 6 for DF

num_slices = args.num_slices #1 #3
slice_spacing = args.slice_spacing #10 # microns
f = args.f # 0 # -10  # f is distance from the last slice to the focal plane
dataset_type_ind = args.dataset_type_ind # must be from 0-3 inclusive

upsample_factor = args.upsample_factor # 2
dpix_c = args.dpix_c #6.5



vary_phase = False
add_poisson_noise = False 

randomize_filter = False # apply a different NA filter for each slice, maxing out at synthetic_NA

filter_obj_slices = True
different_slices = True # different base image for each slice
zernike_poly_order = 5 # polynomial order for the pupil function

wavelength = args.wavelength
NA = args.NA
mag = args.mag
ds_led_x = args.ds_led_x
ds_led_y = args.ds_led_y
z_led = args.z_led


pac = args.pac # which pupil_angle_coeff to choose


### END OF INPUTS ###

if pac == 0:
    pupil_angle_coeff = np.zeros([21,])
elif pac == 1: # coeff for 32 x 32 pixel spiral algae
    pupil_angle_coeff = np.array([-2.04910830e-07,  3.75384501e-01,  2.92857743e-01, -4.26261872e-01,
                                  3.55038938e-01, -1.91027850e-01,  1.43513249e-01,  4.56309749e-02,
                                  7.14212881e-02,  5.02212709e-02, -9.22143719e-02,  7.23976345e-03,
                                  1.55727654e-01,  6.85754873e-03,  2.87878445e-01,  1.61213051e-02,
                                  -5.43829218e-02,  9.89681249e-04,  7.96468714e-02,  3.24678290e-02,
                                  6.56303883e-02]) #coeff for 32 x 32 pixel spiral algae
elif pac == 2: # coeff for 512 x 512 pixel spiral algae 
    pupil_angle_coeff = np.array([-3.08290632e-05,  1.92146293e-01,  1.15356908e-01,  9.28136061e-02,
                                  3.46543913e-01,  9.33004907e-02, -3.08130106e-02,  3.91602512e-02,
                                  -7.11402283e-02, -3.91311711e-02,  2.14382272e-03,  5.03207900e-02,
                                  -1.00275367e-02, -6.23491586e-03, -4.61241552e-04, -9.54524156e-03,
                                  3.76899487e-03,  3.95392908e-03, -2.67817431e-02,  2.90121983e-03,
                                  2.40856322e-02]) # coeff for 512 x 512 pixel spiral algae 

dataset_types = ['mnist', 'cifar', 'cells','random','colorectal_histology', 'foam', 'squares']
dataset_input = dataset_types[dataset_type_ind]


np.random.seed(1) # make results reproducible

# Create training and testing directories

training_dir = dir_name + '/training' # name of the folder to save training examples in 
test_dir = dir_name + '/test'  # name of folder to save testing examples in

create_folder(dir_name)
create_folder(training_dir)
create_folder(test_dir)

# Find FPM parameters
    
Ns, Np, LED_vec, \
pupil, NAfilter_synthetic, \
LEDs_used_boolean, \
LitCoord, \
LED_x, LED_y, ds_led_x, ds_led_y, \
dd, NA, dpix_c, wavelength, mag, \
dx_obj, synthetic_NA, \
LED_center_x, LED_center_y, z_led = find_fpm_params(N_obj, LED_radius, 
                                                    LED_radius_inner,
                                                    upsample_factor=upsample_factor,
                                                    dpix_c = dpix_c,
                                                    wavelength = wavelength,
                                                    NA = NA,
                                                    mag = mag,
                                                    ds_led_x = ds_led_x,
                                                    ds_led_y = ds_led_y,
                                                    z_led = z_led,
                                                    )

image_x, image_y = Np

dpix_m = dpix_c/mag
x_size = Np[0]
y_size = Np[1]

zernike_mat = get_poly_mat(x_size, y_size, x_size*dpix_m, \
                           y_size*dpix_m, wavelength, NA,
                           n_upper_bound = zernike_poly_order, show_figures = False)
pupil_angle = np.sum(zernike_mat*pupil_angle_coeff, axis=2)
pupil = pupil*np.exp(1j*pupil_angle)    

np.save(dir_name + '/dx_obj.npy', dx_obj)
np.save(dir_name + '/image_x.npy', image_x)
np.save(dir_name + '/image_y.npy', image_y)
np.save(dir_name + '/image_x_r.npy', N_obj[0])
np.save(dir_name + '/image_y_r.npy', N_obj[1])
np.save(dir_name + '/LitCoord.npy', LitCoord)
np.save(dir_name + '/LED_vec', LED_vec)
np.save(dir_name + '/LEDs_used_boolean', LEDs_used_boolean)
np.save(dir_name + '/NAfilter_synthetic.npy', NAfilter_synthetic)
np.save(dir_name + '/num_leds.npy', np.sum(LEDs_used_boolean))
np.save(dir_name + '/N_obj.npy', N_obj)
np.save(dir_name + '/Np.npy', Np)
np.save(dir_name + '/Ns.npy', Ns)
np.save(dir_name + '/pupil.npy', pupil)

np.save(dir_name + '/LED_x.npy', LED_x)
np.save(dir_name + '/LED_y.npy', LED_y)
np.save(dir_name + '/ds_led_x.npy', ds_led_x)
np.save(dir_name + '/ds_led_y.npy', ds_led_y)
np.save(dir_name + '/dd.npy', dd)
np.save(dir_name + '/NA.npy', NA)
np.save(dir_name + '/dpix_c.npy', dpix_c)
np.save(dir_name + '/wavelength.npy', wavelength)
np.save(dir_name + '/mag.npy', mag)

np.save(dir_name + '/LED_center_x.npy', LED_center_x)
np.save(dir_name + '/LED_center_y.npy', LED_center_y)
np.save(dir_name + '/z_led.npy', z_led)

np.save(dir_name + '/num_slices.npy', num_slices)
np.save(dir_name + '/slice_spacing.npy', slice_spacing)
np.save(dir_name + '/f.npy', f)
np.save(dir_name + '/zernike_mat.npy', zernike_mat)

if show_figures:

    plt.figure()
    plt.title('Low Res Pupil')
    plt.imshow(np.abs(pupil))
    
    plt.figure()
    plt.title('Synthetic pupil')
    plt.imshow(NAfilter_synthetic)
    
    plt.figure()
    plt.title('Scatter plot of LED pixel shifts')
    plt.scatter(Ns[LEDs_used_boolean,0], Ns[LEDs_used_boolean,1])
    
# Read in dataset 

random_flag = False

if dataset_input == 'mnist':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset_input == 'cifar':
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = np.mean(x_train, axis=-1)
    x_test = np.mean(x_test, axis=-1)
elif dataset_input == 'cells':
    dataset = np.load('BBBC048v1.npy')
    dataset_names = np.load('BBBC048v1_file_names.npy')
    
    example_list = list(range(len(dataset)))
    np.random.shuffle(example_list)
    
    dataset = dataset[example_list, :, :]
    dataset_names = dataset_names[example_list]
    
    len_dataset = dataset.shape[0]
    num_train = int(len_dataset*.8)

    x_train = dataset[0:num_train]
    x_train_names = dataset_names[0:num_train]
    np.save(training_dir + '/x_train_names.npy', x_train_names)
    
    x_test = dataset[num_train:]
    x_test_names = dataset_names[num_train:]
    np.save(test_dir + '/x_test_names.npy', x_test_names)
    
elif dataset_input == 'random':
    # x_train = 255*np.random.rand(truncate_number_train, N_obj[0], N_obj[1])
    # x_test = 255*np.random.rand(truncate_number_test, N_obj[0], N_obj[1])
    
    x_train = 255*np.random.rand(1, N_obj[0], N_obj[1]) # placeholder
    x_test = 255*np.random.rand(1, N_obj[0], N_obj[1]) # placeholder
    random_flag = True
elif dataset_input == 'colorectal_histology':
    # https://www.tensorflow.org/datasets/catalog/colorectal_histology
    ds = tfds.load('colorectal_histology', split='train', shuffle_files=False)
    len_dataset = 5000
    num_train = int(len_dataset*.8)
    num_test = len_dataset - num_train
    
    x_train = []
    x_test = []
    for ind, example in enumerate(ds.take(len_dataset)):
        image, label = example["image"], example["label"]
        image = np.expand_dims(np.sum(image,axis=-1),axis=0)
        if ind < num_train:
            x_train.append(image)
        else:
            x_test.append(image)
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
elif dataset_input == 'foam':
    try:
        x_train = np.load('foam_training.npy')
        x_test = np.load('foam_test.npy')
    except FileNotFoundError:
        x_train = np.load('../deep_learning_CT/foam_training.npy')
        x_test = np.load('../deep_learning_CT/foam_test.npy')
    x_train = np.maximum(0,x_train)
    x_test = np.maximum(0,x_test)
    
    x_train = np.minimum(1,x_train)
    x_test = np.minimum(1,x_test)
    
    x_train *= 255
    x_test *=255
elif dataset_input == 'squares':
    x_train = np.load('squares_training.npy')
    x_test = np.load('squares_test.npy')
    
    x_train = np.maximum(0,x_train)
    x_test = np.maximum(0,x_test)
    
    x_train = np.minimum(1,x_train)
    x_test = np.minimum(1,x_test)
    
    x_train *= 255
    x_test *=255
    
else:
    print('Invalid dataset_type_ind.')
    sys.exit()

     
if truncate_dataset:
    x_train = x_train[0:truncate_number_train,:,:]
    x_test = x_test[0:truncate_number_test,:,:]
    
x_train, x_test = x_train / 255.0, x_test / 255.0



# Pad or remove pixels depending on desired object size

pad_x = int(N_obj[0] - x_train.shape[1])
pad_y = int(N_obj[1] - x_train.shape[2])

pad_x_0, pad_x_1 = get_paddings(pad_x)
pad_y_0, pad_y_1 = get_paddings(pad_y)

if pad_x<0:
    x_train = x_train[:,-pad_x_0:pad_x_1,:]
    x_test = x_test[:,-pad_x_0:pad_x_1,:]
    pad_x_0 = 0
    pad_x_1 = 0
    
if pad_y<0:
    x_train = x_train[:,:,-pad_y_0:pad_y_1]
    x_test = x_test[:,:,-pad_y_0:pad_y_1]
    pad_y_0 = 0
    pad_y_1 = 0
    
x_train = np.pad(x_train,((0,0),(pad_x_0,pad_x_1),(pad_y_0,pad_y_1)), mode = 'constant')
x_test = np.pad(x_test,((0,0),(pad_x_0,pad_x_1),(pad_y_0,pad_y_1)), mode = 'constant')

# Turn single images into img stacks

x_train_stack = create_img_stack(x_train, num_slices, different_slices = different_slices)
x_test_stack = create_img_stack(x_test, num_slices, different_slices = different_slices)



# Create scalar propagation functions

H_scalar = scalar_prop_kernel(N_obj,dx_obj,slice_spacing,wavelength)
H_scalar_f = scalar_prop_kernel(N_obj,dx_obj,f,wavelength) # scalar prop from last plane to focal plane

np.save(dir_name + '/H_scalar.npy', H_scalar)
np.save(dir_name + '/H_scalar_f.npy', H_scalar_f)


NAfilter_function = lambda synthetic_NA: NAfilter(N_obj[0], N_obj[1], N_obj[0]*dx_obj[0], \
                                                  N_obj[1]*dx_obj[1], wavelength, synthetic_NA)


# Function to process the examples into complex objects, then LED stacks
process_img_func0 = lambda img_stack: process_img_multislice(img_stack, 
                                                               NAfilter_synthetic,
                                                               N_obj, Ns, pupil, Np,
                                                               LED_vec, LEDs_used_boolean,
                                                               random_transform,
                                                               False,
                                                               num_slices,
                                                               H_scalar,
                                                               H_scalar_f,
                                                               filter_obj_slices,
                                                               random_flag,
                                                               False,
                                                               NAfilter_function,
                                                               synthetic_NA)


process_img_func = lambda img_stack: process_img_multislice(img_stack, 
                                                            NAfilter_synthetic,
                                                            N_obj, Ns, pupil, Np,
                                                            LED_vec, LEDs_used_boolean,
                                                            random_transform,
                                                            vary_phase,
                                                            num_slices,
                                                            H_scalar,
                                                            H_scalar_f,
                                                            filter_obj_slices,
                                                            random_flag,
                                                            NAfilter_function,
                                                            synthetic_NA)


if show_figures:
    low_res_stack, obj_stack = process_img_func(x_train_stack[0,:,:,:])
    plt.figure()
    plt.title('Example Low Res Image')
    plt.imshow(low_res_stack[:,:,10])
    plt.colorbar()

    plt.figure()
    plt.title('Example Obj Stack Slice Angle')
    plt.imshow(np.angle(obj_stack[:,:,0]))
    plt.colorbar()

    plt.figure()
    plt.title('Example Obj Stack Slice Abs')
    plt.imshow(np.abs(obj_stack[:,:,0]))
    plt.colorbar()    

    plt.figure()
    plt.title('Example Obj Stack Slice Real')
    plt.imshow(np.real(obj_stack[:,:,0]))
    plt.colorbar()

    plt.figure()
    plt.title('Example Obj Stack Slice Imag')
    plt.imshow(np.imag(obj_stack[:,:,0]))
    plt.colorbar()    
# Find normalizer
    
im_stack, obj_stack = process_img_func0(x_train_stack[0,:,:,:])


normalizer, offset = find_normalizer(im_stack, reduce_max_factor=reduce_max_factor, offset=0)
normalizer_re, offset_re = find_normalizer(np.real(obj_stack), reduce_max_factor=reduce_max_factor)
normalizer_im, offset_im = find_normalizer(np.imag(obj_stack), reduce_max_factor=reduce_max_factor)

normalizer_ang = [normalizer_re, normalizer_im]
offset_ang = [offset_re, offset_im]

print("normalizer is: " + str(normalizer))
# print("normalizer_ang is: " + str(normalizer_ang))
print("normalizer_re and im is: " + str([normalizer_re, normalizer_im]))

np.save(dir_name + '/normalizer.npy', normalizer)
# np.save(dir_name + '/normalizer_ang.npy', normalizer_ang)
np.save(dir_name + '/normalizer_ang.npy', [normalizer_re, normalizer_im])

np.save(dir_name + '/offset.npy', offset)
np.save(dir_name + '/offset_ang.npy', [offset_re, offset_im])

# Process and save the input images

process_dataset_multislice(x_train_stack, process_img_func, normalizer, 
                           normalizer_ang,
                           offset, offset_ang,
                           add_poisson_noise, poisson_noise_multiplier,
                           training_dir, random_flag, truncate_number_train)

process_dataset_multislice(x_test_stack, process_img_func, normalizer, 
                           normalizer_ang,
                           offset, offset_ang,
                           add_poisson_noise, poisson_noise_multiplier,
                           test_dir, random_flag, truncate_number_test)

