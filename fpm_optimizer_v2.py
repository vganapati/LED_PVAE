#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:10:15 2021

@author: vganapa1
"""

import sys

from fpm_functions import F, Ft, create_low_res_stack_multislice, \
    find_Ns, NAfilter, \
    scalar_prop_kernel, shift_add
from SyntheticMNIST_functions import create_folder
import tensorflow as tf
import numpy as np
import time
import argparse
from zernike_polynomials import get_poly_mat
import matplotlib.pyplot as plt
from helper_functions import physical_preprocess, create_window
from helper_pattern_opt import load_multiplexed
import skimage.transform

### Command line args ###

parser = argparse.ArgumentParser(description='Get command line args')


parser.add_argument('--input_path', action='store', help='path for input dataset', \
                    default = 'dataset_frog_blood_mult')
    
    
parser.add_argument('--save_tag_recons', action='store', help='path for results is /reconstruction_save_name', \
                    default = '')
    
parser.add_argument('--obj_ind', type=int, 
                    action='store', dest='obj_ind',
                    help='example number to analyze', 
                    default = '0')
    
parser.add_argument('-i', type=int, action='store', dest='num_iter', \
                        help='number of iterations', default = 10)

parser.add_argument('-b', type=int, action='store', dest='batch_size', \
                    help='batch_size', default = 85)

parser.add_argument('-p', type=int, action='store', dest='num_patterns', \
                    help='num_patterns when using the multiplexed option', default = None)    
    
parser.add_argument('--t2', type=float, action='store', dest='t2_reg', 
                        help='t2 regularization', default = 1e-3)

parser.add_argument('--alr', type=float, action='store', dest='adam_learning_rate', 
                        help='learning rate for adam optimizer', default = 1e-2)

parser.add_argument('--ae', type=float, action='store', dest='adam_epsilon', 
                        help='adam_epsilon', default = 1e-7)

parser.add_argument('--l1', action='store_true', dest='projected_grad', 
                    help='l1 regularization with projected gradient descent')

parser.add_argument('--mult', action='store_true', dest='multiplexed', 
                    help='use only multiplexed images to reconstruct')

parser.add_argument('--save_tag', action='store', 
                    help='save_tag for the multiplexed images', 
                    default = None)

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier', 
                    help='poisson noise multiplier, higher value means higher SNR, \
                        only used for synthetic data', default = (2**16-1)*0.41)
    
parser.add_argument('--real_data', action='store_true', dest='real_data', 
                    help='uses real data for the image stacks') 

parser.add_argument('--real_mult', action='store_true', dest='real_mult', 
                    help='uses real data for the MULTIPLEXED image stacks') 

parser.add_argument('--xcorner', type=int, action='store', dest='x_corner', \
                    help='corner coords of patch', default = 1200)
        
parser.add_argument('--ycorner', type=int, action='store', dest='y_corner', \
                    help='corner coords of patch', default = 1220)

parser.add_argument('--xcrop', type=int, action='store', dest='x_crop_size', \
                    help='patch size to consider in reconstruction', default = 512)
        
parser.add_argument('--ycrop', type=int, action='store', dest='y_crop_size', \
                    help='patch size to consider in reconstruction', default = 512)

parser.add_argument('--uf', type=int, action='store', dest='upsample_factor', \
                    help='High resolution object pixels = collected image pixels * upsample_factor, only used for real_data', 
                    default = 1)
        
# following 3 arguments only used for real data
parser.add_argument('--num_slices', type=int, action='store', dest='num_slices', \
                    help='num z slices', default = 1)
        
parser.add_argument('--slice_spacing', type=float, action='store', dest='slice_spacing', \
                    help='slice_spacing in um', default = 0)
            
parser.add_argument('--focal_dist', type=float, action='store', dest='f', \
                    help='distance from the focal plane to the last slice in um', default = 0)

parser.add_argument('--md', dest = 'multiplexed_description',
                    action='store', help='description of multiplex type', default = '') # _Dirichlet or _Random

parser.add_argument('--ones', action='store_true', dest='initialize_ones', 
                    help='initial condition is a perfectly transparent object, no RI contrast')

parser.add_argument('--window', action='store_true', dest='use_window', 
                    help='use a windowing function to help eliminate edge artifacts')

parser.add_argument('--stochastic', action='store_true', dest='stochastic', 
                    help='stochastic training loop')

args = parser.parse_args()


#########################

### Parse command line args ###
input_path = args.input_path
obj_ind = args.obj_ind
num_iter = args.num_iter
adam_learning_rate = args.adam_learning_rate
adam_epsilon = args.adam_epsilon
batch_size = args.batch_size
multiplexed = args.multiplexed # reconstruct with only multiplexed low res images
num_patterns = args.num_patterns
t2_reg = args.t2_reg
projected_grad = args.projected_grad # l1 regularization with projected gradient descent
poisson_noise_multiplier = args.poisson_noise_multiplier
real_data = args.real_data
save_tag = args.save_tag
real_mult = args.real_mult
multiplexed_description = args.multiplexed_description

# following only used for real data
x_corner = args.x_corner 
y_corner = args.y_corner
x_crop_size = args.x_crop_size
y_crop_size = args.y_crop_size


if multiplexed:
    if batch_size > num_patterns:
        batch_size = num_patterns
        print('reduced batch size to num_patterns')


# Other Inputs
dataset_type = 'training'
optimize_pupil_ang = True
change_Ns = False # If True, optimizes LED positions
zernike_poly_order = 5
filter_hr = True
save_recons = True
sqrt_reg = np.finfo(np.float32).eps.item()
plt_flag = True # displays plots if True
initialize_ones = args.initialize_ones #True
use_window = args.use_window #False
stochastic = args.stochastic #False
visualize_trim = 64


# only used for real data
# High resolution object pixels = collected image pixels * upsample_factor
upsample_factor = args.upsample_factor

if real_data:
    # multislice parameters in um
    num_slices = args.num_slices
    slice_spacing = args.slice_spacing
    f = args.f # f is distance from the focal plane to the last slice
else:
    num_slices = np.load(input_path + '/num_slices.npy')
    slice_spacing = np.load(input_path + '/slice_spacing.npy')
    f = np.load(input_path + '/f.npy')

### LOAD DATA ###
#############################
# load parameters

folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)

lr_observed_stack = np.load(folder_name + '/im_stack.npy') # low-res stack
if real_data:
    led_position_xy = np.load(input_path + '/led_position_xy.npy')
    num_leds = len(np.load(input_path + '/LED_num.npy')) # LEDs that are used
else:
    num_leds = np.load(input_path + '/num_leds.npy')
    
    
z_led = np.load(input_path + '/z_led.npy')
wavelength = np.load(input_path + '/wavelength.npy')

dpix_c = np.load(input_path + '/dpix_c.npy')
mag = np.load(input_path + '/mag.npy')    
NA = np.load(input_path + '/NA.npy')

# spacing in the low-res image
dpix_m = np.load(input_path + '/dpix_m.npy')

# size of low-res image

image_x = np.load(input_path + '/image_x.npy')
image_y = np.load(input_path + '/image_y.npy')


'''
window_2d is for multiplying the low_res stack
window_2d_sqrt is for multiplying the high res stack
'''

if use_window:
    window_2d = create_window(x_crop_size, y_crop_size)
    # window_2d = window_2d**2
else:
    window_2d = np.ones([x_crop_size,y_crop_size])
window_2d_sqrt = np.sqrt(window_2d)
    
# upsampled window
window_2d_sqrt_us = skimage.transform.rescale(window_2d_sqrt, 
                                              upsample_factor, multichannel = False, order = 0, mode = 'constant')

#############################

object_name = '{}/example_{:06d}'.format(dataset_type, obj_ind)

if multiplexed:
    multiplexed_stack = \
    load_multiplexed(num_patterns,
                     folder_name,
                     save_tag,
                     bit_depth=16,
                     real_mult=real_mult,
                     dtype=tf.float64,
                     multiplexed_description=multiplexed_description)   

    multiplexed_stack = multiplexed_stack.numpy()
    
    # put num_patterns first
    multiplexed_stack = np.transpose(multiplexed_stack,[2,0,1])
    
    if real_data:
        multiplexed_stack = multiplexed_stack[:,x_corner:x_corner+x_crop_size,
                                              y_corner:y_corner+y_crop_size,
                                              ]
    else:
        multiplexed_stack[multiplexed_stack<0] = 0
        multiplexed_stack[multiplexed_stack>1] = 1
    
    alpha = np.load(input_path + '/' + save_tag + '/all_alpha_train' + multiplexed_description + '.npy')[obj_ind,:,0:num_patterns].astype(np.float64)
    alpha_expand = tf.cast(tf.expand_dims(tf.expand_dims(alpha, axis=1), axis=1), tf.float64)

if real_data:
    exposure_time_used = np.load(input_path + '/exposure_time_used.npy')
else:
    normalizer = np.load(input_path + '/normalizer.npy')
    offset = np.load(input_path + '/offset.npy')
    exposure_time_used = np.ones([num_leds,1,1])
    # actual low-res stack, no noise
    lr_observed_stack[lr_observed_stack<0] = 0
    lr_observed_stack[lr_observed_stack>1] = 1

    # add noise to lr_observed_stack
    alpha_identity = np.expand_dims(np.identity(num_leds).astype(np.float64),axis=0)
    lr_observed_stack_noise = physical_preprocess(tf.cast(tf.expand_dims(tf.expand_dims(lr_observed_stack, 0),0), tf.float64),
                                                  tf.expand_dims(alpha_identity,0),
                                                  poisson_noise_multiplier,
                                                  sqrt_reg,
                                                  1, # batch_size
                                                  1, # max_steps
                                                  True, #renorm; doesn't matter if offset ==0 # True if need to remove normalization and offset
                                                  normalizer=normalizer,
                                                  offset=offset,
                                                  zero_alpha=False,
                                                  return_dist = False,
                                                  quantize_noise=False,
                                                  )
    
    lr_observed_stack_noise = np.squeeze(lr_observed_stack_noise)
    # lr_observed_stack_noise = lr_observed_stack  # Uncomment to remove effects of noise
    lr_observed_stack_noise = lr_observed_stack_noise/normalizer + offset
    
    lr_observed_stack = tf.transpose(lr_observed_stack_noise, [2,0,1])
    if multiplexed:
        multiplexed_stack = multiplexed_stack/normalizer



#############################

# coordinates in um
img_coords_x = dpix_m*(np.arange(image_x) - image_x/2)
img_coords_y = dpix_m*(np.arange(image_y) - image_y/2)

img_coords_xm, img_coords_ym = np.meshgrid(img_coords_x,img_coords_y, indexing='ij')
    
if real_data:
    
    # crop lr_observed_stack and img_coords
    lr_observed_stack = lr_observed_stack[:, x_corner:x_corner + x_crop_size, y_corner:y_corner + y_crop_size]
    img_coords_xm = img_coords_xm[x_corner:x_corner + x_crop_size, y_corner:y_corner + y_crop_size]
    img_coords_ym = img_coords_ym[x_corner:x_corner + x_crop_size, y_corner:y_corner + y_crop_size]
    
    
    
    Ns_0, pupil, synthetic_NA, cos_theta = find_Ns(img_coords_xm,
                                                   img_coords_ym,
                                                   led_position_xy,
                                                   dpix_m,
                                                   z_led,
                                                   wavelength,
                                                   NA,
                                                   )  
    
    
    # cos_theta**4 dropoff
    # lr_observed_stack /= np.expand_dims(np.expand_dims(cos_theta**4, axis=1),axis=1)
    cos_theta_dropoff=np.expand_dims(np.expand_dims(cos_theta**4, axis=1),axis=1)
    cos_theta_dropoff = tf.constant(cos_theta_dropoff)
    
    Ns = tf.Variable(Ns_0)
    
    Np=np.array([x_crop_size, y_crop_size])
    N_obj = Np*upsample_factor
    pupil = pupil.astype(np.complex128)
    
    dx_obj = dpix_m/upsample_factor
    dx_obj = [dx_obj,dx_obj]
    NAfilter_synthetic = NAfilter(N_obj[0], N_obj[1], N_obj[0]*dx_obj[0], \
                                  N_obj[1]*dx_obj[1], wavelength, synthetic_NA)
    
    if plt_flag:
        plt.figure()
        plt.title('NA filter synthetic')
        plt.imshow(NAfilter_synthetic)
    
    H_scalar = scalar_prop_kernel(N_obj,dx_obj,slice_spacing,wavelength)
    H_scalar_f = scalar_prop_kernel(N_obj,dx_obj,f,wavelength) # scalar prop from last plane to focal plane
else:
    N_obj = np.load(input_path + '/N_obj.npy')
    Np = np.load(input_path + '/Np.npy')
    upsample_factor = int(N_obj[0]/Np[0])
    Ns_0 = np.load(input_path + '/Ns.npy')
    pupil = np.load(input_path + '/pupil.npy').astype(np.complex128)
    LED_vec = np.load(input_path + '/LED_vec.npy')
    LEDs_used_boolean = np.load(input_path + '/LEDs_used_boolean.npy')
    NAfilter_synthetic = np.load(input_path + '/NAfilter_synthetic.npy')
    LitCoord = np.load(input_path + '/LitCoord.npy')
    
    LED_x = np.load(input_path + '/LED_x.npy')
    LED_y = np.load(input_path + '/LED_y.npy', )
    ds_led_x = np.load(input_path + '/ds_led_x.npy')
    ds_led_y = np.load(input_path + '/ds_led_y.npy')
    dd = np.load(input_path + '/dd.npy')
    NA = np.load(input_path + '/NA.npy')
    dpix_c = np.load(input_path + '/dpix_c.npy')
    wavelength = np.load(input_path + '/wavelength.npy')
    mag = np.load(input_path + '/mag.npy')
    
    LED_center_x = np.load(input_path + '/LED_center_x.npy')
    LED_center_y = np.load(input_path + '/LED_center_y.npy')
    z_led = np.load(input_path + '/z_led.npy')
    
    num_slices = np.load(input_path + '/num_slices.npy')
    slice_spacing = np.load(input_path + '/slice_spacing.npy')
    f = np.load(input_path + '/f.npy')
    
    H_scalar = np.load(input_path + '/H_scalar.npy')
    H_scalar_f  = np.load(input_path + '/H_scalar_f.npy')

    Ns_0 = Ns_0[LEDs_used_boolean]
    Ns = tf.Variable(Ns_0)
    LED_x = ds_led_x*LED_x[LEDs_used_boolean]
    LED_y = ds_led_y*LED_y[LEDs_used_boolean]
    
    led_position_xy= np.stack((LED_x,LED_y),axis=1)

dpix_m = dpix_c/mag
x_size = Np[0]
y_size = Np[1]

if real_data:
    zernike_mat = get_poly_mat(x_crop_size, y_crop_size, x_crop_size*dpix_m, \
                               y_crop_size*dpix_m, wavelength, NA,
                               n_upper_bound = zernike_poly_order, show_figures = False)
else:
    zernike_mat = get_poly_mat(image_x, image_y, image_x*dpix_m, \
                               image_y*dpix_m, wavelength, NA,
                               n_upper_bound = zernike_poly_order, show_figures = False)
    
    
if multiplexed:
    multiplexed_stack = multiplexed_stack*window_2d
lr_observed_stack = lr_observed_stack*window_2d

### Initial Guess

zmin = f-(num_slices-1)*slice_spacing
zmax = f + slice_spacing 
dz = slice_spacing

if num_slices==1:
    z_vec = np.array([f])
else:
    z_vec = np.arange(zmin,zmax,dz)

if multiplexed:
    # lr_observed_stack is unavailable
    # minimum norm
    # Ax = y
    if real_mult:
        A = alpha.T
    else:
        A = (alpha*np.squeeze(exposure_time_used, -1)).T
    A_inv = np.linalg.pinv(A)
    multiplexed_stack_i = np.transpose(np.expand_dims(multiplexed_stack, 1), [2,3,0,1])
    lr_observed_stack_emulated = A_inv @ multiplexed_stack_i
    lr_observed_stack_emulated = np.squeeze(lr_observed_stack_emulated,-1)
    lr_observed_stack_emulated = np.transpose(lr_observed_stack_emulated,[2,0,1])
    # lr_observed_stack_emulated = lr_observed_stack_emulated/exposure_time_used

    '''
    ind = 10
    plt.figure()
    plt.imshow(lr_observed_stack[ind,:,:])
    
    vmin = np.min(lr_observed_stack[ind,:,:])
    vmax = np.max(lr_observed_stack[ind,:,:])
    
    plt.figure()
    plt.imshow(lr_observed_stack_emulated[ind,:,:], vmin=vmin, vmax=vmax)
    '''
    
    '''
    # lr_observed_stack_i = lr_observed_stack_emulated
    lr_observed_stack_i = lr_observed_stack
    lr_observed_stack_i = np.transpose(np.expand_dims(lr_observed_stack_i, 1), [2,3,0,1])
    multiplexed_stack_emulated = A @ lr_observed_stack_i

    multiplexed_stack_emulated = np.squeeze(multiplexed_stack_emulated,-1)
    multiplexed_stack_emulated = np.transpose(multiplexed_stack_emulated,[2,0,1])

    ind = 0
    plt.figure()
    plt.imshow(multiplexed_stack[ind])

    vmin = np.min(multiplexed_stack[ind])
    vmax = np.max(multiplexed_stack[ind])

    plt.figure()
    plt.imshow(multiplexed_stack_emulated[ind], vmin=vmin, vmax=vmax)
    
    '''
else:
    lr_observed_stack_emulated = lr_observed_stack

initial_amplitude_mat, initial_phase_mat, tot_mat = \
    shift_add(lr_observed_stack_emulated, Np, img_coords_xm,
              img_coords_ym, led_position_xy, NA,
              wavelength, dpix_m, z_led, 
              upsample_factor,
              z_vec)
'''
plt.figure()
plt.imshow(initial_amplitude_mat[0])
plt.colorbar()

plt.figure()
plt.imshow(initial_phase_mat[0])
plt.colorbar()
'''
    
obj_stack_init = initial_amplitude_mat*np.exp(1j*initial_phase_mat)
if initialize_ones:
    obj_stack_init = 0.005*tf.ones_like(obj_stack_init)

# need to include pupil for the following
# obj_stack_init = tf.expand_dims(np.load('iter_reconstruction_patch.npy'), axis=0)
# obj_stack_init = tf.expand_dims(np.load('nn_reconstruction_patch.npy'), axis=0)

hr_guess = tf.Variable(obj_stack_init, dtype=tf.complex128)
pupil_angle_coeff = tf.Variable(np.zeros([zernike_mat.shape[-1],]))
optimizer = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate, \
                                     epsilon=adam_epsilon)

if real_data:
    pass
else:
    # distance of LED to center
    LED_dist = np.sqrt(LED_x**2 + LED_y**2)   
    LED_dist_ind = np.argsort(LED_dist)

    # XXX REARRANGE IN DIST ORDER
    


def func(LED_vec_i, batch_vec = None):
    pupil_angle = tf.cast(tf.reduce_sum(zernike_mat*pupil_angle_coeff, axis=2), tf.complex128)
    
    if multiplexed:
        lr_calc_stack = \
            create_low_res_stack_multislice(hr_guess, N_obj, Ns, \
                                             pupil*tf.exp(1j*pupil_angle), Np, LED_vec_i, \
                                             num_slices, H_scalar, H_scalar_f, num_leds, change_Ns, 
                                             use_window, window_2d_sqrt_us)
    else:
        lr_calc_stack = \
            create_low_res_stack_multislice(hr_guess, N_obj, Ns, \
                                             pupil*tf.exp(1j*pupil_angle), Np, LED_vec_i, \
                                             num_slices, H_scalar, H_scalar_f, batch_size, change_Ns, 
                                             use_window, window_2d_sqrt_us)
    # cos_theta**4 dropoff
    lr_calc_stack = lr_calc_stack*tf.gather(cos_theta_dropoff,LED_vec_i)
    if multiplexed:
        lr_calc_stack_expand = tf.expand_dims(lr_calc_stack, axis=-1)
        if real_mult:
            multiplexed_stack_calc = lr_calc_stack_expand*tf.gather(alpha_expand, batch_vec, axis=-1)
            multiplexed_stack_calc = tf.reduce_sum(multiplexed_stack_calc,axis=0)
            multiplexed_stack_calc = tf.transpose(multiplexed_stack_calc,[2,0,1])
            loss = tf.reduce_sum((tf.sqrt(tf.gather(multiplexed_stack/50,batch_vec, axis=0)) - \
                                  tf.sqrt(multiplexed_stack_calc))**2)
        else:
            multiplexed_stack_calc = lr_calc_stack_expand*tf.gather(alpha_expand*tf.expand_dims(exposure_time_used,-1), batch_vec, axis=-1)
            multiplexed_stack_calc = tf.reduce_sum(multiplexed_stack_calc,axis=0)
            multiplexed_stack_calc = tf.transpose(multiplexed_stack_calc,[2,0,1])
            loss = tf.reduce_sum((tf.sqrt(tf.gather(multiplexed_stack,batch_vec, axis=0)) - \
                                  tf.sqrt(multiplexed_stack_calc))**2)

    else:
        # loss = tf.reduce_mean((tf.sqrt(tf.gather(lr_observed_stack*exposure_time_used,LED_vec_i, axis=0)) - \
        #                         tf.sqrt(lr_calc_stack*tf.gather(exposure_time_used,LED_vec_i)))**2)

        loss = tf.reduce_mean((tf.sqrt(tf.gather(lr_observed_stack,LED_vec_i, axis=0)) - \
                                tf.sqrt(lr_calc_stack))**2)
    
        
    if not(projected_grad):
        loss += t2_reg*tf.reduce_mean((tf.abs(hr_guess))**2)
    return loss

def softthr(x, thr):
    # Ref: https://stats.stackexchange.com/questions/357339/soft-thresholding-for-the-lasso-with-complex-valued-data
    
    # softthr - Soft thresholding operator
    # args in -
    #   x - input vector
    #   thr - shrinkage threshold
    # args out - 
    #   z - output vector
    z = tf.cast(tf.abs(x), tf.float64) - tf.cast(thr, tf.float64)
    z = z * tf.cast(tf.cast(tf.abs(x), tf.float64) > tf.cast(thr, tf.float64), tf.float64)
    z = tf.cast(z, tf.complex128)*tf.exp(1j*tf.cast(tf.math.angle(x), dtype=tf.complex128))
    return(z)

@tf.function
def step(LED_vec_i, batch_vec):

    with tf.GradientTape(persistent=False) as tape:
        loss = func(LED_vec_i, batch_vec)

    opt_vars = [hr_guess]
    if optimize_pupil_ang:
        opt_vars += [pupil_angle_coeff]
            
    if change_Ns:
        opt_vars += [Ns]
        
    gradients = tape.gradient(loss, opt_vars)
    optimizer.apply_gradients(zip(gradients, opt_vars))
    
    if projected_grad:
        hr_guess.assign(softthr(hr_guess, adam_learning_rate*t2_reg))   # proximal update
        
    return(loss)

loss_vec = []


start_time = time.time()



if multiplexed:
    LED_vec_ds = tf.data.Dataset.from_tensor_slices(np.arange(num_patterns))
else:
    LED_vec_ds = tf.data.Dataset.from_tensor_slices(np.arange(num_leds))
if stochastic:
    LED_vec_ds = LED_vec_ds.shuffle(buffer_size=100)
LED_vec_ds = LED_vec_ds.repeat()
LED_vec_ds = LED_vec_ds.batch(batch_size)        
LED_vec_ds = iter(LED_vec_ds)

# batch_start = 0

for i in range(num_iter):
    
    # batch_vec = np.arange(batch_start,batch_start+batch_size)
    # batch_start += batch_size

    if multiplexed:
        batch_vec_i = next(LED_vec_ds) #batch_vec%batch_size
        print(batch_vec_i)
        LED_vec_i = np.arange(num_leds)
    else:
        batch_vec_i = None
        LED_vec_i = next(LED_vec_ds) # batch_vec%num_leds # these are the indices for im_stack
        # LED_vec_i = random.sample(range(num_leds), batch_size)
        print(LED_vec_i)
        
    loss = step(LED_vec_i, batch_vec_i)
    print('Iteration: ' + str(i))
    print(loss.numpy())
    loss_vec.append(loss.numpy())

end_time = time.time()
print('total time is: ' + str(end_time - start_time))


hr_computed = hr_guess.numpy()
if filter_hr:
    for ss in range(num_slices):
        hr_computed[ss,:,:] = Ft(F(hr_computed[ss,:,:])*NAfilter_synthetic.astype(np.complex128))
        plt.figure()
        plt.title('slice ' + str(ss) + ' amplitude')
        plt.imshow(np.abs(hr_computed[ss,visualize_trim:-visualize_trim,
                                      visualize_trim:-visualize_trim]))
        plt.colorbar()
        
        plt.figure()
        plt.title('slice ' + str(ss) + ' angle')
        plt.imshow(np.angle(hr_computed[ss,visualize_trim:-visualize_trim,
                                        visualize_trim:-visualize_trim]))
        plt.colorbar()
        


pupil_angle_final = (tf.reduce_sum(zernike_mat*pupil_angle_coeff, axis=2)).numpy()
lr_calc_stack_final = \
        create_low_res_stack_multislice(hr_guess, N_obj, Ns, \
                                        pupil*tf.exp(1j*pupil_angle_final), Np, np.arange(num_leds), \
                                        num_slices, H_scalar, H_scalar_f, num_leds, change_Ns, 
                                        use_window, window_2d_sqrt_us)
mse_loss = tf.reduce_mean((lr_observed_stack - lr_calc_stack_final)**2)
print('mse_loss is: ' + str(mse_loss))                

# save reconstruction if save_recons = True
if save_recons:
    subfolder_name = folder_name + '/reconstruction' + args.save_tag_recons
    create_folder(subfolder_name)
    
    if projected_grad:
        reg_type = 'l1'
    else:
        reg_type = 'l2'
    
    
    save_name = 'x_corner_' + str(x_corner) + '_y_corner_' + str(y_corner)
    
            
    create_folder(subfolder_name + '/' + save_name)
    np.save(subfolder_name + '/' + save_name + '/computed_obj.npy', hr_computed)
    np.save(subfolder_name + '/' + save_name + '/lr_observed_stack.npy', lr_observed_stack)
    np.save(subfolder_name + '/' + save_name + '/lr_calc_stack_final.npy', lr_calc_stack_final)
    np.save(subfolder_name + '/' + save_name + '/num_slices.npy', num_slices)
    np.save(subfolder_name + '/' + save_name + '/loss_vec.npy', loss_vec)
    np.save(subfolder_name + '/' + save_name + '/pupil_angle_final.npy', pupil_angle_final)
    np.save(subfolder_name + '/' + save_name + '/Ns.npy', Ns)
    np.save(subfolder_name + '/' + save_name + '/Ns_0.npy', Ns_0)
    np.save(subfolder_name + '/' + save_name + '/NAfilter_synthetic.npy', NAfilter_synthetic)
    
    print('saved as: ')
    print(subfolder_name + '/' + save_name)
    
    # save the final reconstruction in subfolder_name for merge_patches.py
    np.save(subfolder_name + '/computed_obj_i.npy', hr_computed)

    # save the window
    np.save(subfolder_name + '/window_2d_i.npy', window_2d)
    
if plt_flag:

    plt.figure()
    plt.plot(loss_vec)
    
    low_res_img_ind = 10
    plt.figure()
    plt.title('Low res actual')
    plt.imshow(lr_observed_stack[low_res_img_ind,int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor),
                                 int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor)], vmin=None, vmax=None)
    plt.colorbar()

    vmin_lr = np.min(lr_observed_stack[low_res_img_ind,int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor),
                                 int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor)])
    vmax_lr = np.max(lr_observed_stack[low_res_img_ind,int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor),
                                 int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor)])

    plt.figure()
    plt.title('Low res computed')
    plt.imshow(lr_calc_stack_final[low_res_img_ind,int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor),
                                 int(visualize_trim/upsample_factor):-int(visualize_trim/upsample_factor)], vmin=vmin_lr, vmax=vmax_lr)
    plt.colorbar()
    
    plt.figure()
    plt.title('Final pupil angle')
    plt.imshow(pupil_angle_final)
    
    plt.figure()
    plt.title('LED spatial freqs')
    plt.scatter(Ns_0[:,0], Ns_0[:,1])
    plt.scatter(Ns[:,0], Ns[:,1], c='r')
    