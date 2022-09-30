#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:10:15 2021

@author: vganapa1
"""

import sys

from SyntheticMNIST_functions import create_folder, F, Ft
from fpm_functions import create_low_res_stack_multislice2
import tensorflow as tf
import numpy as np
import time
import argparse
from zernike_polynomials import get_poly_mat

from helper_functions import physical_preprocess, compare, find_angle_offset
import matplotlib.pyplot as plt
from helper_pattern_opt import load_multiplexed


### Command line args ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('-i', type=int, action='store', dest='num_iter', \
                        help='number of iterations', default = 2000)

parser.add_argument('-b', type=int, action='store', dest='batch_size', \
                    help='batch_size', default = 1)

parser.add_argument('-p', type=int, action='store', dest='num_patterns', \
                    help='num_patterns when using the multiplexed option', default = None)    
    
parser.add_argument('--t2', type=float, action='store', dest='t2_reg', 
                        help='t2 regularization', default = 1e-2)

parser.add_argument('--alr', type=float, action='store', dest='adam_learning_rate', 
                        help='learning rate for adam optimizer', default = 1e-2)

parser.add_argument('--ae', type=float, action='store', dest='adam_epsilon', 
                        help='adam_epsilon', default = 1e-7)

parser.add_argument('--input_data', action='store', 
                    help='path to overall folder containing training and test data', 
                    default = 'dataset_test_multislice')

parser.add_argument('--save_tag', action='store', 
                    help='save_tag for the multiplexed images', 
                    default = None)

parser.add_argument('--obj_ind', type=int, 
                    action='store', dest='obj_ind',
                    help='example number to analyze', 
                    default = '0')

parser.add_argument('--mult', action='store_true', dest='multiplexed', 
                    help='use only multiplexed images to reconstruct')

parser.add_argument('--l1', action='store_true', dest='projected_grad', 
                    help='l1 regularization with projected gradient descent')

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier', 
                    help='poisson noise multiplier, higher value means higher SNR', 
                    default = (2**16-1)*0.41)

args = parser.parse_args()


#########################

### Parse command line args ###
num_iter = args.num_iter
adam_learning_rate = args.adam_learning_rate
adam_epsilon = args.adam_epsilon
input_data = args.input_data
save_tag = args.save_tag 
obj_ind = args.obj_ind
batch_size = args.batch_size
poisson_noise_multiplier = args.poisson_noise_multiplier
multiplexed = args.multiplexed # reconstruct with only multiplexed low res images
num_patterns = args.num_patterns
t2_reg = args.t2_reg
projected_grad = args.projected_grad # l1 regularization with projected gradient descent

if multiplexed:
    if batch_size > num_patterns:
        batch_size = num_patterns
        print('reduced batch size to num_patterns')


# Other Inputs
dataset_type = 'training'
optimize_pupil_ang = False
zernike_poly_order = 5
filter_hr = True
save_recons = True
sqrt_reg = np.finfo(np.float32).eps.item()
plt_flag = True # displays plots if True

    
### LOAD DATA ###
#############################

object_name = '{}/example_{:06d}'.format(dataset_type, obj_ind)

if multiplexed:
    image_path = input_data + '/' + object_name
    multiplexed_stack = \
    load_multiplexed(num_patterns,
                     image_path,
                     save_tag,
                     bit_depth=16,
                     dtype=tf.float64,
                     )    
    multiplexed_stack = multiplexed_stack.numpy()
    multiplexed_stack[multiplexed_stack<0] = 0
    multiplexed_stack[multiplexed_stack>1] = 1
    multiplexed_stack = tf.cast(multiplexed_stack, dtype=tf.float64)
    
    alpha = np.load(input_data + '/' + save_tag + '/all_alpha_train.npy')[obj_ind,:,0:num_patterns]
    alpha_expand = tf.cast(tf.expand_dims(tf.expand_dims(alpha, axis=0), axis=0), tf.float64)

# actual low-res stack, no noise
lr_observed_stack = np.load(input_data + '/' + object_name + '/im_stack.npy') # normalized
lr_observed_stack[lr_observed_stack<0] = 0
lr_observed_stack[lr_observed_stack>1] = 1


'''
lr_observed_stack_expand = tf.expand_dims(lr_observed_stack, axis=-1)
multiplexed_stack_calc = lr_observed_stack_expand*alpha_expand
multiplexed_stack_calc = tf.reduce_sum(multiplexed_stack_calc,axis=-2)

# Following should be approximately the same

plt.figure()
plt.imshow(multiplexed_stack_calc[:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(multiplexed_stack[:,:,0])
plt.colorbar()

'''

hr_obj = np.load(input_data + '/' + object_name + '/obj_stack.npy')
hr_obj = np.transpose(hr_obj, axes=[2,0,1])
#############################


normalizer = np.load(input_data + '/normalizer.npy')
offset = np.load(input_data + '/offset.npy')
N_obj = np.load(input_data + '/N_obj.npy')
Np = np.load(input_data + '/Np.npy')
Ns = np.load(input_data + '/Ns.npy')
pupil = np.load(input_data + '/pupil.npy').astype(np.complex128)
num_leds = np.load(input_data + '/num_leds.npy')
LED_vec = np.load(input_data + '/LED_vec.npy')
LEDs_used_boolean = np.load(input_data + '/LEDs_used_boolean.npy')
LED_num_used_leds = LED_vec[LEDs_used_boolean] 
NAfilter_synthetic = np.load(input_data + '/NAfilter_synthetic.npy')
LitCoord = np.load(input_data + '/LitCoord.npy')

LED_x = np.load(input_data + '/LED_x.npy')
LED_y = np.load(input_data + '/LED_y.npy', )
ds_led_x = np.load(input_data + '/ds_led_x.npy')
ds_led_y = np.load(input_data + '/ds_led_y.npy')
dd = np.load(input_data + '/dd.npy')
NA = np.load(input_data + '/NA.npy')
dpix_c = np.load(input_data + '/dpix_c.npy')
wavelength = np.load(input_data + '/wavelength.npy')
mag = np.load(input_data + '/mag.npy')

LED_center_x = np.load(input_data + '/LED_center_x.npy')
LED_center_y = np.load(input_data + '/LED_center_y.npy')
z_led = np.load(input_data + '/z_led.npy')

num_slices = np.load(input_data + '/num_slices.npy')
slice_spacing = np.load(input_data + '/slice_spacing.npy')
f = np.load(input_data + '/f.npy')

H_scalar = np.load(input_data + '/H_scalar.npy')
H_scalar_f  = np.load(input_data + '/H_scalar_f.npy')


dpix_m = dpix_c/mag
x_size = Np[0]
y_size = Np[1]

zernike_mat = get_poly_mat(x_size, y_size, x_size*dpix_m, \
                           y_size*dpix_m, wavelength, NA,
                           n_upper_bound = zernike_poly_order, show_figures = False)
    
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

if multiplexed:
    multiplexed_stack = multiplexed_stack/normalizer



# distance of LED to center:

LED_dist = np.sqrt(LED_x**2 + LED_y**2)   
LED_dist = LED_dist[LEDs_used_boolean]
LED_dist_ind = np.argsort(LED_dist)


obj_stack_init = tf.ones([num_slices, N_obj[0], N_obj[1]], dtype=tf.complex128)
hr_guess = tf.Variable(obj_stack_init)
pupil_angle_coeff = tf.Variable(np.zeros([zernike_mat.shape[-1],]))
optimizer = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate, \
                                     epsilon=adam_epsilon)
    


def func(LED_ind_vec_i, LED_vec_i, batch_vec):
    pupil_angle = tf.cast(tf.reduce_sum(zernike_mat*pupil_angle_coeff, axis=2), tf.complex128)
    
    if multiplexed:
        lr_calc_stack = \
            create_low_res_stack_multislice2(hr_guess, N_obj, Ns, \
                                            pupil*tf.exp(1j*pupil_angle), Np, LED_vec_i, \
                                            num_slices, H_scalar, H_scalar_f, num_leds)
    else:
        lr_calc_stack = \
            create_low_res_stack_multislice2(hr_guess, N_obj, Ns, \
                                            pupil*tf.exp(1j*pupil_angle), Np, LED_vec_i, \
                                            num_slices, H_scalar, H_scalar_f, batch_size)
    if multiplexed:
        lr_calc_stack_expand = tf.expand_dims(lr_calc_stack, axis=-1)
        multiplexed_stack_calc = lr_calc_stack_expand*tf.gather(alpha_expand, batch_vec, axis=-1)
        multiplexed_stack_calc = tf.reduce_sum(multiplexed_stack_calc,axis=-2)
        loss = tf.reduce_sum((tf.sqrt(tf.gather(multiplexed_stack,batch_vec, axis=-1)) - tf.sqrt(multiplexed_stack_calc))**2)
    else:
        loss = tf.reduce_sum((tf.sqrt(tf.gather(lr_observed_stack_noise,LED_ind_vec_i, axis=-1)) - tf.sqrt(lr_calc_stack))**2)
    
    if not(projected_grad):
        loss += t2_reg*tf.reduce_sum((tf.abs(hr_guess))**2)
    return loss

def softthr(x, thr):
    # Ref: https://stats.stackexchange.com/questions/357339/soft-thresholding-for-the-lasso-with-complex-valued-data
    
    # softthr - Soft thresholding operator
    # args in -
    #   x - input vector
    #   thr - shrinkage threshold
    # args out - 
    #   z - output vector
    z = tf.abs(x) - tf.cast(thr, tf.float64)
    z = z * tf.cast(tf.abs(x) > thr, tf.float64)
    z = tf.cast(z, tf.complex128)*tf.exp(1j*tf.cast(tf.math.angle(x), dtype=tf.complex128))
    return(z)

@tf.function
def step(LED_ind_vec_i, LED_vec_i, batch_vec):

    with tf.GradientTape(persistent=False) as tape:
        loss = func(LED_ind_vec_i, LED_vec_i, batch_vec)
    
    opt_vars = [hr_guess]
    if optimize_pupil_ang:
        opt_vars += [pupil_angle_coeff]
    gradients = tape.gradient(loss, opt_vars)
    optimizer.apply_gradients(zip(gradients, opt_vars))
    
    if projected_grad:
        hr_guess.assign(softthr(hr_guess, adam_learning_rate*t2_reg))   # proximal update
        
    return(loss)

loss_vec = []


start_time = time.time()

batch_start = 0


for i in range(num_iter):
    
    batch_vec = np.arange(batch_start,batch_start+batch_size)
    batch_start += batch_size

    if multiplexed:
        batch_vec = batch_vec%num_patterns
        LED_ind_vec_i = np.arange(num_leds)
        LED_vec_i = LED_vec[LEDs_used_boolean]
    else:
        LED_ind_vec_i = LED_dist_ind[batch_vec%num_leds] # these are the indices for im_stack
        LED_vec_i = LED_vec[LEDs_used_boolean][LED_ind_vec_i]
    
    
    loss = step(LED_ind_vec_i, LED_vec_i, batch_vec)
    print('Iteration: ' + str(i))
    print(loss.numpy())
    loss_vec.append(loss.numpy())

end_time = time.time()
print('total time is: ' + str(end_time - start_time))

hr_computed = hr_guess.numpy()
if filter_hr:
    for ss in range(num_slices):
        hr_computed[ss,:,:] = Ft(F(hr_computed[ss,:,:])*NAfilter_synthetic)

if plt_flag: # only needed for plotting purposes
    pupil_angle_final = (tf.reduce_sum(zernike_mat*pupil_angle_coeff, axis=2)).numpy()
    lr_calc_stack_final = create_low_res_stack_multislice2(hr_computed, N_obj, Ns, \
                                                           pupil*tf.exp(1j*pupil_angle_final), Np, LED_vec[LEDs_used_boolean], \
                                                               num_slices, H_scalar, H_scalar_f, num_leds)

compare_values_all = []
for s in range(num_slices):
    angle_offset = find_angle_offset(hr_obj[s], hr_computed[s])
    
    hr_computed[s] = hr_computed[s]*np.exp(1j*angle_offset)
    
    # output is mse_recon, psnr_recon, ssim_recon_angle, ssim_recon_abs, ssim_recon_intensity
    compare_values = \
        compare(hr_obj[s],hr_computed[s])
    compare_values_all.append(compare_values)

compare_values_all = np.stack(compare_values_all)

# save reconstruction if save_recons = True
if save_recons:
    subfolder_name = input_data + '/' + object_name + '/reconstruction'
    create_folder(subfolder_name)
    if projected_grad:
        reg_type = 'l1'
    else:
        reg_type = 'l2'
    if multiplexed:
        save_name = 'mult_iter_' + str(num_iter) + '_' + reg_type + '_' + str(t2_reg) +\
            '_pnm_' + str(poisson_noise_multiplier) + '_lr_' + str(adam_learning_rate) + '_b_' + str(batch_size) +\
            '_' + save_tag + '_p_' + str(num_patterns)            
    else:
        save_name = 'all_leds_iter_' + str(num_iter) + '_' + reg_type + '_' + str(t2_reg) +\
            '_pnm_' + str(poisson_noise_multiplier) + '_lr_' + str(adam_learning_rate) + '_b_' + str(batch_size)
            
    
    np.save(subfolder_name + '/' + save_name + '_computed_obj.npy', hr_computed)
    np.save(subfolder_name + '/' + save_name + '_compare_values_all.npy', compare_values_all)


if plt_flag:

    low_res_img_ind = 0
    plt.figure()
    plt.title('Low res actual')
    plt.imshow(lr_observed_stack_noise[:,:,low_res_img_ind], vmin=None, vmax=None)
    plt.colorbar()

    vmin_lr = np.min(lr_observed_stack_noise[:,:,low_res_img_ind])
    vmax_lr = np.max(lr_observed_stack_noise[:,:,low_res_img_ind])

    plt.figure()
    plt.title('Low res computed')
    plt.imshow(lr_calc_stack_final[:,:,low_res_img_ind], vmin=None, vmax=None)
    plt.colorbar()
    
    
    for ss in range(num_slices):
        if hr_obj is not None:
            plt.figure()
            plt.title('Absolute error map, slice ' + str(ss))
            plt.imshow(np.abs(hr_computed[ss,:,:] - hr_obj[ss,:,:]), vmin=None, vmax=None)
            plt.colorbar()

        

            plt.figure()
            plt.title('actual amplitude, slice ' + str(ss))
            plt.imshow(np.abs(hr_obj[ss,:,:]))
            plt.colorbar()
            
            v_min_amp = np.min(np.abs(hr_obj[ss,:,:]))
            v_max_amp = np.max(np.abs(hr_obj[ss,:,:]))
            
            plt.figure()
            plt.title('actual angle, slice ' + str(ss))
            plt.imshow(np.angle(hr_obj[ss,:,:]))
            plt.colorbar()
            
            v_min_ang = np.min(np.angle(hr_obj[ss,:,:]))
            v_max_ang = np.max(np.angle(hr_obj[ss,:,:]))    
            
        
            plt.figure()
            plt.title('computed amplitude, slice ' + str(ss))
            plt.imshow(np.abs(hr_computed[ss,:,:]), vmin=v_min_amp, vmax=v_max_amp)
            plt.colorbar()
        
            plt.figure()
            plt.title('computed angle, slice ' + str(ss))
            plt.imshow(np.angle(hr_computed[ss,:,:]), vmin=v_min_ang, vmax=v_max_ang)
            plt.colorbar()

    
    plt.figure()
    plt.title('Final pupil angle')
    plt.imshow(pupil_angle_final)
    

    