#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:52:13 2021

@author: vganapa1

Variational autoencoder approach to Mutual Information Maximization and
Probabilistic Reconstruction

"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import argparse
from scipy import signal
from SyntheticMNIST_functions import create_folder, F, Ft, NAfilter
from SyntheticMNIST_multislice_functions import get_real_data_params
from helper_pattern_opt import load_img_stack, load_img_stack_real_data
from helper_functions import trim_lit_coord, create_window, physical_preprocess, \
                             merge_patches_func
from visualizer_functions import show_figs2, \
                                 show_figs_alpha, \
                                 show_figs_input_output, show_alpha_scatter
from models_v2 import create_encode_net, \
                      create_decode_net
from FPM_VAE_helper_functions_v3 import create_dataset_iter, \
                                        find_loss_vae_unsup, calculate_log_prob_M_given_R, \
                                        calculate_log_prob_M_given_R_real_data
import os
import matplotlib.pyplot as plt
import skimage.transform

# tf.config.experimental.set_device_policy('warn')
# tf.debugging.enable_check_numerics()

tfd = tfp.distributions

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


setup_start_time = time.time()

### Command line args ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--real_data', action='store_true', dest='real_data', 
                    help='uses real data for the image stacks') 

parser.add_argument('--real_mult', action='store_true', dest='real_mult', 
                    help='uses real data for the MULTIPLEXED image stacks') 

parser.add_argument('--change_Ns', action='store_true', dest='change_Ns', 
                    help='optimizes led position, brightness, and pnm during training') 

parser.add_argument('--vary_pupil', action='store_true', dest='vary_pupil', 
                    help='allows the pupil to optimize for synthetic data. \
                        pupil is always optimized with real data') 

parser.add_argument('--use_window', action='store_true', dest='use_window', 
                    help='uses a windowing function for real data (ignored in synthetic data)') 

parser.add_argument('--ufs', action='store_true', dest='use_first_skip', 
                    help='use the first skip connection in the unet')

parser.add_argument('--normal', action='store_true', dest='use_normal', 
                    help='use a normal distribution as final distribution') 

parser.add_argument('--det', action='store_true', dest='deterministic', 
                    help='no latent variable, simply maximizes log probability of output_dist') 

parser.add_argument('--train', action='store_true', dest='train', 
                    help='run the training loop')

parser.add_argument('--final_train', action='store_true', dest='final_train', 
                    help='run all the training examples through the final trained net')

parser.add_argument('--fff_reconstruct', action='store_true', dest='final_full_field_reconstruct', 
                    help='only works for real data, reconstruct the entire FoV')

parser.add_argument('--input_path', action='store', help='path(s) to overall folder containing training data')

parser.add_argument('--save_path', action='store', help='path to save output', default = None)

parser.add_argument('--md', dest = 'multiplexed_description',
                    action='store', help='description of multiplex type', default = '') # _Dirichlet or _Random


parser.add_argument('--save_tag_mult', action='store', help='folder name describing the multiplexed image', dest = 'save_tag_multiplexed',
                    default = None)

parser.add_argument('-b', type=int, action='store', dest='batch_size', \
                        help='batch size', \
                        default = 4)    

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier', 
                    help='poisson noise multiplier, higher value means higher SNR')

parser.add_argument('--pnm_start', type=float, action='store', dest='pnm_start', 
                    help='poisson noise multiplier starting value, anneals to pnm value', default = None)

parser.add_argument('--lr', type=float, action='store', dest='learning_rate', 
                        help='learning rate', default = 1e-3)

parser.add_argument('--ae', type=float, action='store', dest='adam_epsilon', 
                        help='adam_epsilon', default = 1e-7)

parser.add_argument('-i', type=int, action='store', dest='num_iter', \
                        help='number of training iterations', default = 100)
    
parser.add_argument('--si', type=int, action='store', dest='save_interval', \
                        help='save_interval for checkpoints and intermediate values', default = 50000)
    
parser.add_argument('--restore', action='store_true', dest='restore', \
                    help='restore from previous training')

parser.add_argument('-r', type=int, action='store', dest='restore_num', \
                        help='checkpoint number to restore from', default = None)

parser.add_argument('--ulc', action='store_true', dest='use_latest_ckpt', \
                    help='uses latest checkpoint, overrides -r')
    
parser.add_argument('--dp', type=float, action='store', dest='dropout_prob', \
                        help='dropout_prob, percentage of nodes that are dropped', \
                        default=0.2)

parser.add_argument('--norm', type=float, action='store', dest='norm', \
                        help='gradient clipping by norm', \
                        default=100)

parser.add_argument('--td', type=int, action='store', dest='truncate_dataset', \
                        help='truncate_dataset by this value to not load in entire dataset; overriden when restoring a net', default = 2)

parser.add_argument('--klm', type=float, action='store', dest='kl_multiplier', \
                        help='multiply the kl_divergence term in the loss function by this factor', \
                        default=1)

parser.add_argument('--klaf', type=float, action='store', dest='kl_anneal_factor', \
                        help='multiply kl_anneal by this factor each iteration', \
                        default=1)

parser.add_argument('--astd', type=float, action='store', dest='anneal_std', \
                    help='anneal the standard dev of the dist P(M|O) by this constant term', \
                    default=0)
    
parser.add_argument('-p', type=int, action='store', dest='num_patterns', \
                        help='number of illumination patterns used per example', \
                        default = 2)
    
    
parser.add_argument('--ns', type=int, action='store', dest='num_samples', \
                        help='number of z samples to evaluate loss function', default = 2)
    
# for model_encode
parser.add_argument('--nfm', type=int, action='store', dest='num_feature_maps', \
                        help='number of features/2 in the first block of model_encode', default = 49)

parser.add_argument('--nfmm', type=float, action='store', dest='num_feature_maps_multiplier', \
                        help='multiplier of features for each block of model_encode', default = 1.1)
    
parser.add_argument('--ks', type=int, action='store', dest='kernel_size',
                    help='kernel size in model_encode_I_m', default = 4)
    
parser.add_argument('--se', type=int, action='store', dest='stride_encode',
                        help='convolution stride in model_encode_I_m', default = 2)

parser.add_argument('--nb', type=int, action='store', dest='num_blocks', \
                        help='num convolution blocks in model_encode_I_m and model_encode_alpha', default = 3)

# intermediate layers
parser.add_argument('--il', type=int, action='store', dest='intermediate_layers', \
                        help='intermediate_layers for model_encode', default = 2)

parser.add_argument('--ik', type=int, action='store', dest='intermediate_kernel', \
                    help='intermediate_kernel for model_encode', default = 4)
    
    
parser.add_argument('--pro', type=float, action='store', dest='pr_offset', \
                        help='offset for positive range function', default = np.finfo(np.float32).eps.item() )
    

# visualization parameters

parser.add_argument('--visualize', action='store_true', dest='visualize', 
                    help='visualize results')

parser.add_argument('--en', type=int, action='store', dest='example_num', \
                        help='example number in file name for visualization', default = 0)

parser.add_argument('--pi', type=int, action='store', dest='pattern_ind', \
                        help='pattern index for visualization of I_multiplexed', default = 0)
    
# parameters for real data

parser.add_argument('--uf', type=int, action='store', dest='upsample_factor', \
                    help='High resolution object pixels = collected image pixels * upsample_factor', default = 2)
    
parser.add_argument('--xcrop', type=int, action='store', dest='x_crop_size', \
                    help='patch size to consider in reconstruction', default = 256)
        
parser.add_argument('--ycrop', type=int, action='store', dest='y_crop_size', \
                    help='patch size to consider in reconstruction', default = 256)

parser.add_argument('--zernike', type=int, action='store', dest='zernike_poly_order', \
                    help='zernike_poly_order', default = 5)

    

parser.add_argument('--num_slices', type=int, action='store', dest='num_slices', \
                    help='num z slices', default = 1)
        
parser.add_argument('--slice_spacing', type=float, action='store', dest='slice_spacing', \
                    help='slice_spacing in um', default = 0)
            
parser.add_argument('--focal_dist', type=float, action='store', dest='f', \
                    help='distance from the focal plane to the last slice in um', default = 0)
    
args = parser.parse_args()



#########################

### Parse command line args ###
real_data = args.real_data
real_mult = args.real_mult
change_Ns = args.change_Ns
vary_pupil = args.vary_pupil
save_tag_multiplexed = args.save_tag_multiplexed
use_normal = args.use_normal
deterministic = args.deterministic
use_window = args.use_window
multiplexed_description = args.multiplexed_description

train = args.train
final_train = args.final_train
final_full_field_reconstruct = args.final_full_field_reconstruct
visualize = args.visualize

input_path = args.input_path
save_path = args.save_path
norm = args.norm
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
save_interval = args.save_interval
num_iter = args.num_iter
dropout_prob = args.dropout_prob

restore = args.restore
restore_num = args.restore_num
use_latest_ckpt = args.use_latest_ckpt

kl_multiplier = args.kl_multiplier
kl_anneal_factor = args.kl_anneal_factor

truncate_dataset = args.truncate_dataset
num_samples = args.num_samples
num_patterns = args.num_patterns

batch_size = args.batch_size

# for model_encode
num_feature_maps = args.num_feature_maps
num_feature_maps_multiplier = args.num_feature_maps_multiplier
kernel_size = args.kernel_size
stride_encode = args.stride_encode
num_blocks = args.num_blocks

# itermediate_layers
intermediate_layers = args.intermediate_layers
intermediate_kernel = args.intermediate_kernel

pr_offset = args.pr_offset

# Example to visualize

example_num = args.example_num # overall example, starting index

# parameters for real data    

upsample_factor = args.upsample_factor # High resolution object pixels = collected image pixels * upsample_factor
x_crop_size = args.x_crop_size
y_crop_size = args.y_crop_size
zernike_poly_order = args.zernike_poly_order
num_slices = args.num_slices
slice_spacing = args.slice_spacing
f = args.f

### Create folder for output ###
create_folder(save_path)  

num_leds = int(np.load(input_path + '/num_leds.npy'))


if real_data:
    image_x = x_crop_size
    image_y = y_crop_size
    
    full_image_x = int(np.load(input_path + '/image_x.npy'))
    full_image_y = int(np.load(input_path + '/image_y.npy'))
else:
    image_x = int(np.load(input_path + '/image_x.npy'))
    image_y = int(np.load(input_path + '/image_y.npy'))

    full_image_x = image_x
    full_image_y = image_y
    
    use_window = False # ignore use_window

if use_window:
    window_2d_0 = create_window(x_crop_size, y_crop_size)
    window_2d_sqrt_0 = np.sqrt(window_2d_0)
    window_2d_sqrt_us_0 = skimage.transform.rescale(window_2d_sqrt_0, 
                                                  upsample_factor, multichannel = False, order = 0, mode = 'constant')
    
    window_2d = tf.constant(window_2d_0, dtype=tf.float32)
    window_2d_sqrt = tf.constant(window_2d_sqrt_0, dtype=tf.float32)
    window_2d_sqrt_us = tf.constant(window_2d_sqrt_us_0, dtype=tf.float32)
else:
    window_2d = tf.ones([x_crop_size, y_crop_size])
    window_2d_sqrt_us = tf.ones([x_crop_size, y_crop_size])

# reconstructed dimensions
if real_data:
    image_x_r = upsample_factor*image_x
    image_y_r = upsample_factor*image_y
else:
    image_x_r = int(np.load(input_path + '/image_x_r.npy'))
    image_y_r = int(np.load(input_path + '/image_y_r.npy'))


if real_data:
    exposure_time_used = np.load(input_path + '/exposure_time_used.npy')  
    exposure_time_used = exposure_time_used/np.max(exposure_time_used) # normalize
    if real_mult:
        exposure_time_used = np.ones_like(exposure_time_used)
    
    normalizer = None
    normalizer_ang = None
    offset = None
    offset_ang = None
                        
else:
    
    exposure_time_used = None
    
    normalizer = np.load(input_path + '/normalizer.npy')
    normalizer_ang = np.load(input_path + '/normalizer_ang.npy')
    
    offset = np.load(input_path + '/offset.npy')
    offset_ang = np.load(input_path + '/offset_ang.npy')

### Set parameters ###

sqrt_reg = np.finfo(np.float32).eps.item() # regularizing sqrt in backprop
poisson_noise_multiplier = args.poisson_noise_multiplier #(2**16-1)*0.41
pnm_start = args.pnm_start

if pnm_start is not None:
    pnm_anneal_factor = np.exp(np.log(poisson_noise_multiplier/pnm_start)/num_iter)
else:
    pnm_anneal_factor = 1.0

poisson_noise_multiplier = tf.Variable(poisson_noise_multiplier, dtype=tf.float64)
pnm_anneal_factor = tf.Variable(pnm_anneal_factor, dtype=tf.float64)
'''
iter_vec = np.arange(num_iter)
plt.figure()
plt.plot(pnm_start*pnm_anneal_factor**iter_vec)
'''
### Load parameters ###

if real_data:
    led_position_xy = np.load(input_path + '/led_position_xy.npy')
else:
    LitCoord = np.load(input_path + '/LitCoord.npy')
    LitCoord2 = trim_lit_coord(LitCoord)
    LitCoord2 = tf.constant(LitCoord2)


if real_data:
    dpix_m = float(np.load(input_path + '/dpix_m.npy'))
    wavelength = float(np.load(input_path + '/wavelength.npy'))
    NA = float(np.load(input_path + '/NA.npy'))
    z_led = float(np.load(input_path + '/z_led.npy'))
    
    dx_obj = [dpix_m/upsample_factor, dpix_m/upsample_factor]


    zernike_mat, \
    img_coords_xm_full, \
    img_coords_ym_full, \
    H_scalar, \
    H_scalar_f,  \
    du, \
    um_m, \
    pupil, \
    N_obj, \
    Np= \
        get_real_data_params(full_image_x,
                             full_image_y,
                             dpix_m,
                             wavelength,
                             NA,
                             zernike_poly_order,
                             x_crop_size,
                             y_crop_size,
                             upsample_factor,
                             slice_spacing,
                             f,)
    
    
    LEDs_used_boolean = None
    LED_vec = None
    Ns = None
    NAfilter_synthetic = None

else:
    N_obj = np.load(input_path + '/N_obj.npy')
    Ns = np.load(input_path + '/Ns.npy')
    pupil = np.load(input_path + '/pupil.npy')
    Np = np.load(input_path + '/Np.npy')
    LED_vec = np.load(input_path + '/LED_vec.npy')
    LEDs_used_boolean = np.load(input_path + '/LEDs_used_boolean.npy')
    num_slices = int(np.load(input_path + '/num_slices.npy'))
    H_scalar = np.load(input_path + '/H_scalar.npy')
    H_scalar_f = np.load(input_path + '/H_scalar_f.npy')
    NAfilter_synthetic = np.load(input_path + '/NAfilter_synthetic.npy')
    LED_vec_i = LED_vec[LEDs_used_boolean]   

    zernike_mat = np.load(input_path + '/zernike_mat.npy')
    
    led_position_xy = None
    dpix_m = None
    z_led = None
    wavelength = None
    NA = None
    img_coords_xm_full = None
    img_coords_ym_full = None
    du = None
    um_m = None

num_zernike_coeff = zernike_mat.shape[-1]
zernike_mat = tf.expand_dims(zernike_mat,0) # give a batch dimension

#########################

### Create data input pipeline ###

train_ds, train_ds_no_shuffle, r_channels, load_img_stack2,\
       train_folders = create_dataset_iter(input_path,
                                           save_path,
                                           restore,
                                           truncate_dataset,
                                           batch_size,
                                           num_patterns,
                                           example_num,
                                           save_tag_multiplexed,
                                           real_data,
                                           image_x,
                                           image_y,
                                           full_image_x,
                                           full_image_y,
                                           led_position_xy,
                                           dpix_m,
                                           z_led,
                                           wavelength,
                                           NA,
                                           img_coords_xm_full, img_coords_ym_full, # full field coordinates
                                           du,
                                           um_m,
                                           real_mult,
                                           multiplexed_description,
                                           )
if real_data:
    r_channels = 2*num_slices # factor of 2 for real and imaginary

#########################

### Neural Networks ###
#########################


if real_data:
    pass
else:
    coords_x_np = (np.arange(0,image_x_r) - image_x_r/2)/image_x_r
    coords_y_np = (np.arange(0,image_y_r) - image_y_r/2)/image_y_r
    coords_xm_np, coords_ym_np = np.meshgrid(coords_x_np, coords_y_np, indexing='ij')
    
    coords_xm_np = np.expand_dims(coords_xm_np, axis=-1)
    coords_ym_np = np.expand_dims(coords_ym_np, axis=-1)
    coords_np = np.concatenate((coords_xm_np,coords_ym_np),axis=-1)
    coords_np = np.expand_dims(coords_np,axis=0)
    coords_np = np.repeat(coords_np, batch_size, axis=0)
    
    coords = tf.Variable(coords_np, dtype = tf.float32)

kl_anneal = tf.Variable(1, dtype=tf.float32)

anneal_std = tf.Variable(args.anneal_std, dtype=tf.float32)
'''
Initializers in:
https://www.tensorflow.org/api_docs/python/tf/keras/initializers
'''

# initializer = tf.keras.initializers.Constant(
#     value=1e-3
# )

initializer = tf.keras.initializers.GlorotUniform()
# initializer = tf.keras.initializers.Ones()
# initializer = tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)
apply_norm = False
norm_type = 'batchnorm', #'instancenorm'

num_feature_maps_vec = [int(num_feature_maps*num_feature_maps_multiplier**i) for i in range(num_blocks)]

if deterministic:
    feature_maps_multiplier = 2
else:
    feature_maps_multiplier = 4    

if real_data:
    coords=None # each example has its own input coords
    
model_encode, skips_pixel_x, skips_pixel_y, skips_pixel_z = \
create_encode_net(image_x,
                  image_y, 
                  image_x_r,
                  image_y_r,
                  num_leds,
                  num_feature_maps_vec,
                  batch_size,
                  num_blocks, 
                  kernel_size, 
                  stride_encode,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  coords,
                  initial_repeats = 10, # XXX CHANGE TO INPUT
                  encode_net_ind=0,
                  feature_maps_multiplier=feature_maps_multiplier,
                  real_data=real_data)


                    
model_decode = \
create_decode_net(skips_pixel_x,
                  skips_pixel_y,
                  skips_pixel_z,
                  num_leds,
                  num_zernike_coeff,
                  batch_size,
                  r_channels, # number of output channels
                  kernel_size, 
                  stride_encode,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  net_number = 0,
                  feature_maps_multiplier = feature_maps_multiplier,
                  use_first_skip = args.use_first_skip,
                  real_data=real_data,
                  change_Ns=change_Ns,
                  vary_pupil=vary_pupil,
                  )

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, \
                                     epsilon=adam_epsilon)



# save checkpoints

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(save_path, checkpoint_dir, "ckpt")



checkpoint = tf.train.Checkpoint(model_encode = model_encode,
                                 model_decode = model_decode,
                                 optimizer = optimizer,
                                 kl_anneal = kl_anneal,
                                 anneal_std = anneal_std,
                                 )    

if restore:
    # restore a checkpoint
    if use_latest_ckpt:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(save_path, checkpoint_dir)))
    else:
        checkpoint.restore(os.path.join(save_path, checkpoint_dir,'ckpt-')+str(restore_num))


# prior on z, the latent variable 

skip_shapes = np.array([batch_size*np.ones_like(skips_pixel_x), skips_pixel_x, skips_pixel_y, np.array(skips_pixel_z)//feature_maps_multiplier]).T
if use_normal:
    prior = [tfd.Normal(loc=tf.zeros(skip_shapes[i]), scale=1) for i in range(num_blocks+1)]
else:
    prior = [tfd.Beta(tf.ones(skip_shapes[i]), tf.ones(skip_shapes[i])) for i in range(num_blocks+1)]


# Trainable variables

trainable_vars = model_encode.trainable_variables + \
    model_decode.trainable_variables


@tf.function
def train_step(alpha,
               im_stack_multiplexed,
               training=False,
               use_prior = False,
               img_coords_xm=None, 
               img_coords_ym=None, 
               Ns_0=Ns, 
               cos_theta=None,
               real_data=real_data,
               poisson_noise_multiplier=poisson_noise_multiplier,
               ):

    alpha = alpha[:,:,0:num_patterns]
    
    with tf.GradientTape(watch_accessed_variables=True, persistent=False) as tape:
        tape.watch(trainable_vars)

        loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
        output_dist, \
        q, q_sample, kl_divergence, loglik, \
        Ns_dist_vec, \
        zernike_dist_vec, \
        cos_dist_vec, \
        pnm_dist_vec = find_loss_vae_unsup(alpha,
                                           im_stack_multiplexed,
                                           num_blocks,  
                                           image_x,
                                           image_y,
                                           num_leds,
                                           model_encode,
                                           model_decode,
                                           poisson_noise_multiplier,
                                           sqrt_reg,
                                           num_patterns,
                                           batch_size,
                                           prior,
                                           training,
                                           normalizer,
                                           normalizer_ang,
                                           offset,
                                           offset_ang,
                                           use_window,
                                           window_2d,
                                           window_2d_sqrt_us,
                                           use_prior = use_prior, # does not use conditional q(z|M)
                                           kl_anneal = kl_anneal,
                                           kl_multiplier=kl_multiplier,
                                           pr_offset = pr_offset,
                                           num_samples = num_samples,
                                           use_normal = use_normal,
                                           N_obj = N_obj,
                                           Ns = Ns_0,
                                           pupil = pupil,
                                           Np = Np,
                                           LED_vec = LED_vec,
                                           LEDs_used_boolean = LEDs_used_boolean,
                                           num_slices = num_slices,
                                           H_scalar = H_scalar,
                                           H_scalar_f = H_scalar_f,
                                           deterministic = deterministic,
                                           use_first_skip = args.use_first_skip,
                                           anneal_std = anneal_std,
                                           img_coords_xm=img_coords_xm, 
                                           img_coords_ym=img_coords_ym, 
                                           cos_theta=cos_theta,
                                           real_data=real_data,
                                           exposure_time_used = exposure_time_used,
                                           zernike_mat = zernike_mat,
                                           change_Ns = change_Ns,
                                           vary_pupil = vary_pupil,
                                           )

            

        loss_M_VAE = tf.reduce_mean(loss_M_VAE, axis=0)/1e5
        if real_data:
            loss_M_VAE = loss_M_VAE/1e3
            
    if training:
     
        # loss_M_VAE          
        gradients = tape.gradient(loss_M_VAE, trainable_vars)
        gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
        gradients = [tf.clip_by_norm(g, norm)
                      for g in gradients]   
        optimizer.apply_gradients(zip(gradients, trainable_vars))   
     
    
    return(loss_M_VAE, alpha_vec, im_stack_multiplexed_vec,
           output_dist, q, q_sample, kl_divergence, loglik,
           Ns_dist_vec,
           zernike_dist_vec,
           cos_dist_vec,
           pnm_dist_vec)


if train:

    train_loss_vec = []
    train_loss_kl = []
    train_loss_loglik = []
    iter_vec = []
    
    
    if num_iter == 0:
        start_time = time.time()
        
    ### Training and Validation ###
    for iter_i in range(num_iter):
        # print(tf.config.experimental.get_memory_usage("GPU:0"))
        
        # kl_anneal.assign(tf.minimum(tf.maximum(kl_anneal*kl_anneal_factor,1),100))
        kl_anneal.assign(tf.maximum(kl_anneal*kl_anneal_factor,1))

        anneal_std.assign(anneal_std*.9999)
        # print(anneal_std)
        
        if real_data:
            image_path, alpha, im_stack_multiplexed, img_coords_xm, img_coords_ym, Ns_0, synthetic_NA, cos_theta = next(train_ds)
            
        else:
            path, im_stack, im_stack_r, alpha, im_stack_multiplexed = next(train_ds)
            img_coords_xm=None 
            img_coords_ym=None
            Ns_0=Ns
            cos_theta=None
            real_data=None
  
        loss_M_VAE, _, _, _, _, _, kl_divergence, loglik, _, _, _, _ = train_step(alpha,
                                                                 im_stack_multiplexed,
                                                                 training=True,
                                                                 use_prior = False,
                                                                 img_coords_xm=img_coords_xm, 
                                                                 img_coords_ym=img_coords_ym, 
                                                                 Ns_0=Ns_0, 
                                                                 cos_theta=cos_theta,
                                                                 real_data=real_data,
                                                                 poisson_noise_multiplier=poisson_noise_multiplier*pnm_anneal_factor**iter_i
                                                                 )

        print('Iteration number: ' + str(iter_i))
        print('Training loss_M_VAE: ' + str(loss_M_VAE))
        train_loss_vec.append(loss_M_VAE)
        train_loss_kl.append(kl_divergence)
        train_loss_loglik.append(loglik)

        if np.isnan(loss_M_VAE):
            sys.exit()
        
  
        
        if iter_i == 0:
            setup_end_time = time.time()
            print('Setup took ' + str((setup_end_time-setup_start_time)/60) + ' minutes.')
            start_time = time.time()

        
        if((iter_i%save_interval == 0) or (iter_i == num_iter-1)):
            #save and checkpoint
            iter_vec.append(iter_i)            
            np.save(save_path + '/train_loss_vec.npy', train_loss_vec)
            np.save(save_path + '/train_loss_kl.npy', train_loss_kl)
            np.save(save_path + '/train_loss_loglik.npy', train_loss_loglik)
            np.save(save_path + '/iter_vec.npy', iter_vec)
            checkpoint.save(file_prefix = checkpoint_prefix)
            
            
    end_time = time.time()
    print('Training took ' + str((end_time-start_time)/60) + ' minutes.')
    

    show_figs2(save_path, 
               iter_vec,
               train_loss_vec,
               )
    

### End of Training ###

def output_point_estimate(output_dist, 
                          sample_ind=0,
                          synthetic_NA=None,
                          NAfilter_synthetic=None):

    
    # output_mean = output_dist[sample_ind].sample()
    # output_mean = output_dist[sample_ind].loc()
    
    output_mean = []
    for sample_ind in range(num_samples):
        output_mean.append(output_dist[sample_ind].loc)
    
    output_mean = tf.stack(output_mean)
    output_mean = tf.experimental.numpy.mean(output_mean,axis=0)
    
    
    # filter the output mean by the Synthetic NA
    obj_re, obj_im = tf.split(output_mean,2,axis=-1)

    if not(real_data):
        obj_re = obj_re/normalizer_ang[0] + offset_ang[0]
        obj_im = obj_im/normalizer_ang[1] + offset_ang[1]
    obj = tf.cast(obj_re, tf.complex64) + tf.cast(obj_im, tf.complex64)*1j
    
    all_filtered_obj = []
    for b in range(batch_size):
        # obj_reals = []
        # obj_imags = []
        filtered_obj_vec = []
        
        if real_data:
            NAfilter_synthetic = NAfilter(N_obj[0], N_obj[1], N_obj[0]*dx_obj[0], \
                                          N_obj[1]*dx_obj[1], wavelength, synthetic_NA[b],
                                          )        
        for s in range(num_slices):
            obj_i = obj[b,:,:,s]
            # filter by sythetic NA
            O = F(obj_i)
            O = O*NAfilter_synthetic
            filtered_obj = Ft(O) #low resolution field
            filtered_obj_vec.append(filtered_obj)
    
            # obj_reals.append((np.real(filtered_obj) - offset_ang[0])*normalizer_ang[0])
            # obj_imags.append((np.imag(filtered_obj) - offset_ang[1])*normalizer_ang[1])
        
        # Un-normalized reconstructed object
        final_obj_unnorm = tf.stack(filtered_obj_vec,axis=-1)
        all_filtered_obj.append(final_obj_unnorm)
        
    all_filtered_obj = tf.stack(all_filtered_obj,axis=0)
    return(all_filtered_obj, output_mean)

def final_evaluation(ds = train_ds, dataset_type = 'training', folders = train_folders,
                     real_data=real_data, NAfilter_synthetic=None):
    print('Starting final ' + dataset_type + '...')
    start_time = time.time()
    
    val_ds_iter = iter(ds)
    
    loss_final_val = []
    path_final_val = []
    # all_filtered_obj_vec = []
    # all_im_stack_multiplexed = []
    
    val_size = len(folders)
    
    for val_ind in range(1): # single batch # range(val_size//batch_size): # all batches

        if real_data:
            path, alpha, im_stack_multiplexed, img_coords_xm, \
            img_coords_ym, Ns_0, synthetic_NA, cos_theta = next(train_ds)
            NAfilter_synthetic=None
        else:
            path, im_stack, im_stack_r, alpha, im_stack_multiplexed = next(val_ds_iter)
            img_coords_xm=None 
            img_coords_ym=None
            Ns_0=Ns
            cos_theta=None
            real_data=None
            synthetic_NA=None

        
        # all_im_stack_multiplexed.append(im_stack_multiplexed)
        create_folder(save_path + '/' + dataset_type)
        np.save(save_path + '/' + dataset_type  + '/im_stack_multiplexed' + str(val_ind) + '.npy', 
                im_stack_multiplexed)

        loss_M_VAE, _, _, output_dist, _, _, _, _, _, _, _, _ = train_step(alpha,
                                                                           im_stack_multiplexed,
                                                                           training=False,
                                                                           use_prior = False,
                                                                           img_coords_xm=img_coords_xm, 
                                                                           img_coords_ym=img_coords_ym, 
                                                                           Ns_0=Ns_0, 
                                                                           cos_theta=cos_theta,
                                                                           real_data=real_data,
                                                                           )

        all_filtered_obj, output_mean = output_point_estimate(output_dist, synthetic_NA=synthetic_NA, \
                                                 NAfilter_synthetic = NAfilter_synthetic)
        # all_filtered_obj_vec.append(all_filtered_obj)
        np.save(save_path + '/' + dataset_type  + '/all_filtered_obj' + str(val_ind) + '.npy', 
                all_filtered_obj)
        
        #save entropy # XXX fix to account for entropy in latent variable
        entropy_vec = tf.reduce_sum(output_dist[0].entropy(),axis=[1,2,3])
        np.save(save_path + '/' + dataset_type  + '/entropy_vec' + str(val_ind) + '.npy', 
                entropy_vec)
            
        print('loss:')
        print(loss_M_VAE)

        loss_final_val.append(loss_M_VAE)
        path_final_val.append(path)

    loss_final_val = np.stack(loss_final_val)
    path_final_val = np.stack(path_final_val)
    # all_filtered_obj_vec = np.concatenate(all_filtered_obj_vec, axis=0)
    # all_im_stack_multiplexed = np.concatenate(all_im_stack_multiplexed, axis=0)
    
    np.save(save_path + '/loss_final_' + dataset_type + '.npy', loss_final_val)
    np.save(save_path + '/path_final_' + dataset_type + '.npy', path_final_val)
    # np.save(save_path + '/all_filtered_obj_vec_' + dataset_type + '.npy', all_filtered_obj_vec)
    # np.save(save_path + '/all_im_stack_multiplexed_' + dataset_type + '.npy', all_im_stack_multiplexed)
    
    print('Average loss, final ' + dataset_type + ':')
    print(np.mean(loss_final_val))

    end_time = time.time()
    print(dataset_type + ' took ' + str((end_time-start_time)/60) + ' minutes.')
    

if final_train:
    
    final_evaluation(ds = train_ds_no_shuffle, dataset_type = 'training', folders = train_folders,
                     NAfilter_synthetic=NAfilter_synthetic)


    
### End of final train ###

def load_batch(force_x_corner=None,
               force_y_corner=None,
               data_folder='training',
               ):

    test_path_vec = []
    alpha = []
    for batch_i in range(example_num, example_num+batch_size):
        test_path = '{}/{}/example_{:06d}'.format(input_path, data_folder, example_num)  
        test_path_vec.append(test_path)
        alpha_i = np.expand_dims(np.load(save_path + '/all_alpha_train.npy')[example_num],axis=0)
        alpha.append(alpha_i)

    alpha = tf.concat(alpha,axis=0)

    im_stack=[]
    im_stack_r = []
    im_stack_multiplexed = []
    synthetic_NA = []
    img_coords_xm = []
    img_coords_ym = []
    Ns_0 = []
    for ind, test_path in enumerate(test_path_vec):
        print(test_path)
        
        if real_data:
            path, alpha_i, im_stack_multiplexed_i, \
            img_coords_xm_i, img_coords_ym_i, Ns_0_i, \
            synthetic_NA_i, cos_theta_0 = \
                load_img_stack_real_data(test_path, num_patterns, 
                                         alpha[ind],
                                         save_tag_multiplexed,
                                         image_x,
                                         image_y,
                                         full_image_x,
                                         full_image_y,
                                         img_coords_xm_full, img_coords_ym_full, # full field coords
                                         led_position_xy,
                                         dpix_m,
                                         z_led,
                                         wavelength,
                                         NA,
                                         du,
                                         um_m,
                                         16, # bit depth
                                         real_mult=real_mult,
                                         force_x_corner=force_x_corner,
                                         force_y_corner=force_y_corner,
                                         multiplexed_description=multiplexed_description)
            synthetic_NA.append(synthetic_NA_i)
            img_coords_xm.append(img_coords_xm_i)
            img_coords_ym.append(img_coords_ym_i)
            Ns_0.append(Ns_0_i)
        else:
            path, im_stack_i, im_stack_i_r, alpha_i, im_stack_multiplexed_i = \
                load_img_stack(test_path, num_leds, num_patterns,
                               r_channels, alpha[ind], bit_depth = 16,
                               save_tag_multiplexed = save_tag_multiplexed,
                               )

            im_stack_i = tf.expand_dims(im_stack_i, axis=0) # give im_stack a batch dimension
            im_stack.append(im_stack_i)
            
            im_stack_i_r = tf.expand_dims(im_stack_i_r, axis=0) # give im_stack a batch dimension
            im_stack_r.append(im_stack_i_r)
            
        im_stack_multiplexed_i = tf.expand_dims(im_stack_multiplexed_i, axis=0)
        im_stack_multiplexed.append(im_stack_multiplexed_i)
        

    if real_data:
        synthetic_NA = tf.concat(synthetic_NA,axis=0)
        img_coords_xm = tf.stack(img_coords_xm,axis=0)
        img_coords_ym = tf.stack(img_coords_ym,axis=0)
        Ns_0 = tf.stack(Ns_0,axis=0)
    else:
        im_stack = tf.concat(im_stack,axis=0)
        im_stack_r = tf.concat(im_stack_r,axis=0)
    im_stack_multiplexed = tf.concat(im_stack_multiplexed, axis=0)

    if real_data:
        return(synthetic_NA, im_stack_multiplexed, img_coords_xm, \
        img_coords_ym, Ns_0, cos_theta_0, test_path_vec, alpha)
    else:
        return(im_stack, im_stack_r, im_stack_multiplexed, test_path_vec, alpha)


def evaluate_patch_func(force_x_corner,force_y_corner):
    synthetic_NA, im_stack_multiplexed, img_coords_xm, \
    img_coords_ym, Ns_0, cos_theta, _, alpha = load_batch(force_x_corner=force_x_corner,
                                                force_y_corner=force_y_corner
                                                )
    loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
    output_dist, q, q_sample, kl_divergence, loglik, \
    Ns_dist_vec, \
    zernike_dist_vec, \
    cos_dist_vec, \
    pnm_dist_vec = train_step(alpha,
                              im_stack_multiplexed,
                              training=False,
                              use_prior = False,
                              img_coords_xm=img_coords_xm, 
                              img_coords_ym=img_coords_ym, 
                              Ns_0=Ns_0, 
                              cos_theta=cos_theta,
                              real_data=real_data,
                              )

    
    ### XXX save the low res stack
    
    # uncomment next line to overwrite NA
    # synthetic_NA = tf.ones(batch_size)*0.6
    all_filtered_obj, _ = output_point_estimate(output_dist, synthetic_NA=synthetic_NA, 
                                             NAfilter_synthetic=NAfilter_synthetic)
    all_filtered_obj = tf.transpose(all_filtered_obj[0], perm=[2,0,1])
    return(all_filtered_obj,tf.cast(window_2d, tf.complex64))
        
if real_data:
    if final_full_field_reconstruct: # only implemented for real data
        
        # XXX inefficiency because filling each batch with the same patch
        
        visualize_trim = 1
        
        try:
            # full_field = np.load(save_path + '/full_field_restore_' +str(restore_num) + '.npy')
            full_field = np.load(save_path + '/full_field_example_' + str(example_num) + '.npy')
            
        except FileNotFoundError:
            start_x_corner = 0
            start_y_corner = 0
            overlap_x = 256
            overlap_y = 256
            num_patches_x = 7
            num_patches_y = 7
            
            
            if use_window:
                visualize_window = window_2d
            else:
                visualize_window = create_window(x_crop_size,y_crop_size,signal.windows.bartlett)
    
            visualize_window = tf.expand_dims(visualize_window, axis=0)
            visualize_window = tf.cast(visualize_window, tf.complex64)
            
            full_field, full_field_window = \
                merge_patches_func(upsample_factor,
                                   x_crop_size,
                                   y_crop_size,
                                   num_patches_x,
                                   num_patches_y,
                                   overlap_x,
                                   overlap_y,
                                   num_slices,
                                   start_x_corner,
                                   start_y_corner,
                                   evaluate_patch_func, 
                                   visualize_window,
                                   sqrt_reg,
                                   )
    
            np.save(save_path + '/full_field_example_' + str(example_num) + '.npy', full_field)
        
        for ss in range(num_slices):
            plt.figure(figsize=[10,10])
            plt.title('slice ' + str(ss) + ' amplitude')
            plt.imshow(np.abs(full_field[ss,visualize_trim:-visualize_trim,
                                          visualize_trim:-visualize_trim]))
            plt.savefig('slice_' + str(ss) + '_amplitude.png', pad_inches=0, dpi=600)
            plt.colorbar()
            
            
            plt.figure(figsize=[10,10])
            plt.title('slice ' + str(ss) + ' angle')
            plt.imshow(np.angle(full_field[ss,visualize_trim:-visualize_trim,
                                            visualize_trim:-visualize_trim]))
            plt.savefig('slice_' + str(ss) + '_angle.png', pad_inches=0, dpi=600)
            plt.colorbar()
    
if visualize:

    pattern_ind = args.pattern_ind # pattern ind for displaying multiplexed image
    batch_ind = 0
    sample_ind = 1
    force_x_corner=768
    force_y_corner=768
    
    data_folder = 'training'


    
    if real_data:
        synthetic_NA, im_stack_multiplexed, img_coords_xm, \
        img_coords_ym, Ns_0, cos_theta, test_path_vec, alpha = load_batch(force_x_corner=force_x_corner,
                                                    force_y_corner=force_y_corner
                                                    )
        # im_stack_multiplexed = tf.ones_like(im_stack_multiplexed)
        # img_coords_xm = tf.ones_like(img_coords_xm)
        # img_coords_ym = tf.ones_like(img_coords_ym)
    else:
        im_stack, im_stack_r, im_stack_multiplexed, test_path_vec, alpha = load_batch()
        img_coords_xm = None
        img_coords_ym = None
        Ns_0 = Ns
        synthetic_NA = None
        
    test_path = test_path_vec[batch_ind]
    

    loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
    output_dist, q, q_sample, kl_divergence, loglik, \
    Ns_dist_vec, \
    zernike_dist_vec, \
    cos_dist_vec, \
    pnm_dist_vec = train_step(alpha,
                              im_stack_multiplexed,
                              training=False,
                              use_prior = False,
                              img_coords_xm=img_coords_xm, 
                              img_coords_ym=img_coords_ym, 
                              Ns_0=Ns_0, 
                              cos_theta=cos_theta,
                              real_data=real_data,
                              )

        
    # uncomment next line to overwrite NA
    # synthetic_NA = tf.ones(batch_size)*0.2
    all_filtered_obj, output_mean = output_point_estimate(output_dist, synthetic_NA=synthetic_NA, 
                                             NAfilter_synthetic=NAfilter_synthetic)



    create_folder(test_path + '/' + save_path)
    np.save(test_path + '/' + save_path + '/im_stack_multiplexed.npy', im_stack_multiplexed_vec)
    np.save(test_path + '/' + save_path + '/alpha_vec.npy', alpha_vec)
    
    try:
        train_loss_vec = np.load(save_path + '/train_loss_vec.npy')
        iter_vec = np.load(save_path + '/iter_vec.npy')
        
        show_figs2(save_path, 
                   iter_vec,
                   train_loss_vec,
                   )
    except FileNotFoundError:
        pass
    
    
    if len(Ns_dist_vec)>0:        
        plt.figure()
        plt.title('LED spatial freqs')
        plt.scatter(Ns_0[batch_ind,:,0], Ns_0[batch_ind,:,1])
        
        Ns = tf.cast(Ns_dist_vec[sample_ind].sample(), tf.float64) + Ns_0
        plt.scatter(Ns[batch_ind,:,0], Ns[batch_ind,:,1], c='r')
    else:
        Ns = Ns_0
    
    if len(zernike_dist_vec)>0:
        zernike_sample = zernike_dist_vec[sample_ind].sample()
        zernike_sample = tf.expand_dims(tf.expand_dims(zernike_sample,1),1)
        print('!!!')
        pupil_angle = tf.reduce_sum(zernike_mat*tf.cast(zernike_sample, tf.float64), axis=-1)
        pupil_new = pupil*tf.exp(1j*tf.cast(pupil_angle, tf.complex128))
        mask = np.ones_like(pupil)
        mask[np.abs(pupil)<(1-sqrt_reg)] = np.nan
        plt.figure()
        plt.imshow(mask*pupil_angle[batch_ind,:,:])
        plt.colorbar()

    
    if len(cos_dist_vec)>0:
        plt.figure()
        plt.plot(cos_theta[batch_ind])
        plt.plot(cos_theta[batch_ind]+tf.cast(cos_dist_vec[sample_ind].sample()[batch_ind], tf.float64), 'r')
    
    if len(pnm_dist_vec)>0:
        print('pnm delta is: ' + str(pnm_dist_vec[sample_ind].sample()[batch_ind].numpy()[0]))
    
    if real_data:
        show_alpha_scatter(led_position_xy, alpha_vec[batch_ind,:,pattern_ind], 
                           im_stack_multiplexed_vec[batch_ind,:,:,pattern_ind])
        
    else:
        show_figs_alpha(save_path, 
                        alpha_vec,
                        batch_size,
                        im_stack_multiplexed_vec,
                        data_folder,
                        example_num,
                        0,
                        pattern_ind,
                        num_leds,
                        num_patterns,
                        LitCoord2,
                        batch_ind=batch_ind)
        
    '''
    for sample_ind in range(2):
        show_figs_input_output(save_path, 
                                data_folder,
                                im_stack_r,
                                output_dist[sample_ind].mean(),
                                batch_ind,
                                img_ind)
    '''

    
    
    if real_data:

        for slice_ind in range(num_slices):
            plt.figure()
            plt.title('Object Amplitude')
            plt.imshow(np.abs(all_filtered_obj[batch_ind,:,:,slice_ind]))
            plt.savefig(save_path + '/obj_amp.png', dpi=600)
            plt.colorbar()
            
            plt.figure()
            plt.title('Object Phase')
            plt.imshow(np.angle(all_filtered_obj[batch_ind,:,:,slice_ind]))
            plt.savefig(save_path + '/obj_phase.png', dpi=600)
            plt.colorbar()
            
        # Compare multiplexed images
        im_stack_multiplexed_dist, im_stack_multiplexed_emulated, im_stack_emulated = \
            calculate_log_prob_M_given_R_real_data(tf.cast(output_mean, tf.float32), # batch_size x image_x x image_y x num_slices*2
                                         tf.cast(tf.expand_dims(alpha[:,:,0:num_patterns],axis=0),tf.float32), # expand for max_steps, dims are: max_steps x batch_size x num_leds x num_patterns
                                         batch_size,
                                         tf.cast(poisson_noise_multiplier, tf.float64),
                                         sqrt_reg,
                                         N_obj,
                                         Ns,
                                         cos_theta,
                                         pupil_new,
                                         Np,
                                         num_leds,
                                         num_slices,
                                         H_scalar,
                                         H_scalar_f,
                                         exposure_time_used,
                                         use_window,
                                         window_2d,
                                         window_2d_sqrt_us,
                                         change_Ns,
                                         anneal_std,
                                         1, # max_steps
                                         )

    
        
        plt.figure()
        plt.title('im_stack_multiplexed actual')
        plt.imshow(im_stack_multiplexed_vec[batch_ind,:,:,pattern_ind]*window_2d, # first dim is max_steps dim
                   )
        vmin_mult = np.min(im_stack_multiplexed_vec[batch_ind,:,:,pattern_ind]*window_2d)
        vmax_mult = np.max(im_stack_multiplexed_vec[batch_ind,:,:,pattern_ind]*window_2d)
        plt.colorbar()
        
        plt.figure()
        plt.title('im_stack_multiplexed emulated')
        # apply image saturation
        im_stack_multiplexed_emulated = im_stack_multiplexed_dist.loc[0,batch_ind,:,:,pattern_ind]
        im_stack_multiplexed_emulated = tf.maximum(im_stack_multiplexed_emulated,0)
        im_stack_multiplexed_emulated = tf.minimum(im_stack_multiplexed_emulated,1)  
        
        plt.imshow(im_stack_multiplexed_emulated, # first dim is max_steps dim
                   vmin=vmin_mult,
                   vmax=vmax_mult)
        plt.colorbar()
        
        try:
            if multiplexed_description == '_Random':
                im_stack = np.load(test_path_vec[batch_ind] +'/im_stack.npy')*50
            elif multiplexed_description == '_Dirichlet':
                im_stack = np.load(test_path_vec[batch_ind] +'/im_stack.npy')*100
            im_stack = im_stack[:,
                                force_x_corner:force_x_corner + x_crop_size,
                                force_y_corner:force_y_corner + y_crop_size]
            
            im_stack = tf.transpose(im_stack, perm=[1,2,0])
            im_stack = tf.expand_dims(im_stack,0)
            im_stack = tf.repeat(im_stack,batch_size,0)
            im_stack_multiplexed_from_single = \
                physical_preprocess(im_stack, 
                                    tf.expand_dims(alpha_vec,0), # add max_steps dimension
                                    poisson_noise_multiplier, # poisson_noise_multiplier*50
                                    sqrt_reg, #sqrt_reg,
                                    batch_size,
                                    1, # max_steps
                                    False, #renorm, True if need to remove normalization and offset
                                    quantize_noise=True,
                                    dtype=tf.float64,
                                    )
            plt.figure()
            plt.title('im_stack_multiplexed from single stack')
            plt.imshow(tf.cast(im_stack_multiplexed_from_single[0,batch_ind,:,:,pattern_ind], tf.float32)*window_2d, # first dim is max_steps dim
                       vmin=vmin_mult,
                       vmax=vmax_mult)
            plt.colorbar()
            
            
            led_ind=10
            
            plt.figure()
            plt.title('im_stack actual')
            plt.imshow(im_stack[batch_ind,:,:,led_ind]*tf.cast(window_2d,tf.float64))
            plt.colorbar()
            
            vmin=np.min(im_stack[batch_ind,:,:,led_ind]*tf.cast(window_2d,tf.float64))
            vmax=np.max(im_stack[batch_ind,:,:,led_ind]*tf.cast(window_2d,tf.float64))
            
            plt.figure()
            plt.title('im_stack_emulated')
            plt.imshow(im_stack_emulated[batch_ind,:,:,led_ind],vmin=vmin, vmax=vmax)
            plt.colorbar()
            

            
            mse_loss = tf.reduce_mean((im_stack_emulated - im_stack)**2)
        except FileNotFoundError:
            pass
        
        
    else:
        # add normalization back to all_filtered_obj
        
        obj_re = (np.real(all_filtered_obj) - offset_ang[0])*normalizer_ang[0]
        obj_im = (np.imag(all_filtered_obj) - offset_ang[1])*normalizer_ang[1]
        
        output_renorm = tf.concat((obj_re,obj_im), axis=-1)
        for img_ind in range(r_channels):
            show_figs_input_output(save_path, 
                                   data_folder,
                                   im_stack_r,
                                   output_renorm,
                                   batch_ind,
                                   img_ind)
    
    
        # Compare low res images
        im_stack_multiplexed_final, im_stack_final = \
            calculate_log_prob_M_given_R(output_renorm, # batch_size x image_x x image_y x num_slices*2
                                         tf.expand_dims(alpha[:,:,0:num_patterns],axis=0), # expand for max_steps, dims are: max_steps x batch_size x num_leds x num_patterns
                                         batch_size,
                                         tf.cast(poisson_noise_multiplier, tf.float32),
                                         sqrt_reg,
                                         1, # max_steps
                                         normalizer,
                                         normalizer_ang,
                                         offset,
                                         offset_ang,
                                         N_obj = N_obj,
                                         Ns = Ns,
                                         pupil = pupil_new,
                                         Np = Np,
                                         LED_vec = LED_vec,
                                         LEDs_used_boolean = LEDs_used_boolean,
                                         num_slices = num_slices,
                                         H_scalar = H_scalar,
                                         H_scalar_f = H_scalar_f,
                                         visualize=True,
                                         anneal_std=anneal_std,
                                         )
    
        
        plt.figure()
        plt.title('im_stack_multiplexed actual')
        plt.imshow(im_stack_multiplexed_vec[batch_ind,:,:,pattern_ind], # first dim is max_steps dim
                   vmin=np.min(im_stack_multiplexed),
                   vmax=np.max(im_stack_multiplexed))
        plt.colorbar()
        
        plt.figure()
        plt.title('im_stack_multiplexed emulated')
        plt.imshow(im_stack_multiplexed_final[0,batch_ind,:,:,pattern_ind], # first dim is max_steps dim
                   vmin=np.min(im_stack_multiplexed),
                   vmax=np.max(im_stack_multiplexed))
        plt.colorbar()
        
        '''
        show_figs_input_output(save_path, 
                                data_folder,
                                im_stack_multiplexed_vec,
                                im_stack_multiplexed_final[0],
                                batch_ind,
                                pattern_ind)
        '''
        
        mult_MSE_error = tf.reduce_mean((im_stack_multiplexed_final[0]-im_stack_multiplexed_vec)**2)
        im_stack_MSE_error = tf.reduce_mean((im_stack - im_stack_final)**2)
        
        led_ind = 10
        
        plt.figure()
        plt.title('im_stack actual')     
        plt.imshow(im_stack[batch_ind,:,:,led_ind])
        plt.colorbar()
    
        vmin = np.min(im_stack[batch_ind,:,:,led_ind])
        vmax = np.max(im_stack[batch_ind,:,:,led_ind])
        
        plt.figure()
        plt.title('im_stack emulated')    
        plt.imshow(im_stack_final[batch_ind,:,:,led_ind], vmin=vmin, vmax=vmax)
        plt.colorbar()
        
    
        
    
