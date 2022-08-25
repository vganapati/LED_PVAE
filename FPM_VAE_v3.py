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
from SyntheticMNIST_functions import create_folder, F, Ft
from helper_pattern_opt import load_img_stack, process_alpha
from helper_functions import trim_lit_coord
from visualizer_functions import show_figs2, \
                                 show_figs_alpha, show_figs_input_output
from models_v2 import create_encode_net, \
                      create_pi_net2, \
                      create_decode_net
from FPM_VAE_helper_functions_v3 import find_loss_vae, create_dataset_iter, \
                                        find_loss_vae_unsup, calculate_log_prob_M_given_R
from FPM_unrolled_iterative import find_loss_vae_unrolled
import os

# tf.config.experimental.set_device_policy('warn')
# tf.debugging.enable_check_numerics()

tfd = tfp.distributions

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


setup_start_time = time.time()

### Command line args ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--carone', action='store_true', dest='carone', 
                    help='use carone cells data for training') #XXX visualization needs to be implemented

parser.add_argument('--CT', action='store_true', dest='CT', 
                    help='computed tomography mode') #XXX needs to be implemented

parser.add_argument('--ufs', action='store_true', dest='use_first_skip', 
                    help='use the first skip connection in the unet')

parser.add_argument('--unsup', action='store_true', dest='unsupervised', 
                    help='unsupervised mode to do M-->z-->R-->M')

parser.add_argument('--unrolled', action='store_true', dest='unrolled', 
                    help='train an unrolled iterative solve')

parser.add_argument('--pg', action='store_true', dest='projected_grad', 
                    help='uses l1 norm in unrolled iterative solve')

parser.add_argument('--nii', type=int, action='store', dest='num_inner_iter', \
                        help='unrolled iterations for the unrolled iterative solve', \
                        default = 100)   

parser.add_argument('--cp', action='store_true', dest='choose_patterns', 
                    help='choose patterns ahead of time if in unsupervised mode')

parser.add_argument('--normal', action='store_true', dest='use_normal', 
                    help='use a normal distribution as final distribution') 

parser.add_argument('--det', action='store_true', dest='deterministic', 
                    help='no latent variable, simply maximizes log probability of output_dist') 

parser.add_argument('--ta', action='store_true', dest='train_alpha', 
                    help='trains the alpha patterns') 

parser.add_argument('--use_mep', action='store_true', dest='use_model_encode_pi', 
                    help='creates and trains a separate model_encode_pi for input to pi_net') 

parser.add_argument('--rand_i', action='store_true', dest='random_init', 
                    help='randomize initial alpha (turn off for the 2 freq dataset tests)') 

parser.add_argument('--use_coords', action='store_true', dest='use_coords', 
                    help='append a coords mat in model_encode')

parser.add_argument('--use_bias', action='store_true', dest='use_bias', 
                    help='use bias terms in the pi net')

parser.add_argument('--train', action='store_true', dest='train', 
                    help='run the training loop')

parser.add_argument('--validate', action='store_true', dest='final_validate', 
                    help='run all the validation examples through the final trained net')

parser.add_argument('--final_train', action='store_true', dest='final_train', 
                    help='run all the training examples through the final trained net')
     
parser.add_argument('--sfv', action='store', type=float, help='scaling for each step in model_pi', 
                     nargs='+', dest='scale_factor_vec')    

parser.add_argument('--input_path_vec', action='store', help='path(s) to overall folder(s) containing training and test data', 
                     nargs='+')

parser.add_argument('--test_input_path', action='store', help='path to folder containing test data', default = None)


parser.add_argument('--save_path', action='store', help='path to save output', default = None)

parser.add_argument('--save_tag_mult', action='store', help='folder name describing the multiplexed image', dest = 'save_tag_multiplexed',
                    default = None)

parser.add_argument('--restore_patt', action='store', help='path to restore alpha for unsupervised', default = None)

parser.add_argument('-b', type=int, action='store', dest='batch_size', \
                        help='batch size', \
                        default = 4)    

parser.add_argument('--sci', type=int, action='store', dest='skip_connect_ind', \
                        help='batch size', \
                        default = -3)     

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier', 
                    help='poisson noise multiplier, higher value means higher SNR', default = (2**16-1)*0.41)
    
parser.add_argument('--sfd', type=float, action='store', dest='scale_factor_dist', 
                    help='scale factor for the normal unsupervised VAE dist', default = 0)
    

parser.add_argument('--lr', type=float, action='store', dest='learning_rate', 
                        help='learning rate', default = 1e-3)

parser.add_argument('--ae', type=float, action='store', dest='adam_epsilon', 
                        help='adam_epsilon', default = 1e-7)

parser.add_argument('--dm', type=float, action='store', dest='dirichlet_multiplier', 
                        help='dirichlet_multiplier', default = 1)

parser.add_argument('-i', type=int, action='store', dest='num_iter', \
                        help='number of training iterations', default = 100)

parser.add_argument('-p', type=int, action='store', dest='num_patterns', \
                        help='number of illumination patterns PER step', \
                        default = 2)

parser.add_argument('--ms', type=int, action='store', dest='max_steps', \
                        help='maximum number of outer iterations (i.e. steps)', default = 3)

parser.add_argument('--fsp', type=int, action='store', dest='first_step_patterns', \
                        help='number of illumination patterns for the FIRST step, can be different than num_patterns', \
                        default = 1) #XXX DEACTIVATED
    
parser.add_argument('--si', type=int, action='store', dest='save_interval', \
                        help='save_interval for checkpoints and intermediate values', default = 50000)

parser.add_argument('--reconstruct', action='store_true', dest='reconstruct', \
                    help='reconstruct object instead of just completing image stack')
    
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
                    help='anneal the stand dev of the dist P(M|O)', \
                    default=0)
    
# model parameters


parser.add_argument('--ndlp', type=int, action='store', dest='num_dense_layers_pi', \
                    help='number of dense layers in pi net', default = 3)

parser.add_argument('--pi_iter', action='store_true', dest='pi_iter', 
                    help='if passed, use the iteration number in the pi net')    
    
# for model_encode
parser.add_argument('--nfm0', type=int, action='store', dest='num_feature_maps_0', \
                        help='number of features/2 in the first block of model_encode', default = 49)

parser.add_argument('--nfmm0', type=float, action='store', dest='num_feature_maps_multiplier_0', \
                        help='multiplier of features for each block of model_encode', default = 1.1)
    
parser.add_argument('--ks0', type=int, action='store', dest='kernel_size_0',
                    help='kernel size in model_encode_I_m', default = 4)
    
parser.add_argument('--se0', type=int, action='store', dest='stride_encode_0',
                        help='convolution stride in model_encode_I_m', default = 2)

parser.add_argument('--nb', type=int, action='store', dest='num_blocks', \
                        help='num convolution blocks in model_encode_I_m and model_encode_alpha', default = 3)

# intermediate layers
parser.add_argument('--il', type=int, action='store', dest='intermediate_layers', \
                        help='intermediate_layers for model_encode', default = 2)

parser.add_argument('--ik', type=int, action='store', dest='intermediate_kernel', \
                    help='intermediate_kernel for model_encode', default = 4)

# for model_encode_R (same params as model_encode except for channel dimension)
parser.add_argument('--nfm1', type=int, action='store', dest='num_feature_maps_1', \
                        help='number of features/2 in the first block of model_encode', default = 1)

parser.add_argument('--nfmm1', type=float, action='store', dest='num_feature_maps_multiplier_1', \
                        help='multiplier of features for each block of model_encode', default = 1.1)
    
    
parser.add_argument('--pro', type=float, action='store', dest='pr_offset', \
                        help='offset for positive range function', default = np.finfo(np.float32).eps.item() )
    

# visualization parameters

parser.add_argument('--test', action='store_true', dest='test', 
                    help='run the trained model on a test image in visualization step')

parser.add_argument('--visualize', action='store_true', dest='visualize', 
                    help='visualize results')

parser.add_argument('--en', type=int, action='store', dest='example_num', \
                        help='example number in file name for visualization', default = 1)

parser.add_argument('--pi', type=int, action='store', dest='pattern_ind', \
                        help='pattern index for visualization of I_multiplexed', default = 0)

parser.add_argument('--imi', type=int, action='store', dest='img_ind', \
                        help='LED number for visualization', default = 1)
    

args = parser.parse_args()



#########################

### Parse command line args ###


carone = args.carone

save_tag_multiplexed = args.save_tag_multiplexed

unrolled = args.unrolled
num_inner_iter = args.num_inner_iter #100
projected_grad = args.projected_grad #False

CT = args.CT
unsupervised = args.unsupervised
choose_patterns = args.choose_patterns
use_normal = args.use_normal
deterministic = args.deterministic
train_alpha = args.train_alpha
random_init = args.random_init

restore_patt = args.restore_patt

dirichlet_multiplier = args.dirichlet_multiplier

train = args.train
final_validate = args.final_validate
final_train = args.final_train
test = args.test
visualize = args.visualize
save_path = args.save_path
norm = args.norm
pi_iter = args.pi_iter
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
save_interval = args.save_interval
num_iter = args.num_iter
dropout_prob = args.dropout_prob
reconstruct = args.reconstruct

restore = args.restore
restore_num = args.restore_num
use_latest_ckpt = args.use_latest_ckpt

use_coords = args.use_coords
use_bias = args.use_bias

input_path_vec = args.input_path_vec
test_input_path = args.test_input_path

kl_multiplier = args.kl_multiplier
kl_anneal_factor = args.kl_anneal_factor

truncate_dataset = args.truncate_dataset
num_patterns = args.num_patterns
max_steps = args.max_steps
first_step_patterns = num_patterns #args.first_step_patterns # DEACTIVATED
num_dense_layers_pi = args.num_dense_layers_pi

batch_size = args.batch_size

use_model_encode_pi = args.use_model_encode_pi

# for model_encode
num_feature_maps_0 = args.num_feature_maps_0
num_feature_maps_multiplier_0 = args.num_feature_maps_multiplier_0
kernel_size_0 = args.kernel_size_0
stride_encode_0 = args.stride_encode_0
num_blocks = args.num_blocks

# itermediate_layers
intermediate_layers = args.intermediate_layers
intermediate_kernel = args.intermediate_kernel

# for model_encode_R
num_feature_maps_1 = args.num_feature_maps_1
num_feature_maps_multiplier_1 = args.num_feature_maps_multiplier_1

pr_offset = args.pr_offset

# Example to visualize

example_num = args.example_num # overall example, starting index

### Create folder for output ###
create_folder(save_path)  

try:
    input_path = input_path_vec[0]
    
    image_x = int(np.load(input_path + '/image_x.npy'))
    image_y = int(np.load(input_path + '/image_y.npy'))
except TypeError:
    image_x = 2048
    image_y = 2048

if reconstruct:
    image_x_r = int(np.load(input_path + '/image_x_r.npy'))
    image_y_r = int(np.load(input_path + '/image_y_r.npy'))
else:
    image_x_r = image_x
    image_y_r = image_y

normalizer_0 = np.load(input_path + '/normalizer.npy')
normalizer_ang_0 = np.load(input_path + '/normalizer_ang.npy')

offset_0 = np.load(input_path + '/offset.npy')
offset_ang_0 = np.load(input_path + '/offset_ang.npy')

### Set parameters ###

sqrt_reg = np.finfo(np.float32).eps.item() # regularizing sqrt in backprop
poisson_noise_multiplier = args.poisson_noise_multiplier #(2**16-1)*0.41

### Load parameters ###

try:
    LitCoord = np.load(input_path + '/LitCoord.npy')
except NameError:
    LitCoord = np.load('/data2/CaroneLabCells/TrainingDataset/Images_SampleNumber0001_RegionNumber0001/patch_0/LitCoord.npy')
LitCoord2 = trim_lit_coord(LitCoord)
LitCoord2 = tf.constant(LitCoord2)
num_leds_length = LitCoord2.shape[0]

N_obj = np.load(input_path + '/N_obj.npy')
Ns = np.load(input_path + '/Ns.npy')
P = np.load(input_path + '/pupil.npy')
Np = np.load(input_path + '/Np.npy')
LED_vec = np.load(input_path + '/LED_vec.npy')
LEDs_used_boolean = np.load(input_path + '/LEDs_used_boolean.npy')
num_slices = np.load(input_path + '/num_slices.npy')
H_scalar = np.load(input_path + '/H_scalar.npy')
H_scalar_f = np.load(input_path + '/H_scalar_f.npy')
NAfilter_synthetic = np.load(input_path + '/NAfilter_synthetic.npy')
LED_vec_i = LED_vec[LEDs_used_boolean]

if unrolled:
    zernike_mat = np.load(input_path + '/zernike_mat.npy')
    zernike_mat = np.transpose(zernike_mat, axes=[2,0,1])
    zernike_mat = np.expand_dims(zernike_mat, axis=0)

#########################

### Create data input pipeline ###


if CT:
    # x_train_sinograms, theta, num_proj_pix = get_sinograms(save_path_dataset)
    pass
else:
    train_ds, train_ds_no_shuffle, val_ds, num_leds, \
    r_channels, load_img_stack2, \
    train_folders, val_folders= create_dataset_iter(input_path_vec,
                                                    save_path,
                                                    restore,
                                                    reconstruct,
                                                    carone,
                                                    truncate_dataset,
                                                    batch_size,
                                                    num_patterns,
                                                    unsupervised,
                                                    choose_patterns,
                                                    example_num,
                                                    restore_patt,
                                                    dirichlet_multiplier,
                                                    save_tag_multiplexed = save_tag_multiplexed,
                                                    )



#########################

### Neural Networks ###
#########################

if random_init:
    alpha_np = np.random.rand(num_leds,num_patterns) - args.scale_factor_vec[0]
else:
    alpha_np = np.ones([num_leds,num_patterns]) - args.scale_factor_vec[0]

alpha_0 = tf.Variable(alpha_np, dtype=tf.dtypes.float32, name = 'alpha_0')

if unrolled:
    inner_learning_rate = tf.Variable(1e-2, dtype=tf.dtypes.float32, 
                                      name = 'inner_learning_rate')
    t2_reg = tf.Variable(1e-2, dtype=tf.dtypes.float32, name = 't2_reg')
    

    

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

initializer = tf.keras.initializers.GlorotUniform()
# initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
apply_norm = False
norm_type = 'batchnorm', #'instancenorm'

num_feature_maps_vec_0 = [int(num_feature_maps_0*num_feature_maps_multiplier_0**i) for i in range(num_blocks)]
num_feature_maps_vec_1 = [int(num_feature_maps_1*num_feature_maps_multiplier_1**i) for i in range(num_blocks)]

if unsupervised and not(deterministic):
    feature_maps_multiplier = 4
else:
    feature_maps_multiplier = 2


model_encode, skips_pixel_x, skips_pixel_y, skips_pixel_z = \
create_encode_net(image_x,
                  image_y, 
                  image_x_r,
                  image_y_r,
                  num_leds,
                  num_feature_maps_vec_0,
                  batch_size,
                  num_blocks, 
                  kernel_size_0, 
                  stride_encode_0,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  encode_net_ind=1,
                  use_coords=use_coords,
                  coords = coords,
                  feature_maps_multiplier=feature_maps_multiplier)


model_encode_pi, skips_pixel_x_pi, skips_pixel_y_pi, skips_pixel_z_pi = \
create_encode_net(image_x,
                  image_y, 
                  image_x_r,
                  image_y_r,
                  num_leds,
                  num_feature_maps_vec_0,
                  batch_size,
                  num_blocks, 
                  kernel_size_0, 
                  stride_encode_0,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  encode_net_ind = 0,
                  use_coords = use_coords,
                  coords = coords,
                  feature_maps_multiplier=feature_maps_multiplier,
                  )

if not(use_model_encode_pi):
    skips_pixel_x_pi = skips_pixel_x
    skips_pixel_y_pi = skips_pixel_y
    skips_pixel_z_pi = skips_pixel_z
    
skip_connect_ind = args.skip_connect_ind

input_x = skips_pixel_x_pi[skip_connect_ind]
input_y = skips_pixel_y_pi[skip_connect_ind]
input_z = skips_pixel_z_pi[skip_connect_ind]//2
num_dense_layers = num_dense_layers_pi
upsample = num_leds*num_patterns*2

model_pi = \
    create_pi_net2(input_x,
                   input_y,
                   input_z,
                   num_leds,
                   num_patterns,
                   initializer,
                   num_dense_layers,
                   upsample,
                   dropout_prob,
                   apply_norm,
                   norm_type,
                   max_steps,
                   bias_initializer = 'zeros', #'glorot_uniform'
                   scale_factor_vec = tf.constant(args.scale_factor_vec[1:], dtype=tf.float32),
                   use_bias = use_bias,
                   pi_iter = pi_iter)


model_encode_R, skips_pixel_x_R, skips_pixel_y_R, skips_pixel_z_R = \
create_encode_net(image_x,
                  image_y, 
                  image_x_r,
                  image_y_r,
                  num_leds,
                  num_feature_maps_vec_1,
                  batch_size,
                  num_blocks, 
                  kernel_size_0, 
                  stride_encode_0,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  encode_net_ind=2,
                  use_coords=use_coords,
                  coords = coords,
                  feature_maps_multiplier=4,
                  append_r = True,
                  r_channels=r_channels)

if deterministic or unsupervised:
    skips_pixel_z_decode = skips_pixel_z
else:
    skips_pixel_z_decode = skips_pixel_z[:-1] + [skips_pixel_z[-1]+skips_pixel_z_R[-1]//2]

model_decode = \
create_decode_net(skips_pixel_x,
                  skips_pixel_y,
                  skips_pixel_z_decode,
                  batch_size,
                  r_channels, # number of output channels
                  kernel_size_0, 
                  stride_encode_0,
                  apply_norm, norm_type, 
                  initializer,
                  dropout_prob,
                  intermediate_layers,
                  intermediate_kernel,
                  net_number = 0,
                  feature_maps_multiplier = feature_maps_multiplier,
                  reconstruct = reconstruct,
                  use_first_skip = args.use_first_skip,
                  scale_factor_dist = args.scale_factor_dist)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, \
                                     epsilon=adam_epsilon)



# save checkpoints

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(save_path, checkpoint_dir, "ckpt")


if unsupervised:
    checkpoint = tf.train.Checkpoint(model_encode = model_encode,
                                     model_decode = model_decode,
                                     optimizer = optimizer,
                                     kl_anneal = kl_anneal,
                                     anneal_std = anneal_std,
                                     )    
else:
    checkpoint = tf.train.Checkpoint(model_encode_pi = model_encode_pi,
                                     model_pi = model_pi,
                                     model_encode = model_encode,
                                     model_encode_R = model_encode_R,
                                     model_decode = model_decode,
                                     optimizer = optimizer,
                                     alpha_0 = alpha_0,
                                     kl_anneal = kl_anneal,
                                     )       

if restore:
    # restore a checkpoint
    if use_latest_ckpt:
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(save_path, checkpoint_dir)))
    else:
        checkpoint.restore(os.path.join(save_path, checkpoint_dir,'ckpt-')+str(restore_num))


# prior on z, the latent variable 


if unsupervised:
    skip_shapes = np.array([batch_size*np.ones_like(skips_pixel_x), skips_pixel_x, skips_pixel_y, np.array(skips_pixel_z)//feature_maps_multiplier]).T
    if use_normal:
        prior = [tfd.Normal(loc=tf.zeros(skip_shapes[i]), scale=1) for i in range(num_blocks+1)]
    else:
        prior = [tfd.Beta(tf.ones(skip_shapes[i]), tf.ones(skip_shapes[i])) for i in range(num_blocks+1)]
else:
    skip_shapes = np.array([batch_size*np.ones_like(skips_pixel_x_R), skips_pixel_x_R, skips_pixel_y_R, np.array(skips_pixel_z_R)//4]).T
    if use_normal:
        prior = tfd.Normal(loc=tf.zeros(skip_shapes[-1]), scale=1)
    else:
        prior = tfd.Beta(tf.ones(skip_shapes[-1]), tf.ones(skip_shapes[-1]))

if unsupervised:
    trainable_vars = model_encode.trainable_variables + \
                      model_decode.trainable_variables
elif unrolled:
    trainable_vars = [alpha_0, t2_reg, inner_learning_rate] 
    if max_steps>1:
        trainable_vars += model_encode_pi.trainable_variables + model_pi.trainable_variables
else:
    trainable_vars = []
    if train_alpha:
        trainable_vars += [alpha_0]   
    
    trainable_vars += model_pi.trainable_variables + \
                      model_encode.trainable_variables + \
                      model_decode.trainable_variables

    if use_model_encode_pi:
        trainable_vars += model_encode_pi.trainable_variables
    
    if not(deterministic):
        trainable_vars += model_encode_R.trainable_variables

'''
# To test the find_loss_vae function:

path, im_stack, im_stack_r = next(train_ds)
training = True

# To test the find_loss_unsup function

path, im_stack, im_stack_r, alpha = next(train_ds)
training = True
'''




@tf.function
def train_step(im_stack,
               im_stack_r,
               training,
               use_prior = False,
               unsupervised = False,
               alpha = None,
               im_stack_multiplexed = None,
               ):

    alpha = alpha[:,:,0:num_patterns]
    
    with tf.GradientTape(watch_accessed_variables=True, persistent=False) as tape:
        tape.watch(trainable_vars)
        if unsupervised:
            loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
            output_dist, \
            q, q_sample, kl_divergence, loglik = find_loss_vae_unsup(im_stack,
                                                                     im_stack_r,
                                                                     alpha,
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
                                                                     reconstruct,
                                                                     normalizer_0,
                                                                     normalizer_ang_0,
                                                                     offset_0,
                                                                     offset_ang_0,
                                                                     use_prior = use_prior, # does not use conditional q(z|M)
                                                                     kl_anneal = kl_anneal,
                                                                     kl_multiplier=kl_multiplier,
                                                                     pr_offset = pr_offset,
                                                                     use_normal = use_normal,
                                                                     N_obj = N_obj,
                                                                     Ns = Ns,
                                                                     P = P,
                                                                     Np = Np,
                                                                     LED_vec = LED_vec,
                                                                     LEDs_used_boolean = LEDs_used_boolean,
                                                                     num_slices = num_slices,
                                                                     H_scalar = H_scalar,
                                                                     H_scalar_f = H_scalar_f,
                                                                     deterministic = deterministic,
                                                                     use_first_skip = args.use_first_skip,
                                                                     anneal_std = anneal_std,
                                                                     im_stack_multiplexed = im_stack_multiplexed,
                                                                     )
        elif unrolled:

            loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
                obj_actual, hr_guess = \
                find_loss_vae_unrolled(im_stack,
                                        im_stack_r,
                                        image_x,
                                        image_y,
                                        image_x_r,
                                        image_y_r,
                                        skips_pixel_x_pi,
                                        skips_pixel_y_pi, 
                                        skips_pixel_z_pi,
                                        num_blocks,  
                                        num_leds,
                                        model_encode_pi,
                                        model_pi,
                                        pi_iter,
                                        poisson_noise_multiplier,
                                        sqrt_reg,
                                        num_patterns,
                                        max_steps,
                                        batch_size,
                                        alpha_0,
                                        first_step_patterns,
                                        training,
                                        skip_connect_ind,
                                        normalizer_0,
                                        offset_0,
                                        normalizer_ang_0,
                                        offset_ang_0,
                                        LED_vec_i,
                                        N_obj,
                                        num_slices,
                                        zernike_mat,
                                        inner_learning_rate,
                                        adam_epsilon,
                                        num_inner_iter,
                                        Ns, 
                                        P, # pupil
                                        Np,
                                        H_scalar, H_scalar_f,
                                        projected_grad, t2_reg,
                                        optimize_pupil_ang = False,
                                        )

            output_dist = hr_guess
            q = obj_actual
            q_sample = None
            kl_divergence = None
            loglik = None
        else:
            loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
            output_dist, \
            q, q_sample, kl_divergence, loglik = find_loss_vae(im_stack,
                                                               im_stack_r,
                                                               image_x,
                                                               image_y,
                                                               image_x_r,
                                                               image_y_r,
                                                               skips_pixel_x_pi,
                                                               skips_pixel_y_pi, 
                                                               skips_pixel_z_pi,
                                                               skips_pixel_z,
                                                               skips_pixel_z_R,
                                                               num_blocks,  
                                                               num_leds,
                                                               use_model_encode_pi,
                                                               model_encode_pi,
                                                               model_encode,
                                                               model_encode_R,
                                                               model_pi,
                                                               model_decode,
                                                               pi_iter,
                                                               poisson_noise_multiplier,
                                                               sqrt_reg,
                                                               num_patterns,
                                                               max_steps,
                                                               batch_size,
                                                               alpha_0,
                                                               first_step_patterns,
                                                               prior,
                                                               training,
                                                               skip_connect_ind,
                                                               normalizer_0,
                                                               offset_0,
                                                               use_prior=use_prior,
                                                               kl_anneal = kl_anneal,
                                                               kl_multiplier = kl_multiplier,
                                                               pr_offset = pr_offset,
                                                               use_normal = use_normal,
                                                               deterministic = deterministic,
                                                               )
            

        loss_M_VAE = tf.reduce_mean(loss_M_VAE)/1e5
            
    if training:
     
        # loss_M_VAE          
        gradients = tape.gradient(loss_M_VAE, trainable_vars)
        gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
        gradients = [tf.clip_by_norm(g, norm)
                      for g in gradients]   
        optimizer.apply_gradients(zip(gradients, trainable_vars))   
     
    
    return(loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
           output_dist, q, q_sample, kl_divergence, loglik)


if train:

    train_loss_vec = []
    val_loss_vec = []
    iter_vec = []
    
    if num_iter == 0:
        # placeholder
        start_time = time.time()
        
    ### Training and Validation ###
    for iter_i in range(num_iter):
        # print(tf.config.experimental.get_memory_usage("GPU:0"))
        
        kl_anneal.assign(tf.minimum(tf.maximum(kl_anneal*kl_anneal_factor,0),100))
        
        anneal_std.assign(anneal_std*.9999)
        # print(anneal_std)
        
        if unsupervised:
            if save_tag_multiplexed is not None:
                path, im_stack, im_stack_r, alpha, im_stack_multiplexed = next(train_ds)
            else:
                path, im_stack, im_stack_r, alpha = next(train_ds)
                im_stack_multiplexed = None
            loss_M_VAE, _, _, _, _, _, _, _ = train_step(im_stack, 
                                                      im_stack_r, 
                                                      True,
                                                      unsupervised = unsupervised,
                                                      alpha = alpha,
                                                      im_stack_multiplexed = im_stack_multiplexed,
                                                     )
        else:
            path, im_stack, im_stack_r = next(train_ds)
            loss_M_VAE, _, _, _, _, _, _, _ = train_step(im_stack, 
                                                      im_stack_r, 
                                                      True,
                                                     )
        
        print('Iteration number: ' + str(iter_i))
        print('Training loss_M_VAE: ' + str(loss_M_VAE))
        train_loss_vec.append(loss_M_VAE)

        if np.isnan(loss_M_VAE):
            sys.exit()
        
  
        
        if iter_i == 0:
            setup_end_time = time.time()
            print('Setup took ' + str((setup_end_time-setup_start_time)/60) + ' minutes.')
            start_time = time.time()

        
        if((iter_i%save_interval == 0) or (iter_i == num_iter-1)):
            if unsupervised:
                pass
            else:
                path, im_stack, im_stack_r = next(val_ds)
                loss_M_VAE, _, _, _, _, _, _, _ = train_step(im_stack, 
                                                          im_stack_r, 
                                                          False,
                                                         )

            print('Validation loss_M_VAE: ' + str(loss_M_VAE))
            val_loss_vec.append(loss_M_VAE)       
            iter_vec.append(iter_i)
            
            np.save(save_path + '/train_loss_vec.npy', train_loss_vec)
            np.save(save_path + '/val_loss_vec.npy', val_loss_vec)
            np.save(save_path + '/iter_vec.npy', iter_vec)
            
            checkpoint.save(file_prefix = checkpoint_prefix)
            
            
    end_time = time.time()
    print('Training took ' + str((end_time-start_time)/60) + ' minutes.')
    

    show_figs2(save_path, 
               iter_vec,
               train_loss_vec,
               val_loss_vec)
    

### End of Training ###

def renorm(im_stack, im_stack_r, normalizer_1, offset_1, normalizer_ang_1, offset_ang_1,
           im_stack_multiplexed=None):
    im_stack = ((im_stack/normalizer_1 + offset_1)-offset_0)*normalizer_0 
    if im_stack_multiplexed is not None:
        im_stack_multiplexed = im_stack_multiplexed/normalizer_1*normalizer_0 
    if reconstruct:
        im_stack_r_0, im_stack_r_1 = tf.split(im_stack_r, 2, axis=-1)
        im_stack_r_0 = (im_stack_r_0/normalizer_ang_1[0]+offset_ang_1[0] - offset_ang_0[0])*normalizer_ang_0[0] 
        im_stack_r_1 = (im_stack_r_1/normalizer_ang_1[1]+offset_ang_1[1] - offset_ang_0[1])*normalizer_ang_0[1] 
        im_stack_r = tf.concat((im_stack_r_0,im_stack_r_1),axis=-1)
    else:
        im_stack_r = im_stack
    return(im_stack,im_stack_r,im_stack_multiplexed)

def output_point_estimate(output_dist, sample_ind=0):
    '''
    Function is created only for the --reconstruct option.
    '''
    
    output_mean = output_dist[sample_ind].sample()
    # filter the output mean by the Synthetic NA
    obj_re, obj_im = tf.split(output_mean,2,axis=-1)
    obj_re = obj_re/normalizer_ang_0[0] + offset_ang_0[0]
    obj_im = obj_im/normalizer_ang_0[1] + offset_ang_0[1]
    obj = tf.cast(obj_re, tf.complex64) + tf.cast(obj_im, tf.complex64)*1j
    
    all_filtered_obj = []
    for b in range(batch_size):
        obj_reals = []
        obj_imags = []
        filtered_obj_vec = []
        for s in range(num_slices):
            obj_i = obj[b,:,:,s]
            # filter by sythetic NA
            O = F(obj_i)
            O = O*NAfilter_synthetic
            filtered_obj = Ft(O) #low resolution field
            filtered_obj_vec.append(filtered_obj)
    
            obj_reals.append((np.real(filtered_obj) - offset_ang_0[0])*normalizer_ang_0[0])
            obj_imags.append((np.imag(filtered_obj) - offset_ang_0[1])*normalizer_ang_0[1])
        
        # Un-normalized reconstructed object
        final_obj_unnorm = tf.stack(filtered_obj_vec,axis=-1)
        all_filtered_obj.append(final_obj_unnorm)
        
    all_filtered_obj = tf.stack(all_filtered_obj,axis=0)
    return(all_filtered_obj)

def final_evaluation(ds = train_ds, dataset_type = 'training', folders = train_folders,
                     normalizer_1 = normalizer_0, 
                     offset_1 = offset_0,
                     normalizer_ang_1 = normalizer_ang_0,
                     offset_ang_1 = offset_ang_0):
    print('Starting final ' + dataset_type + '...')
    start_time = time.time()
    
    val_ds_iter = iter(ds)
    
    loss_final_val = []
    path_final_val = []
    # all_filtered_obj_vec = []
    # all_im_stack_multiplexed = []
    
    val_size = len(folders)
    
    for val_ind in range(1): # single batch # range(val_size//batch_size): # all batches
        if unsupervised:
            if save_tag_multiplexed is not None:
                path, im_stack, im_stack_r, alpha, im_stack_multiplexed = next(val_ds_iter)
            else:
                path, im_stack, im_stack_r, alpha = next(val_ds_iter)
                im_stack_multiplexed = None
        else:
            path, im_stack, im_stack_r = next(val_ds_iter)
            
        im_stack, im_stack_r, im_stack_multiplexed = renorm(im_stack, im_stack_r, 
                                                            normalizer_1, offset_1, normalizer_ang_1, offset_ang_1,
                                                            im_stack_multiplexed)
        
        # all_im_stack_multiplexed.append(im_stack_multiplexed)
        create_folder(save_path + '/' + dataset_type)
        np.save(save_path + '/' + dataset_type  + '/im_stack_multiplexed' + str(val_ind) + '.npy', 
                im_stack_multiplexed)

        if unsupervised:
            loss_M_VAE, _, _, output_dist, _, _, _, _ = train_step(im_stack, im_stack_r, False, unsupervised=unsupervised, alpha=alpha, 
                                                         im_stack_multiplexed = im_stack_multiplexed)
        else:
            loss_M_VAE, _, _, output_dist, _, _, _, _ = train_step(im_stack, im_stack_r, False)
        
        if reconstruct:
            all_filtered_obj = output_point_estimate(output_dist)
            # all_filtered_obj_vec.append(all_filtered_obj)
            np.save(save_path + '/' + dataset_type  + '/all_filtered_obj' + str(val_ind) + '.npy', 
                    all_filtered_obj)
            
            #save entropy
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
    

if final_validate:
    final_evaluation(ds = val_ds, dataset_type = 'val', folders = val_folders)



if final_train:
    
    final_evaluation(ds = train_ds_no_shuffle, dataset_type = 'training', folders = train_folders)
    
    
if test_input_path is not None: # XXX does not include predetermined im_stack_multiplexed
    print('Starting final test...')
    
    # load normalizers and offsets
    
    normalizer_1 = np.load(test_input_path + '/normalizer.npy')
    normalizer_ang_1 = np.load(test_input_path + '/normalizer_ang.npy')
    
    offset_1 = np.load(test_input_path + '/offset.npy')
    offset_ang_1 = np.load(test_input_path + '/offset_ang.npy')
    
    # Create test_ds
    test_ds, _, num_leds, \
    r_channels, load_img_stack2, \
    test_folders, _ = create_dataset_iter([test_input_path],
                                          None, #save_path
                                          False, #restore,
                                          reconstruct,
                                          carone,
                                          truncate_dataset, # XXX truncate_dataset for test is the same as for train
                                          batch_size,
                                          num_patterns,
                                          unsupervised,
                                          choose_patterns,
                                          example_num,
                                          restore_patt,
                                          dirichlet_multiplier,
                                          test = True, # makes val_ds empty
                                          )
    
    final_evaluation(ds = test_ds, dataset_type = 'test', folders = test_folders,
                     normalizer_1 = normalizer_1, 
                     offset_1 = offset_1,
                     normalizer_ang_1 = normalizer_ang_1,
                     offset_ang_1 = offset_ang_1)
    


### End of final train and validation and test ###

def load_batch(test_path_vec, alpha_vec, normalizer_1, offset_1, normalizer_ang_1, offset_ang_1): # vector of im_stack paths, len == batch_size
    im_stack=[]
    im_stack_r = []
    alpha = []
    im_stack_multiplexed = []
    for ind, test_path in enumerate(test_path_vec):
        print(test_path)
        if unsupervised:
            if save_tag_multiplexed is not None:
                path, im_stack_i, im_stack_i_r, alpha_i, im_stack_multiplexed_i = load_img_stack(test_path, num_leds, num_patterns,
                                                                                                 r_channels, bit_depth = 16, 
                                                                                                 reconstruct = reconstruct,
                                                                                                 unsupervised = unsupervised,
                                                                                                 choose_patterns = choose_patterns,
                                                                                                 alpha = alpha_vec[ind],
                                                                                                 save_tag_multiplexed = save_tag_multiplexed,
                                                                                                 )
                im_stack_multiplexed_i = tf.expand_dims(im_stack_multiplexed_i, axis=0)
                im_stack_multiplexed.append(im_stack_multiplexed_i)
            else:
                path, im_stack_i, im_stack_i_r, alpha_i = load_img_stack(test_path, num_leds, num_patterns,
                                                                         r_channels, bit_depth = 16, 
                                                                         reconstruct = reconstruct,
                                                                         unsupervised = unsupervised,
                                                                         choose_patterns = choose_patterns,
                                                                         alpha = alpha_vec[ind],
                                                                         save_tag_multiplexed = save_tag_multiplexed,
                                                                         )
            alpha_i = tf.expand_dims(alpha_i, axis=0) # give alpha_i a batch dimension
            alpha.append(alpha_i)
            

        
        else:
            path, im_stack_i, im_stack_i_r = load_img_stack(test_path, num_leds, num_patterns,
                                                            r_channels, bit_depth = 16, 
                                                            reconstruct = reconstruct,
                                                            )
        
        im_stack_i = tf.expand_dims(im_stack_i, axis=0) # give im_stack a batch dimension
        im_stack.append(im_stack_i)
        
        im_stack_i_r = tf.expand_dims(im_stack_i_r, axis=0) # give im_stack a batch dimension
        im_stack_r.append(im_stack_i_r)

    im_stack = tf.concat(im_stack,axis=0)
    im_stack_r = tf.concat(im_stack_r,axis=0)
    if unsupervised:
        alpha = tf.concat(alpha,axis=0)
        if save_tag_multiplexed is not None:
            im_stack_multiplexed = tf.concat(im_stack_multiplexed, axis=0)

    if len(im_stack_multiplexed)==0:
        im_stack_multiplexed = None
    # re-normalize
    im_stack, im_stack_r, im_stack_multiplexed = renorm(im_stack, im_stack_r, normalizer_1, offset_1, normalizer_ang_1, offset_ang_1, im_stack_multiplexed)

    test_path = test_path_vec[batch_ind]
    
    return(test_path, im_stack, im_stack_r, alpha, im_stack_multiplexed)



if visualize:

    # iter_ind is action number
    pattern_ind = args.pattern_ind # pattern ind for displaying multiplexed image
    img_ind = args.img_ind # LED number or r_channel ind
    batch_ind = 0
    sample_ind = 1
    
    if test:
        # data_folder = 'test'
        input_path = test_input_path
        data_folder = 'training'
        normalizer_test = np.load(input_path + '/normalizer.npy')
        normalizer_test_ang = np.load(input_path + '/normalizer_ang.npy')
        offset_test = np.load(input_path + '/offset.npy')
        offset_ang_test = np.load(input_path + '/offset_ang.npy')
    else:
        data_folder = 'training'
        normalizer_test = normalizer_0
        normalizer_test_ang = normalizer_ang_0
        offset_test = offset_0
        offset_ang_test = offset_ang_0


    
    test_path_vec = []
    alpha_vec = []
    for batch_i in range(example_num, example_num+batch_size):
        test_path = '{}/{}/example_{:06d}'.format(input_path, data_folder, example_num)  
        test_path_vec.append(test_path)
        if unsupervised:
            if restore_patt is not(None):
                alpha = np.expand_dims(np.load(restore_patt + '/all_alpha_train.npy')[example_num],axis=0)
            else:
                if truncate_dataset==1:
                    alpha = np.load(save_path + '/all_alpha_train.npy')
                else:
                    alpha = np.expand_dims(np.load(save_path + '/all_alpha_train.npy')[example_num],axis=0)
            alpha_vec.append(alpha)
    if unsupervised:
        alpha_vec = tf.concat(alpha_vec,axis=0)
    
    test_path, im_stack, im_stack_r, alpha, im_stack_multiplexed = load_batch(test_path_vec, alpha_vec, normalizer_test, offset_test, normalizer_test_ang, offset_ang_test)
    
    if unsupervised:
        use_prior = False
    else:
        use_prior = True
        
    loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
           output_dist, q, q_sample, kl_divergence, loglik = train_step(im_stack, im_stack_r, False,
                                                                        use_prior = use_prior,
                                                                        unsupervised = unsupervised,
                                                                        alpha = alpha,
                                                                        im_stack_multiplexed = im_stack_multiplexed,
                                                                        )
    if unrolled:
        hr_guess = output_dist
        obj_actual = q
        
    create_folder(test_path + '/' + save_path)
    np.save(test_path + '/' + save_path + '/im_stack_multiplexed.npy', im_stack_multiplexed_vec)
    np.save(test_path + '/' + save_path + '/alpha_vec.npy', alpha_vec)
    
    '''       
    loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
     output_dist, q, q_sample, kl_divergence, loglik \
               = train_step(im_stack, im_stack_r, False,use_prior = False)
    print(tf.reduce_sum(prior.log_prob(prior.sample()),axis=[1,2,3]))
    print(tf.reduce_sum(prior.log_prob(q_sample),axis=[1,2,3]))
    '''
    
    try:
        train_loss_vec = np.load(save_path + '/train_loss_vec.npy')
        val_loss_vec = np.load(save_path + '/val_loss_vec.npy')
        iter_vec = np.load(save_path + '/iter_vec.npy')
        
        show_figs2(save_path, 
                   iter_vec,
                   train_loss_vec,
                   val_loss_vec)
    except FileNotFoundError:
        pass
    
    if unsupervised:
        max_steps = 1
        
    for iter_ind in range(max_steps):
        if unsupervised:
            alpha_scaled = alpha_vec
            im_stack_multiplexed = im_stack_multiplexed_vec
        else:
            alpha_i = alpha_vec[iter_ind]
            alpha_scaled = process_alpha(alpha_i, sqrt_reg)
            im_stack_multiplexed = im_stack_multiplexed_vec[iter_ind,:,:,:,:]
            
        show_figs_alpha(save_path, 
                        alpha_scaled,
                        batch_size,
                        im_stack_multiplexed,
                        data_folder,
                        example_num,
                        iter_ind,
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
    if not(unrolled):
        # output_mean = output_dist[sample_ind].mean()
        output_mean = output_dist[sample_ind].sample()
        if not(reconstruct):    
            create_folder(test_path + '/' + save_path)
            np.save(test_path + '/' + save_path + '/im_stack_mean.npy', output_mean)
    
        show_figs_input_output(save_path, 
                               data_folder,
                               im_stack_r,
                               output_mean,
                               batch_ind,
                               img_ind)
    
    
    if reconstruct:
        if unrolled:
            obj = hr_guess[batch_ind]
        else:
            # filter the output mean by the Synthetic NA
            obj_re, obj_im = tf.split(output_mean,2,axis=-1)
            obj_re = obj_re/normalizer_ang_0[0] + offset_ang_0[0]
            obj_im = obj_im/normalizer_ang_0[1] + offset_ang_0[1]
            obj = tf.cast(obj_re, tf.complex64) + tf.cast(obj_im, tf.complex64)*1j
            obj = obj[batch_ind]
        obj_reals = []
        obj_imags = []
        filtered_obj_vec = []
        for s in range(num_slices):
            obj_i = obj[:,:,s]
            # filter by sythetic NA
            O = F(obj_i)
            O = O*NAfilter_synthetic
            filtered_obj = Ft(O) #low resolution field
            filtered_obj_vec.append(filtered_obj)

            obj_reals.append((np.real(filtered_obj) - offset_ang_0[0])*normalizer_ang_0[0])
            obj_imags.append((np.imag(filtered_obj) - offset_ang_0[1])*normalizer_ang_0[1])
        
        obj_reals = tf.stack(obj_reals, axis=-1)
        obj_imags = tf.stack(obj_imags, axis=-1)
        output_mean_filtered = tf.concat((obj_reals,obj_imags),axis=-1)
        output_mean_filtered = tf.expand_dims(output_mean_filtered, axis=0)
        # add batch ind
        output_mean_filtered = tf.repeat(output_mean_filtered, batch_size, axis=0)
        
        # Un-normalized reconstructed object
        final_obj_unnorm = tf.stack(filtered_obj_vec,axis=-1)
        np.save(test_path + '/' + save_path + '/final_obj.npy', final_obj_unnorm)
        
        # XXX FIX to be true object error
        # object_MSE_error = tf.reduce_mean((output_mean_filtered - im_stack_r)**2)
    
        for img_ind in range(r_channels):
            show_figs_input_output(save_path, 
                                   data_folder,
                                   im_stack_r,
                                   output_mean_filtered,
                                   batch_ind,
                                   img_ind)


    if unsupervised:
        im_stack_multiplexed_final, im_stack_final = \
            calculate_log_prob_M_given_R(output_mean, # im_stack_r # batch_size x image_x x image_y x num_leds
                                         tf.expand_dims(alpha[:,:,0:num_patterns],axis=0), # expand for max_steps, dims are: max_steps x batch_size x num_leds x num_patterns
                                         batch_size,
                                         poisson_noise_multiplier,
                                         sqrt_reg,
                                         1, # max_steps
                                         normalizer_0,
                                         normalizer_ang_0,
                                         offset_0,
                                         offset_ang_0,
                                         reconstruct = reconstruct,
                                         N_obj = N_obj,
                                         Ns = Ns,
                                         P = P,
                                         Np = Np,
                                         LED_vec = LED_vec,
                                         LEDs_used_boolean = LEDs_used_boolean,
                                         num_slices = num_slices,
                                         H_scalar = H_scalar,
                                         H_scalar_f = H_scalar_f,
                                         visualize=True
                                         )
    
        import matplotlib.pyplot as plt
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
        

        
    elif not(unrolled):

        loss_M_VAE, alpha_vec, im_stack_multiplexed_vec, \
           output_dist, q, q_sample, kl_divergence, loglik = train_step(im_stack, im_stack_r, False,
                                                         use_prior = False)
        show_figs_input_output(save_path, 
                               data_folder,
                               im_stack_r,
                               output_dist[sample_ind].mean(),
                               batch_ind,
                               img_ind)
