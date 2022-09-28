#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:37:09 2021

@author: vganapa1
"""

import tensorflow as tf
from tensorflow import keras
from helper_functions import InstanceNormalization
import numpy as np
from fractions import Fraction

def find_conv_output_dim(input_length, stride, kernel_size):
    # finds output dimension for 'valid' padding
    output_length = np.zeros([input_length,],dtype=np.int32)
    output_length[0::stride]=1
    output_length = output_length[:-kernel_size+1]
    output_length = np.sum(output_length)  
    return output_length

def create_encode_net(image_x,
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
                      initial_repeats, # how many times I_m is repeated
                      encode_net_ind=0,                      
                      feature_maps_multiplier=2, 
                      real_data=False,
                      ):
      
    num_feature_maps_vec = feature_maps_multiplier*np.array(num_feature_maps_vec)

    input_Im = tf.keras.Input(shape=(image_x,image_y,1), 
                              batch_size = batch_size, name='I_m')

    input_Im_rescaled = tf.image.resize(input_Im, [image_x_r,image_y_r], 
                                        method='nearest',
                                        preserve_aspect_ratio=False,
                                        antialias=False, name='resize'
                                        )
    
    input_alpha = tf.keras.Input(shape=(num_leds,), 
                                  batch_size=batch_size,
                                  name='alpha_sample')
    
    if real_data:
        input_coords_xm = tf.keras.Input(shape=(image_x,image_y), 
                                  batch_size = batch_size, name='coords_xm')
    
        input_coords_ym = tf.keras.Input(shape=(image_x,image_y), 
                                  batch_size = batch_size, name='coords_ym')
        
        input_coords_xm_rescaled = tf.expand_dims(input_coords_xm, -1)
        input_coords_xm_rescaled = tf.image.resize(input_coords_xm_rescaled, [image_x_r,image_y_r], 
                                            method='nearest',
                                            preserve_aspect_ratio=False,
                                            antialias=False, name='resize'
                                            )
        input_coords_ym_rescaled = tf.expand_dims(input_coords_ym, -1)
        input_coords_ym_rescaled = tf.image.resize(input_coords_ym_rescaled, [image_x_r,image_y_r], 
                                            method='nearest',
                                            preserve_aspect_ratio=False,
                                            antialias=False, name='resize'
                                            )
        coords = tf.concat((input_coords_xm_rescaled,input_coords_ym_rescaled), axis=-1)
    

    

    # repeat input_Im
    extra_repeats = feature_maps_multiplier - (initial_repeats+num_leds)%feature_maps_multiplier
        
    input_Im_repeat = tf.repeat(input_Im_rescaled, initial_repeats+extra_repeats, axis=-1)
    
    # repeat input_alpha
    input_alpha_repeat = tf.expand_dims(input_alpha,axis=-2)
    input_alpha_repeat = tf.expand_dims(input_alpha_repeat,axis=-2)
    input_alpha_repeat = tf.repeat(input_alpha_repeat, image_x_r, axis=-3)
    input_alpha_repeat = tf.repeat(input_alpha_repeat,image_y_r,axis=-2)
    
    
    # combine input_Im and input_alpha
    combined_input = tf.concat((input_Im_repeat,input_alpha_repeat), axis=-1)

    if dropout_prob == 0:
        apply_dropout = False
    else:
        apply_dropout = True
        

    # want channel to be divisible by feature_maps_multiplier
    repeats = Fraction(coords.shape[-1]/feature_maps_multiplier).denominator
    coords_repeat = tf.repeat(coords,repeats,axis=-1)
    output = tf.concat([combined_input,coords_repeat],axis=-1)
    
    # Downsampling through the model
    skips_val = []
    skips_weight = []
    skips_pixel_x = []
    skips_pixel_y = []
    skips_pixel_z = []


    for i in range(num_blocks):
        
        # intermediate layers, no residual connection
        # conv with stride = 1 and padding = same
        for l in range(intermediate_layers):
            output = \
                conv_block(output, # input
                           output.shape[-1], # output size channels
                           intermediate_kernel,
                           apply_norm = apply_norm, norm_type = norm_type,
                           apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                           initializer = initializer, 
                           transpose = False,
                           stride = (1,1), 
                           )   
        
        output_val, output_weight = tf.split(output, [output.shape[-1]//2, output.shape[-1]//2], axis=-1, num=None, name='split')
        skips_val.append(output_val)
        skips_weight.append(output_weight) 
        skips_pixel_x.append(output.shape[1])
        skips_pixel_y.append(output.shape[2])
        skips_pixel_z.append(output.shape[3])

        
        output = \
            conv_block(output, # input
                       num_feature_maps_vec[i], # output size channels
                       kernel_size,
                       apply_norm = apply_norm, norm_type = norm_type,
                       apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                       initializer = initializer, 
                       transpose = False,
                       stride = (stride_encode, stride_encode), 
                       )


    output_val, output_weight = tf.split(output, [output.shape[-1]//2, output.shape[-1]//2], axis=-1, num=None, name='split')
    skips_val.append(output_val)
    skips_weight.append(output_weight)
    
    skips_pixel_x.append(output.shape[1])
    skips_pixel_y.append(output.shape[2])        
    skips_pixel_z.append(output.shape[3])

    if real_data:
        inputs = (input_Im, input_alpha, input_coords_xm, input_coords_ym)
    else:
        inputs = (input_Im, input_alpha)
        
    model = keras.Model(inputs = inputs, outputs = (skips_val,skips_weight), name='encode_net_' + str(encode_net_ind))
    
    model.summary()
    
    return(model, skips_pixel_x, skips_pixel_y, skips_pixel_z)



def create_decode_net(skips_pixel_x,
                      skips_pixel_y,
                      skips_pixel_z,
                      num_leds,
                      num_zernike_coeff,
                      batch_size,
                      final_output_channels, # number of output channels
                      kernel_size, 
                      stride_encode,
                      apply_norm, norm_type, 
                      initializer,
                      dropout_prob,
                      intermediate_layers,
                      intermediate_kernel,
                      net_number = 0,
                      feature_maps_multiplier = 2,
                      use_first_skip = True,
                      real_data=False,
                      change_Ns=False,
                      vary_pupil=False
                      ):
    
    skips = []
    for skip in range(len(skips_pixel_x)):
        skip_input = tf.keras.Input(shape=(skips_pixel_x[skip],
                                           skips_pixel_y[skip],
                                           skips_pixel_z[skip]//feature_maps_multiplier), 
                                    batch_size = batch_size,
                                    name=str(skip))
        skips.append(skip_input)
    

            
    if dropout_prob == 0:
        apply_dropout = False
    else:
        apply_dropout = True    
    
    output = skips[-1]
    skips_reverse = reversed(skips[:-1])
    skips_pixel_z_reverse = reversed(skips_pixel_z[:-1])

    
    
    # Upsampling and establishing the skip connections
    for i, skip in enumerate(skips_reverse):
        
        output_channels = next(skips_pixel_z_reverse)

        # intermediate layers, don't change shape
        # conv with stride = 1 and padding = same
        for l in range(intermediate_layers):
            output = \
                conv_block(output, # input
                           output.shape[-1], # output size channels
                           intermediate_kernel,
                           apply_norm = apply_norm, norm_type = norm_type,
                           apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                           initializer = initializer, 
                           transpose = False,
                           stride = (1,1), 
                           )     


        output = conv_block(output, # input
                            output_channels, # output size channels
                            kernel_size, 
                            apply_norm = apply_norm, norm_type=norm_type,
                            apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                            initializer = initializer, 
                            transpose = True,
                            stride = (stride_encode,stride_encode),
                            )
        

        skip_x = skip.shape[1]
        skip_y = skip.shape[2]
        
        # crop output
        remove_pad_x = output.shape[1] - skip_x
        remove_pad_y = output.shape[2] - skip_y
        
        r_x = remove_pad_x%2
        r_y = remove_pad_y%2
    
        output = output[:,remove_pad_x//2+r_x:remove_pad_x//2+r_x+skip_x,remove_pad_y//2+r_y:remove_pad_y//2+r_y+skip_y,:]
        
        if i==(len(skips_pixel_x)-2) and not(use_first_skip):
            pass
        else:
            output = tf.keras.layers.Concatenate()([output, skip])


    # final set of intermediate layers
    for l in range(intermediate_layers):
        output = \
            conv_block(output, # input
                        output.shape[-1], # output size channels
                        intermediate_kernel,
                        apply_norm = apply_norm, norm_type = norm_type,
                        apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                        initializer = initializer, 
                        transpose = False,
                        stride = (1,1), 
                        )   
    
    # reconstruction output
    output = \
    conv_block(output, # input
               final_output_channels*2, # output size channels
               kernel_size,
               apply_norm = apply_norm, norm_type = norm_type,
               apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
               initializer = initializer, 
               transpose = False,
               stride = (1,1),
               )   

    output_mean, output_var = tf.split(output, [output.shape[-1]//2, output.shape[-1]//2], 
                                       axis=-1, num=None, name='split')

    if real_data or vary_pupil:
        output_small = \
        conv_block(skips[-1], # input
                   1, # output size channels
                   kernel_size,
                   apply_norm = apply_norm, norm_type = norm_type,
                   apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                   initializer = initializer, 
                   transpose = False,
                   stride = (1,1),
                   )   
        output_flattened = tf.keras.layers.Flatten()(output_small)
        
        # zernike coeff output (effectively delta, as we always initialize zernike coeff with zeros)

        zernike_coeff = dense_block(output_flattened, # input
                                    num_zernike_coeff*2, # extra 2 for probability distribution output
                                    apply_norm = apply_norm, norm_type=norm_type,
                                    apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                                    initializer = initializer,
                                    bias_initializer = initializer,
                                    use_bias = True,
                                   )
        

        zernike_coeff_mean, zernike_coeff_var = tf.split(zernike_coeff, [num_zernike_coeff,num_zernike_coeff], 
                                           axis=-1, num=None, name='split')    
        
        zernike_coeff_mean = zernike_coeff_mean/1e5
        zernike_coeff_var = zernike_coeff_var-10
        if change_Ns:
            # Ns delta output
            Ns_delta = dense_block(output_flattened, # input
                                   num_leds*2*2, # Ns is num_leds x 2, extra 2 for probability distribution output
                                   apply_norm = apply_norm, norm_type=norm_type,
                                   apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                                   initializer = initializer,
                                   bias_initializer = initializer,
                                   use_bias = True,
                                   )
            Ns_delta = Ns_delta/100
            Ns_delta = tf.keras.layers.Reshape([num_leds,2*2])(Ns_delta)
            Ns_delta_mean, Ns_delta_var = tf.split(Ns_delta, [2, 2], 
                                               axis=-1, num=None, name='split')     
            
             
            
            # cos theta delta output
    
            cos_delta = dense_block(output_flattened, # input
                                    num_leds*2, # extra 2 for probability distribution output
                                    apply_norm = apply_norm, norm_type=norm_type,
                                    apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                                    initializer = initializer,
                                    bias_initializer = initializer,
                                    use_bias = True,
                                    )
            cos_delta = cos_delta/1e6
            cos_delta_mean, cos_delta_var = tf.split(cos_delta, [num_leds,num_leds], 
                                               axis=-1, num=None, name='split') 
            
            # poisson noise multiplier delta
    
            pnm_delta = dense_block(output_flattened, # input
                                    2, # extra 2 for probability distribution output
                                    apply_norm = apply_norm, norm_type=norm_type,
                                    apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                                    initializer = initializer,
                                    bias_initializer = initializer,
                                    use_bias = True,
                                    )
    
            pnm_delta_mean, pnm_delta_var = tf.split(pnm_delta, [1,1], 
                                               axis=-1, num=None, name='split')     
            outputs = (output_mean, output_var, Ns_delta_mean, Ns_delta_var, zernike_coeff_mean, zernike_coeff_var, 
                       cos_delta_mean, cos_delta_var, pnm_delta_mean, pnm_delta_var)
        else:
            outputs = (output_mean, output_var, zernike_coeff_mean, zernike_coeff_var, 
                       )
    else:
        outputs = (output_mean, output_var)
        
    model = keras.Model(inputs = (skips), outputs = outputs, name='decode_net_' + str(net_number))

    model.summary()
    return model





def process_flattened_input(flattened_input,
                            num_dense_layers,
                            intermediate_dim,
                            output_last_dim,
                            final_shape,
                            apply_norm,
                            norm_type,
                            apply_dropout,
                            dropout_prob,
                            initializer,
                            bias_initializer,
                            use_bias):
    for layer_ind in range(num_dense_layers):
        if layer_ind == num_dense_layers - 1:
            output_dim = output_last_dim
        else:
            output_dim = intermediate_dim
        
        flattened_input = dense_block(flattened_input, # input
                                      output_dim,
                                      apply_norm = apply_norm, 
                                      norm_type=norm_type,
                                      apply_dropout=apply_dropout, 
                                      dropout_prob = dropout_prob, 
                                      initializer = initializer,
                                      bias_initializer = bias_initializer,
                                      use_bias=use_bias
                                      )
    flattened_input = tf.keras.layers.Reshape(final_shape)(flattened_input)
    return flattened_input
    


def create_pi_net2(input_x,
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
                  bias_initializer = 'glorot_uniform',
                  scale_factor_vec = None, # scale factor is subtractive
                  use_bias = False,
                  pi_iter = False):
    '''
    This function outputs both patterns and time_fraction.
    '''
    if dropout_prob == 0:
        apply_dropout = False
    else:
        apply_dropout = True  
     
    # output of encode_net is the input to pi_net
    inputs = tf.keras.Input(shape=(input_x,input_y,input_z), name='output_encode')
    if pi_iter:
            num_iter = tf.keras.Input(shape=(1,), name='num_iter')
    
    flattened_input = tf.keras.layers.Flatten()(inputs)
    if pi_iter:
        flattened_input = tf.concat((flattened_input,num_iter*tf.ones_like(flattened_input)), axis=-1)
    
    output = \
    process_flattened_input(flattened_input,
                            num_dense_layers,
                            upsample,
                            num_leds*num_patterns*2,
                            [num_leds, num_patterns*2],
                            apply_norm,
                            norm_type,
                            apply_dropout,
                            dropout_prob,
                            initializer,
                            bias_initializer,
                            use_bias)       
    

    output_patterns, output_time = tf.split(output, [num_patterns, num_patterns], axis=-1, num=None, name='split')

    flattened_output_time = tf.keras.layers.Flatten()(output_time)
    output_time = \
    process_flattened_input(flattened_output_time,
                            1, # num_dense_layers
                            upsample,
                            num_patterns,
                            [num_patterns,],
                            apply_norm,
                            norm_type,
                            apply_dropout,
                            dropout_prob,
                            initializer,
                            bias_initializer,
                            use_bias)     

    # rescaling layers
    # output_patterns = keras.layers.experimental.preprocessing.Rescaling(scale = scale_factor)(output_patterns)
    # output_time = keras.layers.experimental.preprocessing.Rescaling(scale = scale_factor)(output_time)
        
    if pi_iter:
        one_hot = tf.one_hot(tf.cast(num_iter, tf.int32), max_steps-1)
        scale_factor = tf.expand_dims(tf.reduce_sum(scale_factor_vec*one_hot,axis=-1),axis=-2)
        output_patterns = tf.add(output_patterns, -scale_factor)

    if pi_iter:
        model = keras.Model(inputs = (inputs, num_iter), outputs = (output_patterns), name='pi_net')
    else:
        model = keras.Model(inputs = (inputs), outputs = (output_patterns), name='pi_net')
    
    model.summary()
    return model




def dense_block(x, # input
                output_last_dim,
                apply_norm = False, norm_type='batchnorm',
                apply_dropout=False, dropout_prob = 0, 
                initializer = 'glorot_uniform',
                bias_initializer = 'zeros',
                use_bias = False,
                ):
    """
    Dropout => Dense => Maxout => Batchnorm

    """
    
    if apply_dropout:
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    x1 = tf.keras.layers.Dense(output_last_dim, activation=None, use_bias=use_bias, kernel_initializer=initializer,
                          bias_initializer=bias_initializer)(x)

    x2 = tf.keras.layers.Dense(output_last_dim, activation=None, use_bias=use_bias, kernel_initializer=initializer,
                          bias_initializer=bias_initializer)(x)
    
    ## Maxout
    x = tf.maximum(x1, x2)

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
      elif norm_type.lower() == 'instancenorm':
        x = InstanceNormalization()(x)

    return x


def periodic_padding(image, padding_tuple): # padding is added to beginnings and ends of x and y
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    https://github.com/tensorflow/tensorflow/issues/956
    
    usage example:

    image = tf.reshape(tf.range(30, dtype='float32'), shape=[5,6])
    padded_image = periodic_padding(image, padding=2)


    '''

    padding_0,padding_1 = padding_tuple[0]
    partial_image = image
 
 
 
    if padding_0 != 0:  
        upper_pad = tf.repeat(image,int(np.ceil(padding_0/image.shape[1])),axis=1)[:,-padding_0:,:,:]
        # upper_pad = image[:,-padding_0:,:,:]
        partial_image = tf.concat([upper_pad, partial_image], axis=1)
        
    if padding_1 != 0: 
        lower_pad = tf.repeat(image,int(np.ceil(padding_1/image.shape[1])),axis=1)[:,:padding_1,:,:]
        # lower_pad = image[:,:padding_1,:,:]
        partial_image = tf.concat([partial_image, lower_pad], axis=1)


    padded_image = partial_image
    padding_0,padding_1 = padding_tuple[1]
  
    if padding_0 != 0:   
        left_pad = tf.repeat(partial_image,int(np.ceil(padding_0/image.shape[2])),axis=2)[:,:,-padding_0:,:]
        # left_pad = partial_image[:,:,-padding_0:,:]
        padded_image = tf.concat([left_pad, padded_image], axis=2)
        
    if padding_1 != 0:
        right_pad = tf.repeat(partial_image,int(np.ceil(padding_1/image.shape[2])),axis=2)[:,:,:padding_1,:]
        # right_pad = partial_image[:,:,:padding_1,:]
        padded_image = tf.concat([padded_image, right_pad], axis=2)
        
    return padded_image



def conv_block(x, # input
               output_last_dim, # output size channels
               kernel_size,
               apply_norm = False, norm_type='batchnorm',
               apply_dropout=False, dropout_prob = 0, 
               initializer = 'glorot_uniform', 
               bias_initializer='zeros',
               transpose = False,
               stride = (2,2),
               use_bias = True,
               ):
    """
    Dropout => Conv2D => Maxout => Batchnorm

    """
    stride_x, stride_y = stride
    
    if apply_dropout:
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    
    if transpose:
        x1 = keras.layers.Conv2DTranspose(output_last_dim, (kernel_size, kernel_size), 
                                          strides=stride, padding='same',
                                          dilation_rate=(1, 1), 
                                          use_bias=use_bias, 
                                          kernel_initializer=initializer,   
                                          bias_initializer=bias_initializer,
                                          output_padding=None,
                                          )(x)
        
        x2 = keras.layers.Conv2DTranspose(output_last_dim, (kernel_size, kernel_size), 
                                          strides=stride, padding='same',
                                          dilation_rate=(1, 1), 
                                          use_bias=use_bias, 
                                          kernel_initializer=initializer,  
                                          bias_initializer=bias_initializer,
                                          output_padding=None,
                                          )(x)        
        
        # # Add output_padding_x and output_padding_y
        
        # r_x = output_padding_x%2
        # r_y = output_padding_y%2
        
        # x1 = periodic_padding(x1,((output_padding_x//2+r_x,output_padding_x//2),(output_padding_y//2+r_y,output_padding_y//2)))
        # x2 = periodic_padding(x2, ((output_padding_x//2+r_x,output_padding_x//2),(output_padding_y//2+r_y,output_padding_y//2)))
        
        # x1 = tf.pad(x1, ((0,0),(output_padding_x//2+r_x,output_padding_x//2),(output_padding_y//2+r_y,output_padding_y//2),(0,0)), \
        #             mode = "SYMMETRIC")   
        
        # x2 = tf.pad(x2, ((0,0),(output_padding_x//2+r_x,output_padding_x//2),(output_padding_y//2+r_y,output_padding_y//2),(0,0)), \
        #     mode = "SYMMETRIC")  
        
    else:
        input_x = x.shape[-3]
        input_y = x.shape[-2]
        
        # Add padding such that the convolution doesn't change the input shape beyond stride effects

        if input_x%stride_x:
            pad_x = kernel_size - input_x%stride_x
        else: # no remainder
            pad_x = kernel_size - stride_x
        
        if input_y%stride_y:
            pad_y = kernel_size - input_y%stride_y
        else: # no remainder
            pad_y = kernel_size - stride_y
 
        r_x = pad_x%2
        r_y = pad_y%2


        x = periodic_padding(x,((pad_x//2+r_x,pad_x//2),(pad_y//2+r_y,pad_y//2)))

        # x = tf.pad(x, ((0,0),(pad_x//2+r_x,pad_x//2),(pad_y//2+r_y,pad_y//2),(0,0)), mode = "SYMMETRIC")        


        x1 = keras.layers.Conv2D(output_last_dim, (kernel_size, kernel_size), strides=stride, padding='valid',
                                 kernel_initializer=initializer, 
                                 bias_initializer=bias_initializer,
                                 use_bias=use_bias)(x)
        x2 = keras.layers.Conv2D(output_last_dim, (kernel_size, kernel_size), strides=stride, padding='valid',
                                       kernel_initializer=initializer, 
                                       bias_initializer=bias_initializer,
                                       use_bias=use_bias)(x)

    
    ## Maxout
    x = tf.maximum(x1, x2)

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
      elif norm_type.lower() == 'instancenorm':
        x = InstanceNormalization()(x)

    return(x)


