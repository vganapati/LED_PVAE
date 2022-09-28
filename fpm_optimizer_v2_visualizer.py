#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:03:20 2022

@author: vganapa1
"""

import numpy as np
import matplotlib.pyplot as plt
from fpm_functions import F, Ft

## Input
visualize_trim = 64
##

path = 'dataset_sea_urchin/training/example_000000/reconstruction/all_leds_iter_10000_l2_0.001_lr_0.01_b_10_s_1'
hr_computed = np.load(path + '/computed_obj.npy')
lr_observed_stack = np.load(path + '/lr_observed_stack.npy')
lr_calc_stack_final = np.load(path + '/lr_calc_stack_final.npy')
num_slices = np.load(path + '/num_slices.npy')
loss_vec = np.load(path + '/loss_vec.npy')
pupil_angle_final = np.load(path + '/pupil_angle_final.npy')
Ns = np.load(path + '/Ns.npy')
Ns_0 = np.load(path + '/Ns_0.npy')
NAfilter_synthetic = np.load(path + '/NAfilter_synthetic.npy')

upsample_factor = int(hr_computed.shape[1]/pupil_angle_final.shape[0])


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
    