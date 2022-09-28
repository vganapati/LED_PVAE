#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:23:31 2022

@author: vganapa1
"""

import numpy as np
import matplotlib.pyplot as plt

iterative = True

if iterative:
    # Iterative solution
    save_tag_recons = '_single_lr4'
    subfolder_name = 'dataset_frog_blood_v3/training/example_000000/reconstruction' + save_tag_recons
    filepath = subfolder_name + '/full_field.npy'
else:
    # Neural network solution
    save_path = 'frog_mult6_v3_100k'
    restore_num = 'None' # 'None' if using the latest checkpoint
    filepath = save_path + '/full_field_restore_' + restore_num + '.npy'
    subfolder_name = save_path

full_field = np.load(filepath)
num_slices = full_field.shape[0]
visualize_trim = 64

for ss in range(num_slices):
    plt.figure()
    plt.title('slice ' + str(ss) + ' amplitude')
    plt.imshow(np.abs(full_field[ss,visualize_trim:-visualize_trim,
                                  visualize_trim:-visualize_trim]))
    plt.colorbar()
    # plt.savefig(subfolder_name + '/full_field_amp.png')
    
    plt.figure()
    plt.title('slice ' + str(ss) + ' angle')
    plt.imshow(np.angle(full_field[ss,visualize_trim:-visualize_trim,
                                    visualize_trim:-visualize_trim]))
    plt.colorbar()
    # plt.savefig(subfolder_name + '/full_field_angle.png')