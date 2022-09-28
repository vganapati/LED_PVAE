#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:28:53 2022

@author: vganapa1

Downloads reconstruction from server
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from SyntheticMNIST_functions import create_folder
import argparse

### COMMAND LINE ARGS ###


parser = argparse.ArgumentParser(description='Get command line args')


parser.add_argument('--save_tag_recons', action='store', help='path for results is /reconstruction_save_name', \
                    default = '_multiplexed_p1')
    
    
args = parser.parse_args()

### INPUTS ###

input_path = 'dataset_frog_blood_v3'
dataset_type = 'training'
obj_ind = 0
save_tag_recons = args.save_tag_recons


visualize_trim = 1


### END OF INPUTS ###


folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)
subfolder_name = folder_name + '/reconstruction' + save_tag_recons
print(subfolder_name)


# os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name \
#           + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/')


create_folder(subfolder_name)

# download full_field
os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name + '/full_field.npy' \
          + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/' + subfolder_name)
# download full_field_window
os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name + '/full_field_window.npy' \
          + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/' + subfolder_name)

print('saved in ' + subfolder_name)

