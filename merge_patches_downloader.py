#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:28:53 2022

@author: vganapa1

Downloads reconstruction from server
"""
import os
import argparse
from SyntheticMNIST_functions import create_folder

### COMMAND LINE ARGS ###


parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--input_path', action='store', help='path(s) to overall folder containing training data',
                    default = 'dataset_frog_blood_v3')

parser.add_argument('--save_tag_recons', action='store', help='path for results is /reconstruction_save_tag_recons', \
                    )

parser.add_argument('--obj', type=int, action='store', dest='obj_ind', \
                        help='obj number to reconstruct', default = 0)
    
parser.add_argument('--da', action='store_true', dest='download_all', 
                    help='downloads all patches, otherwise just downloads the full field reconstruction') 
    
args = parser.parse_args()

### INPUTS ###

input_path = args.input_path
dataset_type = 'training'
obj_ind = args.obj_ind
save_tag_recons = args.save_tag_recons
download_all = args.download_all
### END OF INPUTS ###

folder_name = '{}/{}/example_{:06d}'.format(input_path, dataset_type, obj_ind)
subfolder_name = folder_name + '/reconstruction' + save_tag_recons

if download_all:
    os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name \
              + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/' + folder_name)
else:
    create_folder(subfolder_name)
    os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name \
              + '/full_field.npy' + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/' + subfolder_name)
    os.system('scp -r ganapativ@login2.int.janelia.org:/groups/funke/home/ganapativ/FPM_VAE/' + subfolder_name \
              + '/full_field_window.npy' + ' /Users/vganapa1/Dropbox/Github/FPM_VAE/' + subfolder_name)
        
print('results saved in ' + subfolder_name)


