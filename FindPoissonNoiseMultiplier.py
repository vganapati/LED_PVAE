#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:03:41 2018

@author: vganapa1
"""

import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

### Inputs ###
Nx = 2048
Ny = 2048
input_folder_name = "/Users/vganapa1/Dropbox/Github/E90_Yolanda_Andrew_FPM/030322/Noise"
exposure = 50 # ms
remove_img = [13] # remove these images due to some problem in imaging
bit_depth = 16

# Get file names 
filenames = np.sort(glob.glob(input_folder_name + "/*.png"))

remove_filenames = []
for i in remove_img:
    remove_filenames.append(input_folder_name + '/' + str(i) + '.png')

num_images = len(filenames) - len(remove_img)
all_imgs = np.zeros([Nx,Ny,num_images])
    
for i in range(num_images):    
    print(filenames[i])
    if filenames[i] not in remove_filenames:
        img = imageio.imread(filenames[i])
        all_imgs[:,:,i] = img/(2**bit_depth - 1)/exposure
   
# plot the histogram for a random pixel

a = all_imgs[1024,1024,:]   
plt.figure()   
plt.hist(a, bins='auto') 

a = all_imgs[512,512,:]   
plt.figure()   
plt.hist(a, bins='auto') 


# get the mean and standard deviation for each point

mean = np.mean(all_imgs, axis=2, dtype=np.float32)
std = np.std(all_imgs, axis=2, dtype=np.float32)


# reshape to a vector
mean = np.reshape(mean,[Nx*Ny,])
std = np.reshape(std,[Nx*Ny,])


# define X, Y
X = mean # np.sqrt(mean)
X = sm.add_constant(X)
Y = std

# regression, plot on top of scatterplot of X, Y

results = sm.OLS(Y,X).fit()

print(results.summary())

plt.figure()
plt.scatter(X[:,1],Y,s=0.1)

X_plot = np.linspace(0,np.max(mean),1000)
X_plot = sm.add_constant(X_plot)
plt.plot(X_plot[:,1], np.sum(X_plot*np.expand_dims(results.params,axis=0),axis=1),'r', linewidth=1)

plt.xlabel('mean')
plt.ylabel('standard deviation')
  
plt.savefig('poisson_noise.png',bbox_inches='tight', dpi = 300)     

    

# fit is aX+b

print('fit is a + b*X')
print('a is: ', results.params[0])
print('b is: ', results.params[1])