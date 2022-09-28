#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:02:01 2022

@author: vganapa1
"""

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

plt.rcParams.update({'font.size': 22})

vmax=0.16236609913962916
vmin=-0.10173261940236053

orientation='horizontal' # horizontal or vertical

a = np.array([[vmin,vmax]])
pl.figure(figsize=(9, 1.5))
img = pl.imshow(a, cmap="gray") # cmap is 'Greens' for alpha plots, 'gray' for others
pl.gca().set_visible(False)
pl.colorbar(ticks=a[0],orientation=orientation)
pl.savefig("colorbar.png",bbox_inches='tight',dpi=600)