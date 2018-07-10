
# coding: utf-8

# In[1]:


import os
import h5py
import argparse
import numpy as np

from hdf5_utils import *

cars_hdf5_name = '../models/cars/2_watertight/sdf.h5'

h5 = read_hdf5(cars_hdf5_name)
print h5.shape

S = h5.reshape((h5.shape[0], -1))
print S.shape


# In[8]:


S_mean = np.mean(S, axis=0)
S_white = S - S_mean
t = S.shape[0] # number of samples, 87
n = S.shape[1] # dim of feature, 32*32*32

cov = 1./(t-1) * np.dot(S_white.T , S_white)
print cov.shape


# In[ ]:


from numpy import linalg as LA
w, v = LA.eig(cov)


# In[ ]:


print w.shape, v.shape

