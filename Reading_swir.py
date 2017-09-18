"""
@uthor: Anthony Ortiz
Creation Date: 09/11/2017
Last Modification date: 09/11/2017

Reading SWIR Image using Python
"""
import h5py
import numpy as np


file_path = 'your_directory/r_swir_reg.mat'
swir = {}
f = h5py.File(file_path, libver='earliest')
swir = np.array(f['r_swir_reg'])

#You can now scale the values between 0-1 or 
#0-255 and plot different images
