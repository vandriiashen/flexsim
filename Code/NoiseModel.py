import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

class NoiseModel(object):
    def __init__(self, proj_shape):
        self.proj_shape = proj_shape
    
    def get_flatfield_intensity(self, voltage):
        return 11000.
    
    def create_flatfield_image(self, voltage):
        ff = np.zeros(self.proj_shape, dtype=float)
        ff.fill(self.get_flatfield_intensity(voltage))
        return ff
    
    def add_poisson(self, proj):
        # True value
        # g = 1./2.4
        # Arbitrary one
        g = 1./20.
        res = np.random.poisson(g * proj)
        res = res.astype(float) / g
        return res
    
    def add_gaussian(self, proj):
        proj += np.random.normal(0, 20.)
        return proj
    
    def add_blur(self, proj):
        for i in range(proj.shape[1]):
            proj[:,i,:] = ndimage.gaussian_filter(proj[:,i,:], sigma = 0.7)
        return proj
    
    def add_noise(self, proj):
        res = self.add_poisson(proj)
        res = self.add_gaussian(res)
        res = self.add_blur(res)
        return res
