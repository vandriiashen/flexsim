import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

class NoiseModel(object):
    '''Class providing parameters for noise simulation. Right now it only works for a monochromatic simulation with all parameters provided in the config for a certain tube voltage.
    
    :param proj_shape: Array with 3 dimensions: (height, number of angles, width)
    :type config: :class:`np.ndarray`
    :param proj_shape: Dictionary of noise parameters
    :type config: :class:`dict`
    
    '''
    def __init__(self, proj_shape, noise_config):
        '''Constructor method.
        '''
        self.proj_shape = proj_shape
        self.flatfield = noise_config['Flatfield']
        self.poisson_scaling = noise_config['Poisson_scaling']
        self.gaussian_std = noise_config['Gaussian_std']
        self.blur_width = noise_config['Blur_width']
    
    def get_flatfield_intensity(self):
        '''Gets the flatfield intensity level specified in the config.
        
        :return: Flatfield intensity value
        :rtype: :class:`float`
        
        '''
        return self.flatfield
    
    def create_flatfield_image(self):
        '''Creates a uniform image with a shape of projection filled with flatfield intensity value.
        
        :return: Flatfield image
        :rtype: :class:`np.ndarray`
        
        '''
        ff = np.zeros(self.proj_shape, dtype=np.float32)
        ff.fill(self.get_flatfield_intensity())
        return ff
    
    def add_poisson(self, proj):
        '''Adds Poisson noise to the image: I -> P(alpha I). Scaling a is taken from the config.
        
        :param proj: Original image
        :type proj: :class:`np.ndarray`
        :return: Noisy image
        :rtype: :class:`np.ndarray`
        
        '''
        g = self.poisson_scaling
        
        if g == 0.:
            return proj
        
        res = np.random.poisson(g * proj)
        res = res.astype(float) / g
        return res
    
    def add_gaussian(self, proj):
        '''Adds Gaussian noise to the image: I -> I + N(0, sigma). Std value is taken from the config.
        
        :param proj: Original image
        :type proj: :class:`np.ndarray`
        :return: Noisy image
        :rtype: :class:`np.ndarray`
        
        '''
        if self.gaussian_std == 0.:
            return proj
        
        proj += np.random.normal(0, self.gaussian_std, size=proj.shape)
        return proj
    
    def add_blur(self, proj):
        '''Adds Gaussian blur to the image. Blur width is taken from the config.
        
        :param proj: Original image
        :type proj: :class:`np.ndarray`
        :return: Noisy image
        :rtype: :class:`np.ndarray`
        
        '''
        if self.blur_width == 0.:
            return proj
        
        for i in range(proj.shape[1]):
            proj[:,i,:] = ndimage.gaussian_filter(proj[:,i,:], sigma = self.blur_width)
        return proj
    
    def add_noise(self, proj):
        '''Adds a combination of Poisson and Gaussian noise to the image.
        
        :param proj: Original image
        :type proj: :class:`np.ndarray`
        :return: Noisy image
        :rtype: :class:`np.ndarray`
        
        '''
        res = self.add_poisson(proj)
        res = self.add_gaussian(res)
        res = self.add_blur(res)
        return res
