import numpy as np
import cupy
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import cupyx.scipy.ndimage
import scipy.ndimage
import skimage.measure
import skimage.morphology
from voltools import transform
from tqdm import tqdm
import time

from flexsim import utils

class ObjectCreator(object):
    '''Class providing object's model. Data augmentation through volume transformation is done here.
    
    :param size: Volume shape with 3 dimensions: (height, width, width)
    :type size: :class:`np.ndarray`
    :param matHandler: Instance of Material Handler providing information about object's materials.
    :type matHandler: :class:`MaterialHandler`
    
    '''
    def __init__(self, size, matHandler):
        '''Constructor method.
        '''
        self.volume = np.zeros(size, dtype = int)
        self.size = size
        #self.voxel_size = 0.114723907 # in mm, account for this later
        self.mat = matHandler
        
    def get_volume(self):
        return self.volume
    
    def set_volume(self, volume):
        self.volume = volume
        
    def set_flexray_volume(self, obj_folder):
        '''Initializes object volume by reading it from the folder
        
        :param obj_folder: Path to the folder containing slices of the segmentation.
        :type obj_folder: :class:`pathlib.Path`
        '''
        self.volume = utils.read_volume(obj_folder)
        
    def modify_volume(self, func, kwargs):
        '''General method to modify volume using the provided function
        
        :param func: Function to modify object volume
        :type func: :class:`function`
        :param kwargs: Dictionary of arguments to pass into the function
        :type func: :class:`dict`
        '''
        self.volume = func(self.volume, **kwargs)
        
    def get_stats(self):
        '''Save the total count of voxels of every material. The numbers are the same for all projections of the same volume
        '''
        mat_counts = []
        for i in range(1, self.mat.mat_count+1):
            mat_counts.append(np.count_nonzero(self.volume == i))
        mat_counts = np.array(mat_counts)
        
        return mat_counts
    
    def save_volume(self, folder):
        (folder / "Volume").mkdir(exist_ok=True)
        folder = folder / "Volume"
        
        h = self.volume.shape[0]
        for i in range(h):
            imageio.imsave(folder / '{:06d}.tiff'.format(i), self.volume[i,:,:].astype(np.int32))
