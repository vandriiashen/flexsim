import numpy as np
import cupy
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import cupyx.scipy.ndimage
import scipy.ndimage
from tqdm import tqdm

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
        self.voxel_size = 0.114723907 # in mm, account for this later
        self.mat = matHandler
    
    def shift_volume(self, vol, shift):
        ''' Shifts the object's volume using GPU acceleration (`cupyx.scipy.ndimage.shift`).
        
        :param vol: Array containing the object's model
        :type vol: :class:`np.ndarray`
        :param shift: Shift along the axes.
        :type shift: :class:`float` or :class:`list`
        :return: Shifted volume
        :rtype: :class:`np.ndarray`
        
        '''
        vol_gpu = cupy.asarray(vol)
        vol_gpu = cupyx.scipy.ndimage.shift(vol_gpu, shift)
        vol_cpu = vol_gpu.get()
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        return vol_cpu
    
    def zoom_volume(self, vol, zoom):
        ''' Zooms the object's volume using GPU acceleration (`cupyx.scipy.ndimage.zoom`). The same shape is maintained in the output array.
        
        :param vol: Array containing the object's model
        :type vol: :class:`np.ndarray`
        :param zoom: Zoom along the axes.
        :type zoom: :class:`float` or :class:`list`
        :return: Zoomed volume
        :rtype: :class:`np.ndarray`
        
        '''
        old_s = vol.shape
        vol_gpu = cupy.asarray(vol)
        vol_gpu = cupyx.scipy.ndimage.zoom(vol_gpu, zoom)
        vol_cpu = vol_gpu.get()
        new_s = vol_cpu.shape
        
        pad = [[0, 0], [0, 0], [0, 0]]
        select = [[0, new_s[0]], [0, new_s[1]], [0, new_s[2]]]
        for i in range(3):
            if new_s[i] < old_s[i]:
                pad[i] = [(old_s[i]-new_s[i]) // 2, old_s[i]-new_s[i] - (old_s[i]-new_s[i]) // 2]
            if new_s[i] > old_s[i]:
                select[i] = [(new_s[i]-old_s[i]) // 2, old_s[i] + (new_s[i]-old_s[i]) // 2]
                
        res = vol_cpu[select[0][0]:select[0][1] , select[1][0]:select[1][1] , select[2][0]:select[2][1]]
        res = np.pad(res, pad)

        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return res
    
    def affine_volume(self, vol, matrix):
        '''Performs affine transformation of the object's volume using GPU acceleration (`cupyx.scipy.ndimage.affine_transform`).
        
        :param vol: Array containing the object's model
        :type vol: :class:`np.ndarray`
        :param matrix: Matrix of the affine transformation
        :type matrix: :class:`np.ndarray`
        :return: Transformed volume
        :rtype: :class:`np.ndarray`
        
        '''
        vol_gpu = cupy.asarray(vol)
        mat_gpu = cupy.asarray(matrix)
        vol_gpu = cupyx.scipy.ndimage.affine_transform(vol_gpu, mat_gpu, output_shape=vol.shape)
        vol_cpu = vol_gpu.get()
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return vol_cpu
    
    def create_spherical_pocket(self, mat_num, centre, radius):
        Z, Y, X = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        Z -= centre[0]
        Y -= centre[1]
        X -= centre[2]
        dist = np.sqrt(X**2 + Y**2 + Z**2)
        
        mask = dist < radius
        mask = np.logical_and(mask, self.volume == 2)
        print("Sphere done")
        self.volume[mask] = mat_num
        
    def create_plane_pocket(self, mat_num, z_lim, start_point, step_size, direction_vector):
        mask = np.zeros_like(self.volume, dtype=bool)
        direction_vector /= np.power(direction_vector, 2).sum()
        
        cur_point = start_point
        for i in range(40 * 10 // step_size):
            rand_step = np.random.normal(0.,2.0)
            rand_step -= 2*np.abs(i-20) // 5
            rand_step = int(step_size + rand_step)
            mask[z_lim[0]:z_lim[1],
                 cur_point[1]-rand_step:cur_point[1]+rand_step,
                 cur_point[0]-rand_step:cur_point[0]+rand_step] = True
            
            rand_incr = np.random.normal(0.,3.0, size=(2))
            point_incr = (step_size * direction_vector + rand_incr).astype(int)
            cur_point += point_incr
        
        mask = np.logical_and(mask, self.volume == 2)
        print("Plane done")
        self.volume[mask] = mat_num
        
    
    def replace_material(self, src_num, dest_num):
        '''Changes material in voxels from source to dest.
        
        :param src_num: ID of material that should be removed
        :type src_num: :class:`i`
        :param dest_num: ID of material that should be used instead
        :type dest_num: :class:`i`
        '''
        self.volume[self.volume == src_num] = dest_num
        
    def set_flexray_volume(self, obj_folder):
        '''Initializes object volume by reading it from the folder
        
        :param obj_folder: Path to the folder containing slices of the segmentation.
        :type obj_folder: :class:`pathlib.Path`
        '''
        self.volume = utils.read_volume(obj_folder)
    
    def save_volume(self, folder):
        (folder / "GT_Recon").mkdir(exist_ok=True)
        folder = folder / "GT_Recon"
        
        h = self.volume.shape[0]
        for i in range(h):
            imageio.imsave(folder / '{:06d}.tiff'.format(i), self.volume[i,:,:].astype(np.int32))
