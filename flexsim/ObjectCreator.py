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
        self.voxel_size = 0.114723907 # in mm, account for this later
        self.mat = matHandler
    
    def affine_volume(self, scale, shear, rotation, translation, verbose=False):
        '''Performs affine transformation of the object's volume
                
        '''
        if verbose == True:
            start_time = time.time()
            
        res_vol = np.zeros_like(self.volume, dtype=int)

        for i in range(1, self.mat.mat_count+1):
            tmp_vol = np.zeros_like(self.volume, dtype=bool)
            tmp_vol[self.volume == i] = True
            transformed_vol = transform(tmp_vol, interpolation='linear', device='cpu', scale=scale, shear=shear, translation=translation, rotation=rotation, rotation_units='deg', rotation_order='rzxz')
            transformed_vol = transformed_vol.astype(bool)
            transformed_vol = skimage.morphology.binary_dilation(transformed_vol)
            res_vol[transformed_vol] = i
            
        self.volume = res_vol
                            
        if verbose == True:
            end_time = time.time() - start_time
            print("Affine transform time = {:.2f}s".format(end_time))
    
    def affine_material(self, replace_num, fill_num, scale, shear, rotation, translation, verbose=False):
        '''Performs affine transformation of the material
        
        '''
        if verbose == True:
            start_time = time.time()
        
        tmp_vol = np.zeros_like(self.volume, dtype=bool)
        
        tmp_vol[self.volume == replace_num] = True
        self.volume[self.volume == replace_num] = fill_num
        
        transformed_vol = transform(tmp_vol, interpolation='linear', device='cpu', scale=scale, shear=shear, translation=translation, rotation=rotation, rotation_units='deg', rotation_order='rzxz')
        transformed_vol = transformed_vol.astype(bool)
        transformed_vol = skimage.morphology.binary_dilation(transformed_vol)
        
        self.volume[transformed_vol] = replace_num
                
        if verbose == True:
            end_time = time.time() - start_time
            print("Affine transform time = {:.2f}s".format(end_time))
    
    def create_spherical_pocket(self, mat_num, centre, radius, dilation_num):
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
        :type src_num: :class:`int`
        :param dest_num: ID of material that should be used instead
        :type dest_num: :class:`int`
        '''
        self.volume[self.volume == src_num] = dest_num
        
    def remove_material_clusters(self, src_num, dest_num, num_keep):
        '''Compute clusters of voxels filled with a certain material.
        '''
        labels, nfeatures = scipy.ndimage.label(self.volume == src_num)
        props = skimage.measure.regionprops(labels)
        self.replace_material(src_num, dest_num)
        
        propL = []
        for prop in props:
            propL.append ((prop.area, prop))
        propL = sorted (propL, key=lambda r:r[0], reverse=True)
        
        for i in range(min(num_keep, len(propL))):
            area, prop = propL[i]
            self.volume[labels == prop.label] = src_num
            
    def remove_points_random(self, src_num, dest_num, remove_fraction, verbose=True):
        '''...
        '''
        points = np.nonzero(self.volume == src_num)
        points_num = points[0].shape[0]
        coords = np.zeros((points_num, 3), dtype=np.int32)
        for i in range(3):
            coords[:,i] = points[i]
        
        seq = np.arange(points_num)
        np.random.shuffle(seq)
        
        for i in range(int(remove_fraction*points_num)):
            point_coords = coords[seq[i],:]
            self.volume[point_coords[0], point_coords[1], point_coords[2]] = dest_num
            
    def split_clusters(self, src_num, dest_num, num_classes, num_drop_classes, verbose=True):
        '''
        '''
        points = np.nonzero(self.volume == src_num)
        points_num = points[0].shape[0]
        coords = np.zeros((points_num, 3), dtype=np.int32)
        for i in range(3):
            coords[:,i] = points[i]
        
        seq = np.arange(points_num)
        np.random.shuffle(seq)
        
        dist_map = np.zeros((points_num, num_classes), dtype=np.float32)
        for i in range(num_classes):
            dist_map[:,i] = np.power(np.subtract(coords, coords[seq[i],:]), 2).sum(axis=1)
        class_map = np.argmin(dist_map, axis=1)
        if verbose:
            print(dist_map[:10,:])
            print(class_map[:10])
            for i in range(num_classes):
                print(np.count_nonzero(class_map == i))
                    
        for i in range(num_drop_classes):
            select = coords[class_map==i,:]
            self.volume[select[:,0], select[:,1], select[:,2]] = dest_num
        
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
