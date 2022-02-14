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
            
    def check_material_inside(self, inner_mat, outer_mat):
        '''Checks if material is inside of another material
        '''
        
        inner_vol = self.volume == inner_mat
        outer_vol = np.copy(self.volume == outer_mat)
        shape_before_downscale = outer_vol.shape
        outer_downsampled = skimage.transform.downscale_local_mean(outer_vol, (4,4,4))
        outer_downsampled_convex = skimage.morphology.convex_hull.convex_hull_image(outer_downsampled)
        outer_convex = skimage.transform.resize(outer_downsampled_convex, shape_before_downscale)
        outer_convex = (outer_convex>0).astype(bool)
        
        outside_voxels = np.logical_and(inner_vol, ~outer_convex)
        if np.count_nonzero(outside_voxels) > 0:
            print(np.count_nonzero(outside_voxels))
            print("Outside")
            return False
        
        return True
    
    def create_sphere(self, replace_num, fill_num, centre, radius):
        Z, Y, X = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        Z -= centre[0]
        Y -= centre[1]
        X -= centre[2]
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        mask = R < radius
        mask = np.logical_and(mask, self.volume == replace_num)
        self.volume[mask] = fill_num
        
    def create_spherical_fragments(self, replace_num, fill_num, centre, radius, thickness, arc_start, num_fragments):
        Z, Y, X = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
            
        Z -= int(centre[0])
        Y -= int(centre[1])
        X -= int(centre[2])
        R = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.zeros_like(R)
        theta = np.arccos(Z / R, where=R!=0)
        phi = np.arctan2(Y, X)
                
        mask = self.volume==replace_num
        fragment_mask = np.zeros_like(mask)
        
        for i in range(num_fragments):
            phi_start = np.random.uniform(phi.min(), phi.max())
            phi_step = 0.1
            phi_mask = np.logical_and(phi > phi_start, phi < phi_start + phi_step)
            
            #theta_end = np.random.uniform(theta.min(), arc_start)
            #theta_start = np.random.uniform(theta.min(), theta_end)
            
            theta_end = np.random.uniform(arc_start, theta.max())
            theta_start = np.random.uniform(arc_start, theta_end)
            
            theta_mask = np.logical_and(theta > theta_start, theta < theta_end)
            
            R_step = np.random.uniform(0.5*thickness, 1.5*thickness)
            R_mask = np.logical_and(R > radius, R < radius + R_step)
            
            tmp_mask = np.logical_and(phi_mask, theta_mask)
            tmp_mask = np.logical_and(tmp_mask, R_mask)
            fragment_mask = np.logical_or(fragment_mask, tmp_mask)
            
        mask = np.logical_and(mask, fragment_mask)        
        self.volume[mask] = fill_num
            
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
        
    def save_stats(self, folder, start_num, proj_num):
        '''Save the total count of voxels of every material. The numbers are the same for all projections of the same volume
        '''
        mat_counts = []
        for i in range(1, self.mat.mat_count+1):
            mat_counts.append(np.count_nonzero(self.volume == i))
        
        stat_line = ",".join(str(num) for num in mat_counts)
        
        with open(folder / "stats.csv", "a") as f:
            for i in range(start_num, start_num+proj_num):
                f.write("{},{}\n".format(i, stat_line))
    
    def save_volume(self, folder):
        (folder / "GT_Recon").mkdir(exist_ok=True)
        folder = folder / "GT_Recon"
        
        h = self.volume.shape[0]
        for i in range(h):
            imageio.imsave(folder / '{:06d}.tiff'.format(i), self.volume[i,:,:].astype(np.int32))
