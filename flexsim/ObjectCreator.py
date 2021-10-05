import numpy as np
import cupy
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import cupyx.scipy.ndimage

def get_volume_properties(obj_fname):
    #obj_model = np.load(obj_fname, mmap_mode="r")[80:-80,35:-35,35:-35]
    #obj_model = np.load(obj_fname, mmap_mode="r")[:,70:-70,70:-70]
    obj_model = np.load(obj_fname, mmap_mode="r")[:,:,:]
    obj_shape = obj_model.shape
    return obj_shape

class ObjectCreator(object):
    def __init__(self, size, energy_bins, matHandler):
        #size = (x, y, z)
        self.volume = np.zeros(size, dtype = int)
        self.energy_bins = energy_bins
        self.size = size
        self.voxel_size = 0.114723907 # in mm, account for this later
        self.mat = matHandler
        
    def apply_median_filter(self, object_volume):
        # Remove outliers for better thresholding
        vol_gpu = cupy.asarray(object_volume)
        vol_gpu = cupyx.scipy.ndimage.median_filter(vol_gpu, 3)
        vol_cpu = vol_gpu.get()
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        return vol_cpu
    
    def shift_volume(self, vol, shift):
        vol_gpu = cupy.asarray(vol)
        vol_gpu = cupyx.scipy.ndimage.shift(vol_gpu, shift)
        vol_cpu = vol_gpu.get()
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        return vol_cpu
    
    def zoom_volume(self, vol, zoom):
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
        #print(old_s)
        #print(new_s)
        #print(res.shape)
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return res
    
    def affine_volume(self, vol, matrix):
        vol_gpu = cupy.asarray(vol)
        mat_gpu = cupy.asarray(matrix)
        vol_gpu = cupyx.scipy.ndimage.affine_transform(vol_gpu, mat_gpu, output_shape=vol.shape)
        vol_cpu = vol_gpu.get()
        
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return vol_cpu
    
    def create_spherical_volume(self):
        print(self.volume.shape)
        h, d, w = self.volume.shape
        Z, Y, X = np.ogrid[:h, :d, :w]
        dist = np.sqrt((Z-350)**2+(Y-400)**2+(X-400)**2)
        print(Z.shape)
        print(Y.shape)
        print(X.shape)
        print(dist.shape)
        #print(dist)

        self.volume[dist <= 250] = 2
        self.volume[dist <= 220] = 0
        self.volume[dist <= 200] = 1
        
        self.main_object[self.volume == 1] = 1.0
        self.foreign_object[self.volume == 2] = 1.0
        
        return self.volume
    
    def create_cylinder(self, param):
        print(self.volume.shape)
        h, d, w = self.volume.shape
        Z, Y, X = np.ogrid[:h, :d, :w]
        X = X.astype(float)
        Y = Y.astype(float)
        Z = Z.astype(float)
        
        n = np.array([param['n_x'], param['n_y'], param['n_z']])
        a = np.array([param['a_x'], param['a_y'], param['a_z']])
        radius = param['radius']
        # direction vector should have a norm of 1
        n /= np.sqrt(np.power(n, 2).sum())
        X -= a[0]
        Y -= a[1]
        Z -= a[2]
        
        # distance formula is |(x-a) x n| / |n|
        dist = np.sqrt( ( Y*n[2] - Z*n[1] )**2 
                       +( X*n[2] - Z*n[0] )**2
                       +( X*n[1] - Y*n[0] )**2)
                
        res = np.zeros_like(dist, dtype=bool)
        res[dist <= radius] = True
        return res
        
    def create_flexray_volume(self, model_fname, param):
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        
        object_volume = np.load(model_fname)
        
        for i in range(1, self.mat.mat_count+1):
            material_mask = np.zeros(self.size, dtype=np.int16)
            
            material_mask[object_volume == i] = 1
            
            # insert volume tranform here
            #print(shift_v)
            #foreign_mask = self.shift_volume(foreign_mask, shift_v)
            #print(zoom_v)
            #foreign_mask = self.zoom_volume(foreign_mask, fozoom_v)
            
            #print(mempool.used_bytes())
            #print(mempool.total_bytes() / 1024**2)
            #print(pinned_mempool.n_free_blocks())
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            self.volume[material_mask == 1] = i
            
            
        cylinder_mask = self.create_cylinder(param)
        self.volume[np.logical_and(self.volume == 2, cylinder_mask)] = 3
        self.volume[np.logical_and(object_volume == 5, cylinder_mask)] = 3
        
        return self.volume
    
    def save_volume(self, folder):
        (folder / "GT_Recon").mkdir(exist_ok=True)
        folder = folder / "GT_Recon"
        
        h = self.volume.shape[0]
        for i in range(h):
            imageio.imsave(folder / '{:06d}.tiff'.format(i), self.volume[i,:,:].astype(np.int32))
