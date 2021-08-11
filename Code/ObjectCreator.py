import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter

def get_volume_properties(obj_fname):
    obj_model = np.load(obj_fname, mmap_mode="r")
    obj_shape = obj_model.shape
    return obj_shape

class ObjectCreator(object):
    def __init__(self, size, energy_bins):
        #size = (x, y, z)
        self.volume = np.zeros(size, dtype = int)
        self.main_object = np.zeros(size, dtype = float)
        self.foreign_object = np.zeros(size, dtype = float)
        self.energy_bins = energy_bins
        self.size = size
        self.voxel_size = 0.114723907 # in mm, account for this later
        
    def create_flexray_volume(self, model_fname, thresholds):
        material_model = np.zeros(self.size, dtype=int)
        object_volume = np.load(model_fname)
        #print(object_volume.shape)
        
        main_thr = thresholds[0]
        material_model[object_volume > main_thr] = 1
        
        foreign_thr = thresholds[1]
        print(object_volume.shape[0])
        
        for i in range(object_volume.shape[0]):
            filter_slice = median_filter(object_volume[i,:,:], 3)
            discr_slice = filter_slice > foreign_thr
            material_model[i,discr_slice] = 2
        
        self.volume = material_model
        self.main_object[material_model == 1] = 1.0
        self.foreign_object[material_model == 2] = 1.0
        
        return self.volume
    
if __name__ == "__main__":
    obj_folder = Path("/export/scratch2/vladysla/Data/Real/20.07.2021_playdoh/fo1_scan/")
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    obj_creator = ObjectCreator(obj_shape, energy_bins)
    vol = obj_creator.create_flexray_volume(model_fname, [0.02, 0.06])
    
    plt.imshow(obj_creator.volume[190,:,:])
    plt.show()
