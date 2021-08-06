import astra
import numpy as np
import math as m
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage, misc
import flexdata
import random

from ObjectCreator import ObjectCreator, get_volume_properties
from MaterialHandler import MaterialHandler
from NoiseModel import NoiseModel

class Projector(object):
    def __init__(self, objCreator, matHandler, noiseModel, num_angles, config):
        self.obj = objCreator
        self.mat = matHandler
        self.noise = noiseModel
        self.num_angles = num_angles
        self.energy_bins = self.obj.energy_bins
        self.energy_model = config['energy_model']
        
    def read_flexray_geometry(self, path):
        self.geom = flexdata.data.read_flexraylog(path)
        
        # Calibration fixes
        self.geom.parameters['src_ort'] += (7-5.5) #see below for flexbox hard coded values
        self.geom['det_roll'] -= 0.25
        
        obj_shape = self.obj.size
        width = obj_shape[2]
        height = obj_shape[0]
        #use +1 angle since the last one is similar to the first one
        proj_shape = (height, self.num_angles + 1, width)
        self.vol_geom = self.geom.astra_volume_geom(obj_shape)
        self.foreign_vol_geom = self.geom.astra_volume_geom(obj_shape)
        self.proj_geom = self.geom.astra_projection_geom(proj_shape)
        
        self.det_y = height
        self.det_x = width
        self.detector_pixel = self.geom.parameters['det_pixel']
        self.obj.voxel_size = self.geom.parameters['img_pixel']
        #print(self.obj.voxel_size)
        #self.transform_foreign_geometry(zoom_factor = 0.7, translation_vector = [0.,0.,0.])
        
    def transform_foreign_geometry(self, zoom_factor, translation_vector):
        self.foreign_vol_geom = self.geom.astra_volume_geom(self.obj.size)
        or_voxel = self.obj.voxel_size
        tg_voxel = or_voxel * zoom_factor
        com = np.array(ndimage.center_of_mass(self.obj.foreign_object))
        for i in range(3):
            com[i] -= 0.5 * self.obj.foreign_object.shape[i]
        x_shift = com[2] * self.obj.voxel_size * (1 - zoom_factor) + translation_vector[0] * self.obj.voxel_size
        y_shift = com[1] * self.obj.voxel_size * (1 - zoom_factor) + translation_vector[1] * self.obj.voxel_size
        z_shift = com[0] * self.obj.voxel_size * (1 - zoom_factor) + translation_vector[2] * self.obj.voxel_size
        self.foreign_vol_geom['option']['WindowMinX'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMinX'] += x_shift
        self.foreign_vol_geom['option']['WindowMaxX'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMaxX'] += x_shift
        self.foreign_vol_geom['option']['WindowMinY'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMinY'] += y_shift
        self.foreign_vol_geom['option']['WindowMaxY'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMaxY'] += y_shift
        self.foreign_vol_geom['option']['WindowMinZ'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMinZ'] += z_shift
        self.foreign_vol_geom['option']['WindowMaxZ'] *= zoom_factor
        self.foreign_vol_geom['option']['WindowMaxZ'] += z_shift
    
    def project_materials(self):
        # This code only works when fo is inside mo and stays inside after transformations
        
        proj_id, proj_data = astra.create_sino3d_gpu(self.obj.main_object + self.obj.foreign_object, self.proj_geom, self.vol_geom)
        # Astra has a different definition of top left corner, so flip
        # Drop the last angle because it is similar to the first one
        full_projection = np.flip(proj_data[:,:-1,:], 0)
        
        proj_id, proj_data = astra.create_sino3d_gpu(self.obj.foreign_object, self.proj_geom, self.foreign_vol_geom)
        foreign_projection = np.flip(proj_data[:,:-1,:], 0)
        main_projection = full_projection - foreign_projection
        
        return (main_projection, foreign_projection)
    
    def mono_fp(self, voltage):
        ff = self.noise.create_flatfield_image(voltage)
        
        main_projection, foreign_projection = self.project_materials()
        main_attenuation = self.mat.get_monochromatic_intensity(1, voltage)
        print("{} -> 0.045".format(main_attenuation))
        main_attenuation = 0.045
        foreign_attenuation = self.mat.get_monochromatic_intensity(2, voltage)
        print("{} -> 0.105".format(foreign_attenuation))
        foreign_attenuation = 0.105
        proj = ff * np.exp(- main_projection*main_attenuation - foreign_projection*foreign_attenuation)
        
        return proj
    
    def poly_fp(self, voltage):
        ff = self.noise.create_flatfield_image(voltage)
        proj = np.zeros_like(ff)
        
        main_projection, foreign_projection = self.project_materials()
        main_attenuation = self.mat.get_material_curve(1)
        foreign_attenuation = self.mat.get_material_curve(2)
        spectrum_fractions = self.mat.spectrum_generate(voltage)
        
        for i in range(self.energy_bins):
            temp_spectral_projection = main_attenuation[i] * main_projection + foreign_attenuation[i] * foreign_projection
            temp_count_projection = np.exp(-temp_spectral_projection) * ff * spectrum_fractions[i]
            proj += temp_count_projection
            
        return proj
    
    def fp(self, voltage):
        if self.energy_model == "mono":
            print("Mono")
            return self.mono_fp(voltage)
        elif self.energy_model == "poly":
            print("Poly")
            return self.poly_fp(voltage)
        else:
            print("Unknown energy model, changed to 'mono'")
            return self.mono_fp(voltage)
    
    def create_projection(self, start_num, folder, voltage):
        (folder / "{}".format(voltage)).mkdir(exist_ok=True)
        folder = folder / "{}".format(voltage)
        (folder / "Proj").mkdir(exist_ok=True)
        (folder / "Log").mkdir(exist_ok=True)
        
        ff = self.noise.create_flatfield_image(voltage)
        proj = self.fp(voltage)
        proj = self.noise.add_noise(proj)
        log = -np.log(np.divide(proj, ff))
        
        for i in range(self.num_angles):
            imageio.imsave(folder / 'Proj' / '{}.tiff'.format(start_num+i), proj[:,i,:].astype(np.float32))
            imageio.imsave(folder / 'Log' / '{}.tiff'.format(start_num+i), log[:,i,:].astype(np.float32))
            
    def create_gt(self, start_num, folder):
        (folder / "GT").mkdir(exist_ok=True)
        folder = folder / "GT"
        
        main_projection, foreign_projection = self.project_materials()
        gt = np.zeros_like(foreign_projection, dtype=int)
        gt[foreign_projection > 0.] = 1
        
        for i in range(self.num_angles):
            imageio.imsave(folder / '{}.tiff'.format(start_num+i), gt[:,i,:].astype(np.int32))
        
def default_process(obj_folder, out_folder, config):
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins)
    obj = ObjectCreator(obj_shape, energy_bins)
    vol = obj.create_flexray_volume(model_fname, [0.02, 0.07])
    
    num_angles = 20
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, config)
    proj.read_flexray_geometry(obj_folder)
    
    #proj.create_projection(0, our_folder / "50", 50)
    #proj.create_projection(0, our_folder, 80)
    #proj.create_gt(0, our_folder)
    
    sample_size = 20
    zooms = np.random.uniform(0.4, 1.1, sample_size)
    shift_x = np.random.uniform(-150., 0., sample_size)
    shift_y = np.random.uniform(-10., 10., sample_size)
    shift_z = np.random.uniform(-50., 20., sample_size)
    print(zooms)
    print(shift_x)
    print(shift_y)
    print(shift_z)
    
    for i in range(sample_size):
        proj.transform_foreign_geometry(zooms[i], [shift_x[i], shift_y[i], shift_z[i]])
        proj.create_projection(i*num_angles, our_folder, 80)
        proj.create_gt(i*num_angles, our_folder)

if __name__ == "__main__":
    obj_folder = Path("/export/scratch2/vladysla/Data/Real/20.07.2021_playdoh/fo1_scan/")
    our_folder = Path("../Data")
    config = {
        'energy_model' : 'poly'
        }
    
    default_process(obj_folder, our_folder, config)
