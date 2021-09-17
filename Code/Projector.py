import astra
import numpy as np
import math as m
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from scipy import ndimage
import flexdata

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
        self.noise_flag = config['noise']
        self.save_noiseless_flag = config['save_noiseless']
        # Test this code
        if self.noise_flag == False:
            self.save_noiseless_flag = True
        
    def read_flexray_geometry(self, path, ang_range):
        self.geom = flexdata.data.read_flexraylog(path)
        
        # Calibration fixes
        '''
        self.geom.parameters['src_ort'] += (7-5.5) #see below for flexbox hard coded values
        self.geom['det_roll'] -= 0.25
        self.geom.parameters['ang_range'] = ang_range
        '''
        
        obj_shape = self.obj.size
        width = obj_shape[2]
        height = obj_shape[0]
        #use +1 angle since the last one is similar to the first one
        proj_shape = (height, self.num_angles + 1, width)
        self.vol_geom = self.geom.astra_volume_geom(obj_shape)
        self.proj_geom = self.geom.astra_projection_geom(proj_shape)
        
        self.det_y = height
        self.det_x = width
        self.detector_pixel = self.geom.parameters['det_pixel']
        self.obj.voxel_size = self.geom.parameters['img_pixel']
        
    def transform_main_geometry(self, zoom_vector):
        self.vol_geom = self.geom.astra_volume_geom(self.obj.size)
        
        self.vol_geom['option']['WindowMinX'] *= zoom_vector[0]
        self.vol_geom['option']['WindowMaxX'] *= zoom_vector[0]
        self.vol_geom['option']['WindowMinY'] *= zoom_vector[1]
        self.vol_geom['option']['WindowMaxY'] *= zoom_vector[1]
        self.vol_geom['option']['WindowMinZ'] *= zoom_vector[2]
        self.vol_geom['option']['WindowMaxZ'] *= zoom_vector[2]
        
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
    
    def project_material(self, mat_num):
        material_volume = np.zeros(self.obj.size, dtype = float)
        material_volume[self.obj.volume == mat_num] = 1.0
        
        mat_id, mat_data = astra.create_sino3d_gpu(material_volume, self.proj_geom, self.vol_geom)
        # Astra has a different definition of top left corner, so flip
        # Drop the last angle because it is similar to the first one
        mat_projection = np.flip(mat_data[:,:-1,:], 0)
                
        return mat_projection
    
    def mono_fp(self, voltage):
        ff = self.noise.create_flatfield_image(voltage)
        
        att_proj = np.zeros_like(ff, dtype = float)
        for i in range(1, self.mat.mat_count+1):
            mat_attenuation = self.mat.get_monochromatic_intensity(i, voltage)
            mat_proj = self.project_material(i)
            att_proj += mat_attenuation*mat_proj
            
        proj = ff * np.exp(-att_proj)
        
        return proj
    
    def poly_fp(self, voltage):
        ff = self.noise.create_flatfield_image(voltage)
        proj = np.zeros_like(ff)
        
        """
        main_projection, foreign_projection = self.project_materials()
        main_attenuation = self.mat.get_material_curve(1)
        foreign_attenuation = self.mat.get_material_curve(2)
        spectrum_fractions = self.mat.spectrum_generate(voltage)
        """
        
        for i in range(self.energy_bins):
            temp_spectral_projection = main_attenuation[i] * main_projection + foreign_attenuation[i] * foreign_projection
            temp_count_projection = np.exp(-temp_spectral_projection) * ff * spectrum_fractions[i]
            proj += temp_count_projection
            
        return proj
    
    def fp(self, voltage):
        if self.energy_model == "mono":
            #print("Mono")
            return self.mono_fp(voltage)
        elif self.energy_model == "poly":
            #print("Poly")
            return self.poly_fp(voltage)
        else:
            print("Unknown energy model, changed to 'mono'")
            return self.mono_fp(voltage)
    
    def create_projection(self, start_num, folder, voltage):
        (folder / "{}".format(voltage)).mkdir(exist_ok=True)
        folder = folder / "{}".format(voltage)
        (folder / "Proj").mkdir(exist_ok=True)
        (folder / "Log").mkdir(exist_ok=True)
        if self.save_noiseless_flag:
            (folder / "Noiseless").mkdir(exist_ok=True)
        
        ff = self.noise.create_flatfield_image(voltage)
        proj = self.fp(voltage)
        
        if self.save_noiseless_flag:
            log_noiseless = -np.log(np.divide(proj, ff))
            for i in range(self.num_angles):
                imageio.imsave(folder / 'Noiseless' / '{:06d}.tiff'.format(start_num+i), log_noiseless[:,i,:].astype(np.float32))
        
        if self.noise_flag:
            proj = self.noise.add_noise(proj)
        log = -np.log(np.divide(proj, ff))
        
        for i in range(self.num_angles):
            imageio.imsave(folder / 'Proj' / '{:06d}.tiff'.format(start_num+i), proj[:,i,:].astype(np.float32))
            imageio.imsave(folder / 'Log' / '{:06d}.tiff'.format(start_num+i), log[:,i,:].astype(np.float32))
            
        astra.data3d.clear()
        return log
            
    def create_gt(self, start_num, folder):
        (folder / "GT").mkdir(exist_ok=True)
        folder = folder / "GT"
        
        gt = np.zeros((self.obj.size[0], self.num_angles, self.obj.size[2]))
        for i in range(1, self.mat.mat_count+1):
            mat_proj = self.project_material(i)
            gt[mat_proj > 0.] = i
        
        for i in range(self.num_angles):
            imageio.imsave(folder / '{:06d}.tiff'.format(start_num+i), gt[:,i,:].astype(np.int32))
            
    def create_reconstruction(self, log, start_num, folder, voltage):
        folder = folder / "{}".format(voltage)
        (folder / "Recon").mkdir(exist_ok=True)
        
        proj_shape = log.shape
        upd_log = np.zeros((proj_shape[0], proj_shape[1]+1, proj_shape[2]))
        upd_log[:,:-1,:] = log
        upd_log[:,-1,:] = log[:,0,:]
        log = np.flip(upd_log, 0)
        
        proj_id = astra.data3d.create('-sino', self.proj_geom, log)
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        rec = astra.data3d.get(rec_id)
        for i in range(rec.shape[0]):
            imageio.imsave(folder / 'Recon' / '{:06d}.tiff'.format(start_num+i), rec[i,:,:].astype(np.float32))
        
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
        
def default_process_fod(obj_folder, out_folder, config):
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins)
    obj = ObjectCreator(obj_shape, energy_bins)
    vol = obj.create_flexray_volume(model_fname, [0.025, 0.07])
    
    num_angles = 450
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, config)
    
    for i in range(4):
        proj.read_flexray_geometry(obj_folder, (90*i, 90*i+90))
        print(proj.geom.parameters)
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)

if __name__ == "__main__":
    obj_folder = Path("../../../Data/Real/AutomatedFOD/Object42_Scan20W/")
    out_folder = Path("../Data")
    config = {
        'energy_model' : 'poly',
        'noise' : True,
        'save_noiseless' : True
        }
    
    np.random.seed(seed = 3)
    
    #default_process_augment(obj_folder, out_folder, config)
    default_process_fod(obj_folder, out_folder, config)
    convert_folder = Path("../../../Data/Simulated/AutomatedFOD")
    convert(out_folder, convert_folder, "obj42")
