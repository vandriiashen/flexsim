import astra
import numpy as np
import math as m
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from scipy import ndimage
import flexdata

class Projector(object):
    '''Class performing forward projection and generation of data
    
    :param obj: instance of Object Creator providing object's model
    :type obj: :class:`ObjectCreator`
    :param mat: instance of Material Handler providing absorption properties of project_materials
    :type mat: :class:`MaterialHandler`
    :param noise: instance of Noise Model providing noise parameters
    :type noise: :class:`NoiseModel`
    :param config: Dictionary of config parameters: number of angles, energy model, number of energy bins, noise flags
    :type config: :class:`dict`
    
    '''
    def __init__(self, objCreator, matHandler, noiseModel, config):
        '''Constructor method
        '''
        self.obj = objCreator
        self.mat = matHandler
        self.noise = noiseModel
        self.num_angles = config['num_angles']
        self.energy_bins = config['energy_bins']
        self.energy_model = config['energy_model']
        self.noise_flag = config['noise']
        self.save_noiseless_flag = config['save_noiseless']

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
        #print("Detector pixel - {}".format(self.detector_pixel))
        #print("Object voxel - {}".format(self.obj.voxel_size))
    
    def project_material(self, mat_num):
        material_volume = np.zeros(self.obj.size, dtype = float)
        material_volume[self.obj.volume == mat_num] = 1.0
        
        mat_id, mat_data = astra.create_sino3d_gpu(material_volume, self.proj_geom, self.vol_geom)
        # Astra has a different definition of top left corner, so flip
        # Drop the last angle because it is similar to the first one
        mat_projection = np.flip(mat_data[:,:-1,:], 0)
                
        return mat_projection
    
    def mono_fp(self):
        ff = self.noise.create_flatfield_image()
        att_proj = np.zeros_like(ff, dtype = float)
        for i in range(1, self.mat.mat_count+1):
            mat_attenuation = self.mat.get_monochromatic_intensity(i)
            mat_proj = self.project_material(i)
            att_proj += mat_attenuation*mat_proj
            
        proj = ff * np.exp(-att_proj)
        
        return proj
    
    def poly_fp(self, voltage):
        ff = self.noise.create_flatfield_image()
        proj = np.zeros_like(ff)
        
        for i in range(self.energy_bins):
            temp_spectral_projection = main_attenuation[i] * main_projection + foreign_attenuation[i] * foreign_projection
            temp_count_projection = np.exp(-temp_spectral_projection) * ff * spectrum_fractions[i]
            proj += temp_count_projection
            
        return proj
    
    def fp(self, voltage):
        if self.energy_model == "mono":
            return self.mono_fp()
        elif self.energy_model == "poly":
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
        ff = self.noise.create_flatfield_image()
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
    
    def create_material_map(self, start_num, folder):
        ''' Creates multi-channel material map of the projection (every channel corresponds to the amount of the respective material along the trajectory).
        
        :param start_num: Starting index for writing images
        :type start_num: :class:`int`
        :param folder: Root folder for the data. Maps will be written in GT/ subfolder.
        :type folder: :class:`pathlib.Path`
        
        '''
        (folder / "GT").mkdir(exist_ok=True)
        folder = folder / "GT"
        
        mat_num = self.mat.mat_count
        depth_map = np.zeros((mat_num, self.obj.size[0], self.num_angles, self.obj.size[2]))
        for i in range(0, mat_num):
            mat_proj = self.project_material(i+1)
            depth_map[i,:] = mat_proj
        
        for i in range(self.num_angles):
            imageio.mimwrite(folder / '{:06d}.tiff'.format(start_num+i), depth_map[:,:,i,:].astype(np.float32))
            
    def create_gt(self, start_num, folder):
        (folder / "GT").mkdir(exist_ok=True)
        folder = folder / "GT"
        
        gt = np.zeros((self.obj.size[0], self.num_angles, self.obj.size[2]))
        for i in range(1, self.mat.mat_count+1):
            mat_proj = self.project_material(i)
            gt[mat_proj > 0.] = i
        
        for i in range(self.num_angles):
            imageio.imsave(folder / '{:06d}.tiff'.format(start_num+i), gt[:,i,:].astype(np.uint8))
            
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
