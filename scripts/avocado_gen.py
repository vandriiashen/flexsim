import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import shutil

import flexsim
        
def avocado_gt_gen(config_fname):
    '''Generates projections without augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = Path(config['Paths']['obj_folder'])
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder'])
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
        
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360))
        
    proj.create_projection(0, out_folder, 90)
    proj.create_gt(0, out_folder)
        
    gt_folder = Path(out_folder) / "GT"
    shutil.copytree(gt_folder, Path(obj_folder) / "gt")
    
def avocado_augment(config_fname):
    '''Generates projections with augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = Path(config['Paths']['obj_folder'])
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder'])
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    for j in tqdm(range(0,1)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)
        # Replace existing air gaps by avocado meat
        obj.replace_material(4, 2)
        
        c_z = np.random.uniform(450., 490.)
        c_y = np.random.uniform(400., 480.)
        c_x = np.random.uniform(400., 480.)
        sphere_centre = np.array([int(c_z), int(c_y), int(c_x)])
        if j < 10:
            sphere_radius = int(np.random.uniform(40., 100.))
        else:
            sphere_radius = int(np.random.uniform(100., 140.))
        if j > 1:
            obj.create_spherical_pocket(4, sphere_centre, sphere_radius)
        
        z1 = np.random.uniform(320., 380.)
        z2 = np.random.uniform(520., 580.)
        z_lim = np.array([int(z1), int(z2)])
        s1 = np.random.uniform(420., 480.)
        s2 = np.random.uniform(270., 320.)
        start_point = np.array([int(s1), int(s2)])
        step_size = int(np.random.uniform(10., 20.))
        d1 = np.random.uniform(0., 0.2)
        d2 = np.random.uniform(0.8, 1.0)
        direction_vector = np.array([d1, d2])
        if j > 2:
            obj.create_plane_pocket(4, z_lim, start_point, step_size, direction_vector)
        
        z_lim = np.array([int(z1), int(z2)])
        s1 = np.random.uniform(270., 320.)
        s2 = np.random.uniform(420., 480.)
        start_point = np.array([int(s1), int(s2)])
        step_size = int(np.random.uniform(10., 20.))
        d1 = np.random.uniform(0.8, 1.)
        d2 = np.random.uniform(0., 0.2)
        direction_vector = np.array([d1, d2], dtype=float)
        if j > 2:
            obj.create_plane_pocket(4, z_lim, start_point, step_size, direction_vector)
        
        small_spheres_num = int(np.random.uniform(4,6))
        for i in range(small_spheres_num):
            c_z = np.random.uniform(60., 250.)
            c_y = np.random.uniform(380., 600.)
            c_x = np.random.uniform(380., 600.)
            sph_centre = np.array([int(c_z), int(c_y), int(c_x)])
            sph_radius = int(np.random.uniform(10., 30.))
            #print(sph_centre)
            #print(sph_radius)
            obj.create_spherical_pocket(4, sph_centre, sph_radius)
                                
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0, 360))
            
        proj.create_projection(j*num_angles, out_folder, 90)
        proj.create_gt(j*num_angles, out_folder)

if __name__ == "__main__":
    config_fname = "avocado.ini"
    np.random.seed(seed = 42)
    
    #avocado_gt_gen(config_fname)
    avocado_augment(config_fname)
