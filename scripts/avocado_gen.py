import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import shutil

import flexsim

def generate_affine_matrix():
    mat = np.zeros((3,3))
    main_el = np.random.uniform(0.95, 1.05, size=(3,))
    off_el = np.random.uniform(-0.05, 0.05, size=(3,))
    mat[0,0] = main_el[0]
    mat[1,1] = main_el[1]
    mat[2,2] = main_el[2]
    mat[0,1] = off_el[0]
    mat[0,2] = off_el[1]
    mat[1,2] = off_el[2]
    mat[1,0] = off_el[3]
    mat[2,0] = off_el[4]
    mat[2,1] = off_el[5]
    print(mat)
    return mat
        
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
    
    for j in tqdm(range(2,4)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)
        # Replace existing air gaps by avocado meat
        obj.replace_material(4, 2)
        
        c_z = np.random.uniform(450., 490.)
        c_y = np.random.uniform(400., 480.)
        c_x = np.random.uniform(400., 480.)
        sphere_centre = np.array([int(c_z), int(c_y), int(c_x)])
        if j < 3:
            sphere_radius = int(np.random.uniform(40., 80.))
        else:
            sphere_radius = int(np.random.uniform(80., 120.))
        if j > 1:
            obj.create_spherical_pocket(4, sphere_centre, sphere_radius)
        
        small_spheres_num = int(np.random.uniform(4,6))
        for i in range(small_spheres_num):
            c_z = np.random.uniform(60., 250.)
            c_y = np.random.uniform(380., 600.)
            c_x = np.random.uniform(380., 600.)
            sph_centre = np.array([int(c_z), int(c_y), int(c_x)])
            sph_radius = int(np.random.uniform(5., 20.))
            #print(sph_centre)
            #print(sph_radius)
            obj.create_spherical_pocket(4, sph_centre, sph_radius)
                                
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0, 360))
            
        proj.create_projection(j*num_angles, out_folder, 90)
        proj.create_gt(j*num_angles, out_folder)
        
def avocado_reduce_air(config_fname):
    '''Remove air pockets from volume
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
    
    aug_samples = config['Simulation']['augmentation_samples']
    keep_clusters_num = np.random.randint(2, 6, size=(aug_samples,))
    total_regions_num = np.random.randint(25, 30, size=(aug_samples,))
    drop_regions_num = np.random.randint(20, 25, size=(aug_samples,))
    print(keep_clusters_num)
    print(total_regions_num)
    print(drop_regions_num)
    
    for i in tqdm(range(aug_samples)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)
        
        scale = np.random.uniform(0.7, 1.3, size=(3,))
        shear = np.random.uniform(-0.3, 0.3, size=(3,))
        rotation = (0., 0., 0.)
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, True)
        
        obj.remove_material_clusters(4, 2, keep_clusters_num[i])
        obj.split_clusters(4, 2, total_regions_num[i], drop_regions_num[i], False)
        obj.save_volume(out_folder)
            
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0, 360))
            
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
            
if __name__ == "__main__":
    config_fname = "avocado.ini"
    np.random.seed(seed = 15)
    
    #avocado_gt_gen(config_fname)
    #avocado_augment(config_fname)
    avocado_reduce_air(config_fname)
