import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import shutil

import flexsim
        
def playdoh_gt_gen_full_angles(config_fname):
    '''Generates projections without augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = Path(config['Paths']['obj_folder'])
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder'])
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    sample_size = config['Simulation']['augmentation_samples']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
    
    for i in range(4):
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (90*i, 90*(i+1)))
        
        proj.create_projection(450*i, out_folder, 90)
        proj.create_gt(450*i, out_folder)
    
    gt_folder = Path(out_folder) / "GT"
    shutil.copytree(gt_folder, Path(obj_folder) / "gt")
    
def playdoh_gt_gen(config_fname):
    '''Generates projections without augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = Path(config['Paths']['obj_folder'])
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder'])
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    sample_size = config['Simulation']['augmentation_samples']
    
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
    
def playdoh_augment(config_fname):
    '''Generates new samples based on a single playdoh scan. In every sample, foreign object volume is transformed randomly.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = Path(config['Paths']['obj_folder'])
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder'])
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    sample_size = config['Simulation']['augmentation_samples']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    aug_samples = config['Simulation']['augmentation_samples']
    
    with open(out_folder / "stats.csv", "w") as f:
        f.write("Proj_num,Playdoh,Stone\n")
    
    for i in tqdm(range(aug_samples)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)
        
        scale = np.random.uniform(0.8, 1.2, size=(3,))
        shear = np.random.uniform(-0.1, 0.1, size=(3,))
        rotation = np.random.uniform(0., 20., size=(3,))
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
        
        if i < aug_samples // 2:
            scale = np.random.uniform(0.6, 1.8, size=(3,))
            shear = np.random.uniform(-0.2, 0.2, size=(3,))
            rotation = np.random.uniform(0., 90., size=(3,))
            translation = np.random.uniform(-30., 30., size=(3,))
            obj.affine_material(2, 1, scale, shear, rotation, translation, False)
        else:
            obj.replace_material(2, 1)
            
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0., 360.))
        
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        obj.save_stats(out_folder, i*num_angles, num_angles)

if __name__ == "__main__":
    config_fname = "playdoh.ini"
    np.random.seed(seed = 6)
    
    playdoh_gt_gen(config_fname)
    #playdoh_augment(config_fname)
