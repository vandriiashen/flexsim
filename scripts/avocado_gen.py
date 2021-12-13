import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import shutil

import flexsim

def compute_seed_properties(obj):
    seed_com = ndimage.center_of_mass(obj.volume == 3)
    Z, Y, X = np.ogrid[:obj.size[0], :obj.size[1], :obj.size[2]]
    Z -= int(seed_com[0])
    Y -= int(seed_com[1])
    X -= int(seed_com[2])
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_seed = R[obj.volume == 3].max()
    
    return (seed_com, R_seed)
        
def avocado_gt_gen_full_angles(config_fname):
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
        
    for i in range(4):
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (90*i, 90*(i+1)))
        
        proj.create_projection(360*i, out_folder, 90)
        proj.create_gt(360*i, out_folder)
        
    gt_folder = Path(out_folder) / "GT"
    shutil.copytree(gt_folder, Path(obj_folder) / "gt")
    
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
    
    with open(out_folder / "stats.csv", "w") as f:
        f.write("Proj_num,Peel,Meat,Seed,Air\n")
    
    for i in tqdm(range(aug_samples)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)
        
        scale = np.random.uniform(0.8, 1.2, size=(3,))
        shear = np.random.uniform(-0.2, 0.2, size=(3,))
        rotation = (0., 0., 0.)
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
        
        if i < aug_samples // 3:
            total_regions = np.random.randint(25, 30)
            drop_regions = np.random.randint(0, 10)
            obj.split_clusters(4, 2, total_regions, drop_regions, False)
        elif i < 2 * aug_samples // 3:
            obj.replace_material(4, 2)
        else:
            total_regions = np.random.randint(60, 80)
            keep_regions = np.random.randint(0, 8)
            drop_regions = total_regions - keep_regions
            obj.split_clusters(4, 2, total_regions, drop_regions, False)
            
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0, 360))
            
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        obj.save_stats(out_folder, i*num_angles, num_angles)
        
def avocado_add_air(config_fname):
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
    
    with open(out_folder / "stats.csv", "w") as f:
        f.write("Proj_num,Peel,Meat,Seed,Air\n")
            
    for i in tqdm(range(aug_samples)):
        obj = flexsim.ObjectCreator(obj_shape, mat)
        obj.set_flexray_volume(obj_vol_folder)        
        
        if i < aug_samples // 3:
            small_spheres_num = np.random.randint(1, 2)
            for j in range(small_spheres_num):
                c_z = np.random.randint(60, 120)
                c_y = np.random.randint(180, 300)
                c_x = np.random.randint(180, 300)
                sph_centre = np.array([c_z, c_y, c_x])
                sph_radius = np.random.uniform(1., 5.)
                obj.create_sphere(2, 4, sph_centre, sph_radius)
            seed_com, r_seed = compute_seed_properties(obj)
            obj.create_spherical_fragments(2, 4, seed_com, r_seed, 4., np.pi/2, 5)
        elif i < 2 * aug_samples // 3:
            obj.replace_material(4, 2)
        else:
            small_spheres_num = np.random.randint(1, 3)
            for j in range(small_spheres_num):
                c_z = np.random.randint(60, 120)
                c_y = np.random.randint(180, 300)
                c_x = np.random.randint(180, 300)
                sph_centre = np.array([c_z, c_y, c_x])
                sph_radius = np.random.uniform(5., 12.)
                obj.create_sphere(2, 4, sph_centre, sph_radius)
            seed_com, r_seed = compute_seed_properties(obj)
            obj.create_spherical_fragments(2, 4, seed_com, r_seed, 8., np.pi, 10)
            
        scale = np.random.uniform(0.8, 1.2, size=(3,))
        shear = np.random.uniform(-0.2, 0.2, size=(3,))
        rotation = (0., 0., 0.)
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
                    
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0, 360))
            
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        obj.save_stats(out_folder, i*num_angles, num_angles)
        
def avocado_add_air_single(config_fname, case_num):
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
    
    with open(out_folder / "stats.csv", "w") as f:
        f.write("Proj_num,Peel,Meat,Seed,Air\n")
            
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
    
    if case_num == 0:
        '''No augmentations'''
        pass
    if case_num == 1:
        '''Small amount of air'''
        np.random.seed(seed = 15)
        obj.create_sphere(2, 4, (70, 250, 180), 5.)
        obj.create_sphere(2, 4, (105, 200, 290), 4.)
        seed_com, r_seed = compute_seed_properties(obj)
        obj.create_spherical_fragments(2, 4, seed_com, r_seed, 5., np.pi/2, 4)
    if case_num == 2:
        '''Large amount of air'''
        np.random.seed(seed = 7)
        obj.create_sphere(2, 4, (90, 220, 220), 7.)
        obj.create_sphere(2, 4, (105, 200, 290), 4.)
        seed_com, r_seed = compute_seed_properties(obj)
        obj.create_spherical_fragments(2, 4, seed_com, r_seed, 6., np.pi, 10)
    if case_num == 3:
        '''Large amount of air + main object transform'''
        np.random.seed(seed = 3)
        obj.create_sphere(2, 4, (90, 220, 220), 7.)
        obj.create_sphere(2, 4, (70, 250, 180), 5.)
        seed_com, r_seed = compute_seed_properties(obj)
        obj.create_spherical_fragments(2, 4, seed_com, r_seed, 5., np.pi, 10)
        
        scale = (0.9, 1.1, 1.1)
        shear = (0.1, 0., 0.)
        rotation = (0., 0., 0.)
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
                    
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360))
            
    proj.create_projection(0, out_folder, 90)
    proj.create_gt(0, out_folder)
    obj.save_stats(out_folder, 0, num_angles)
            
if __name__ == "__main__":
    config_fname = "avocado.ini"
    # use different seeds for training and validation
    np.random.seed(seed = 6)
    
    avocado_gt_gen(config_fname)
    #avocado_reduce_air(config_fname)
    #avocado_add_air(config_fname)
    #avocado_add_air_single(config_fname, 3)
