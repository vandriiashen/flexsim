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
    
    #gt_folder = Path(out_folder) / "GT"
    #shutil.copytree(gt_folder, Path(obj_folder) / "gt")
    
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
        '''
        scale = np.random.uniform(0.8, 1.2, size=(3,))
        shear = np.random.uniform(-0.1, 0.1, size=(3,))
        rotation = np.random.uniform(0., 20., size=(3,))
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
        '''
        if i < aug_samples // 2:
            scale = np.random.uniform(0.6, 1.8, size=(3,))
            shear = np.random.uniform(-0.2, 0.2, size=(3,))
            rotation = np.random.uniform(0., 90., size=(3,))
            #translation = np.random.uniform(-30., 30., size=(3,))
            translation = (20., 20., -80.)
            obj.affine_material(2, 1, scale, shear, rotation, translation, False)
        else:
            obj.replace_material(2, 1)
            
        proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0., 360.))
        
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        obj.save_stats(out_folder, i*num_angles, num_angles)
        
def playdoh_augment_alternate(config_fname):
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
    
    for i in tqdm(range(aug_samples//2)):
        obj1 = flexsim.ObjectCreator(obj_shape, mat)
        obj1.set_flexray_volume(obj_vol_folder)
        obj2 = flexsim.ObjectCreator(obj_shape, mat)
        obj2.set_flexray_volume(obj_vol_folder)
        
        scale = np.random.uniform(0.8, 1.2, size=(3,))
        shear = np.random.uniform(-0.1, 0.1, size=(3,))
        rotation = np.random.uniform(0., 20., size=(3,))
        translation = (0., 0., 0.)
        obj1.affine_volume(scale, shear, rotation, translation, False)
        obj2.affine_volume(scale, shear, rotation, translation, False)
        obj1.split_clusters(1, 0, 15, 1, False)
        obj2.split_clusters(1, 0, 15, 1, False)
        
        scale = np.random.uniform(0.6, 1.8, size=(3,))
        shear = np.random.uniform(-0.2, 0.2, size=(3,))
        rotation = np.random.uniform(0., 90., size=(3,))
        translation = np.random.uniform(-40., 40., size=(3,))
        obj1.affine_material(2, 1, scale, shear, rotation, translation, False)
        
        proj = flexsim.Projector(obj1, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0., 360.))
        
        proj.create_projection(2*i*num_angles, out_folder, 90)
        proj.create_gt(2*i*num_angles, out_folder)
        
        obj1.save_stats(out_folder, 2*i*num_angles, num_angles)
        
        obj2.replace_material(2, 1)
        
        proj = flexsim.Projector(obj2, mat, noise, config['Simulation'])
        proj.read_flexray_geometry(obj_folder, (0., 360.))
        
        proj.create_projection((2*i+1)*num_angles, out_folder, 90)
        proj.create_gt((2*i+1)*num_angles, out_folder)
            
        obj2.save_stats(out_folder, (2*i+1)*num_angles, num_angles)
        
def playdoh_nofo_gen(config_fname, obj_folder):
    '''Generates projections without augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    #obj_folder = Path(config['Paths']['obj_folder'])
    obj_folder = Path(obj_folder)
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
    #obj.replace_material(2, 1)
        
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360))
        
    proj.create_projection(0, out_folder, 90)
    
    log_folder = Path(out_folder) / "90" / "Log"
    shutil.copytree(log_folder, Path(obj_folder) / "sim_log")
        
def playdoh_augment_single(config_fname, case_num):
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
        
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
    
    if case_num == 0:
        '''No augmentations'''
        pass
    if case_num == 1:
        '''FO translation'''
        scale = (1., 1., 1.)
        shear = (0., 0., 0.)
        rotation = (0., 0., 0.)
        translation = (80., 0., 0.)
        obj.affine_material(2, 1, scale, shear, rotation, translation, False)
    if case_num == 2:
        '''Change of FO size'''
        scale = (1.4, 1.1, 1.)
        shear = (0., 0., 0.)
        rotation = (0., 0., 0.)
        translation = (0., 0., 0.)
        obj.affine_material(2, 1, scale, shear, rotation, translation, False)
    if case_num == 3:
        '''Affine transformation of the sample'''
        scale = (1.3, 1.1, 1.2)
        shear = (0.1, 0.1, 0.)
        rotation = (0., 0., 40.)
        translation = (0., 0., 0.)
        obj.affine_volume(scale, shear, rotation, translation, False)
            
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0., 360.))
        
    proj.create_projection(0, out_folder, 90)
    proj.create_gt(0, out_folder)
    obj.save_stats(out_folder, 0, num_angles)
    
def get_playdoh_volume_stats(obj):
    mat_counts = []
    for i in range(1, obj.mat.mat_count+1):
        mat_counts.append(np.count_nonzero(obj.volume == i))
    mat_counts = np.array(mat_counts)
    
    labels, nfeatures = ndimage.label(obj.volume == 2)
    sample_class = nfeatures
        
    stat_line = "{},{},{}".format(",".join(str(num) for num in mat_counts), nfeatures, sample_class)
    return stat_line
    
def playdoh_basic_augmentation(config_fname, input_folder, out_subfolder, remove_fo = 0):
    '''Generates projections without augmentation to get GT.
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = input_folder
    obj_vol_folder = obj_folder / "segm"
    out_folder = Path(config['Paths']['out_folder']) / out_subfolder
    out_folder.mkdir(exist_ok=True)
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
    if remove_fo != 0:
        obj.replace_material_cluster(2, 1, remove_fo)
        
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360), 2)
        
    proj.create_projection(0, out_folder, 90)
    stat_line = get_playdoh_volume_stats(obj)
    with open(out_folder / "volume_info.csv", "w") as f:
        f.write("Playdoh, Pebble, Pebble_objects, Sample_class\n")
        f.write(stat_line)
    
def batch_basic_augmentation(config_fname):
    input_root = Path("../../../Data/Generation/Playdoh/Training/")
    subfolders = []
    for path in input_root.iterdir():
        if path.is_dir():
            data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)
            sample_class = data[-1]
            if sample_class == 2:
                subfolders.append(path.name)
    subfolders = sorted(subfolders)
    print(subfolders)
    
    for subfolder in subfolders[:]:
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_2fo".format(subfolder), 0)
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_1fo".format(subfolder), 1)
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_0fo".format(subfolder), 2)
        
def batch_replication(config_fname):
    input_root = Path("../../../Data/Generation/Playdoh/Training/")
    subfolders = []
    for path in input_root.iterdir():
        if path.is_dir():
            subfolders.append(path.name)
    subfolders = sorted(subfolders)
    print(subfolders)
    
    for subfolder in subfolders[:]:
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_noisy".format(subfolder), 0)
    
if __name__ == "__main__":
    config_fname = "playdoh.ini"
    np.random.seed(seed = 6)
    
    # Generate simulated projections based on real volume
    #batch_replication(config_fname)
    
    # Generate new volumes by removing clusters of stone
    batch_basic_augmentation(config_fname)
