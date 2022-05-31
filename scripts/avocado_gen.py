import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
from scipy.stats import loguniform
import random
import shutil
import copy

import flexsim
    
def write_avocado_volume_stats(obj, out_folder):
    mat_counts = obj.get_stats()
    thr = 10**-2
    air_ratio = mat_counts[3] / mat_counts.sum()
    sample_class = 0
    if air_ratio > thr:
        sample_class = 1
        
    stat_line = "{},{}".format(",".join(str(num) for num in mat_counts), sample_class)
    with open(out_folder / "volume_info.csv", "w") as f:
        f.write("Peel,Meat,Seed,Air,Sample_class\n")
        f.write(stat_line)

def modify_main_object(obj):
    ''' Object modification function that performs affine transformation of the whole object
    '''
    scale = np.random.uniform(0.8, 1.2, size=(3,))
    shear = np.random.uniform(-0.2, 0.2, size=(3,))
    rotation = (0., 0., 0.)
    translation = (0., 0., 0.)
    print("Main object transformation: {}, {}, {}, {}".format(scale, shear, rotation, translation))
    kwargs = {'mat_num' : 4, 'scale' : scale, 'shear' : shear, 'rotation' : rotation, 'translation' : translation}
    obj.modify_volume(flexsim.transform.affine_volume, kwargs)
    
def gen_class0_air(obj, fruit_volume):
    ''' Object modification function that removes air from the object to create a Class 0 sample.
    '''
    tg_percentage = loguniform.rvs(1e-4, 0.01)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 20
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
def gen_class1_air(obj, fruit_volume):
    ''' Object modification function that removes a small amount of air, and the resulting sample is still in Class 1.
    '''
    tg_percentage = loguniform.rvs(0.01, 0.05)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 6
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.drop_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
def remove_air_equidistant(obj, fruit_volume, total_samples, num):
    ''' Object modification function that removes air from the object to create a Class 0 sample.
    In this case, the target amount of air is chosen without random number generation.
    The target percentages are distributed from 0% to 1% with a constant step defined by the total number of samples.
    '''
    percentages = np.linspace(0., 0.01, total_samples, endpoint=False)
    print(percentages)
    tg_percentage = percentages[num]
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 20
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
def avocado_pair_generation(config_fname, input_folder, output_subfolders):
    ''' Create an artificial volume and the corresponding X-ray projections.
    This function is used for complex generation, it supports MO and FO modification
    '''
    
    config = flexsim.utils.read_config(config_fname)
    
    obj_vol_folder = input_folder / "segm"
    out_folders = []
    for subfolder in output_subfolders:
        out_folder = Path(config['Paths']['out_folder']) / subfolder
        out_folder.mkdir(exist_ok = True)
        out_folders.append(out_folder)
    out_samples = len(out_folders)
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    obj = flexsim.ObjectCreator(obj_shape, mat)
    
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(input_folder, (0, 360), 4)
    
    obj.set_flexray_volume(obj_vol_folder)
    modify_main_object(obj)
    # Save the volume after main object modification to generate a pair
    volume_save = obj.get_volume()
    fruit_volume = obj.get_stats().sum()
    
    sample_operations = [gen_class0_air, gen_class1_air]
    for i in range(out_samples):
        obj.set_volume(volume_save)
        sample_operations[i](obj, fruit_volume)
        proj.create_projection(0, out_folders[i], 90)
        obj.save_volume(out_folders[i])
        write_avocado_volume_stats(obj, out_folders[i])
        
def avocado_basic_augmentation(config_fname, input_folder, out_subfolder, total_samples = 1, num = 0, remove_air = False):
    ''' Create an artificial volume and the corresponding X-ray projections.
    This function is used for replication and basic augmentation, it only allows FO modification.
    '''
    
    config = flexsim.utils.read_config(config_fname)
    
    obj_vol_folder = input_folder / "segm"
    out_folder = Path(config['Paths']['out_folder']) / out_subfolder
    out_folder.mkdir(exist_ok=True)
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    obj = flexsim.ObjectCreator(obj_shape, mat)
    
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(input_folder, (0, 360), 4)
    
    obj.set_flexray_volume(obj_vol_folder)
    fruit_volume = obj.get_stats().sum()
    if remove_air:
        remove_air_equidistant(obj, fruit_volume, total_samples, num)
        
    proj.create_projection(0, out_folder, 90)
    obj.save_volume(out_folder)
    write_avocado_volume_stats(obj, out_folder)
        
def batch_generation(input_root, config_fname, generation_base_samples):
    '''Create new X-ray images by modifying the main and foreign object.
    The number of artificial volumes to generate is taken from the config file.
    '''
    
    config = flexsim.utils.read_config(config_fname)
    # The samples will be generated in pairs, so //2
    augmentation_samples = config['Simulation']['augmentation_samples'] // 2
    
    subfolders = []
    for sample in generation_base_samples:
        path = input_root / sample
        data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)
        sample_class = data[-1]
        assert sample_class == 1
        
    for sample in generation_base_samples:
        sample_basename = "{}{}".format(sample.split('_')[0], sample.split('_')[1])
        print(sample_basename)
        for i in tqdm(range(augmentation_samples)):
            aug_name = "{}mod{:03d}".format(sample_basename, i)
            out_folders = ["{}_nofo".format(aug_name), "{}_fo".format(aug_name)]
            
            # Training in pairs
            avocado_pair_generation(config_fname, input_root / sample, out_folders)
            
def batch_basic_augmentation(input_root, config_fname, sample_num = -1):
    '''Create new X-ray projections by only modifying the foreign object.
    For every real sample, two artificial objects are created.
    '''
    
    subfolders = []
    vol_counts = {}
    for path in input_root.iterdir():
        if path.is_dir():
            data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)
            print(path)
            print(data)
            sample_class = data[-1]
            if sample_class == 1:
                subfolders.append(path.name)
                vol_counts[path.name] = data[:-1]
    subfolders = sorted(subfolders)
    print(subfolders)
    
    if sample_num != -1:
        subfolders = subfolders[:sample_num]
    
    total_samples = len(subfolders)
    num = 0
    
    for subfolder in subfolders[:]:
        print(subfolder)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_fo".format(subfolder), total_samples, num, False)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_nofo".format(subfolder), total_samples, num, True)
        num += 1
        
def batch_replication(input_root, config_fname):
    ''' Simulate X-ray projections based on their reconstructions without any volume modification.
    This function is used to compare the neural network performance when trained on artifical data with training on real data.
    Noise properties can be changed in config to test noiseless and noisy data
    '''
    
    subfolders = []
    for path in input_root.iterdir():
        if path.is_dir():
            subfolders.append(path.name)
    subfolders = sorted(subfolders)
    print(subfolders)
    
    for subfolder in subfolders[:]:
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_noisy".format(subfolder), False)
            
if __name__ == "__main__":
    config_fname = "avocado.ini"
    input_root = Path("/path/to/data")
    random_seed = 6
    
    np.random.seed(seed = random_seed)
    random.seed(random_seed)
    
    # Generate simulated projections based on real volume
    #batch_replication(input_root, config_fname)
    
    # Generate new volumes by removing air
    #batch_basic_augmentation(input_root, config_fname, 4)
    
    # Generate new volumes using affine transformations
    generation_base_samples = ['s07_d09']
    batch_generation(input_root, config_fname, generation_base_samples)
