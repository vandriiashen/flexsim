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
    scale = np.random.uniform(0.8, 1.2, size=(3,))
    shear = np.random.uniform(-0.2, 0.2, size=(3,))
    rotation = (0., 0., 0.)
    translation = (0., 0., 0.)
    print("Main object transformation: {}, {}, {}, {}".format(scale, shear, rotation, translation))
    kwargs = {'mat_num' : 4, 'scale' : scale, 'shear' : shear, 'rotation' : rotation, 'translation' : translation}
    obj.modify_volume(flexsim.transform.affine_volume, kwargs)
    
def gen_class0_air(obj, fruit_volume):
    #t2
    #tg_percentage = np.random.uniform(0., 0.01)
    #t3
    tg_percentage = loguniform.rvs(1e-4, 0.01)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 20
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
def gen_class1_air(obj, fruit_volume):
    #t2
    #tg_percentage = np.random.uniform(0.01, 0.05)
    #t3
    tg_percentage = loguniform.rvs(0.01, 0.05)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 6
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.drop_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
def avocado_pair_generation(config_fname, input_folder, output_subfolders):
    '''
    '''
    print("Pair generation")
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
        
def avocado_generation(config_fname, input_folder, output_subfolders, vol_counts):
    '''
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
    fruit_volume = vol_counts.sum()
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    obj = flexsim.ObjectCreator(obj_shape, mat)
    
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(input_folder, (0, 360), 4)
        
    sample_operations = [gen_class0_air, gen_class1_air]
    for i in range(out_samples):
        obj.set_flexray_volume(obj_vol_folder)
        modify_main_object(obj)
        sample_operations[i](obj, fruit_volume)
        proj.create_projection(0, out_folders[i], 90)
        write_avocado_volume_stats(obj, out_folders[i])
        
def inpainting_remove_air(obj, fruit_volume):
    remove_all_flag = np.random.randint(0, 2)
    if remove_all_flag == 0:
        obj.replace_material(4, 2)
    else:
        
        print(vol_counts)
        print(fruit_volume)
        tg_percentage = np.random.uniform(0., 0.005)
        tg_count = int(fruit_volume * tg_percentage)
        print(tg_percentage, tg_count)
        total_clusters = 20
        kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
        obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
        
def avocado_basic_augmentation(config_fname, input_folder, out_subfolder, remove_air = False):
    '''
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
        #inpainting_remove_air(obj, fruit_volume)
        gen_class0_air(obj, fruit_volume)
        
    proj.create_projection(0, out_folder, 90)
    obj.save_volume(out_folder)
    write_avocado_volume_stats(obj, out_folder)
        
def batch_generation(config_fname, generation_base_samples):
    config = flexsim.utils.read_config(config_fname)
    # The samples will be generated in pairs, so //2
    augmentation_samples = config['Simulation']['augmentation_samples'] // 2
    
    input_root = Path("../../../Data/Generation/Avocado/Training/")
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
            # Training samples with different main object modifications
            #avocado_generation(config_fname, input_root / sample, out_folders, vol_counts[sample])
            
    
def batch_basic_augmentation(config_fname):
    input_root = Path("../../../Data/Generation/Avocado/Training/")
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
    
    for subfolder in subfolders[:]:
        print(subfolder)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_fo".format(subfolder), False)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_nofo".format(subfolder), True)
        
def batch_replication(config_fname):
    input_root = Path("../../../Data/Generation/Avocado/Training/")
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
    # use different seeds for training and validation
    #random_seed = 6
    random_seed = 7
    np.random.seed(seed = random_seed)
    random.seed(random_seed)
    
    # Generate simulated projections based on real volume
    #batch_replication(config_fname)
    
    # Generate new volumes by removing air
    batch_basic_augmentation(config_fname)
    
    # Generate new volumes using affine transformations
    generation_base_samples = ['s05_d09']
    #batch_generation(config_fname, generation_base_samples)
