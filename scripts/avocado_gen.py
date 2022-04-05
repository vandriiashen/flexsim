import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import random
import shutil

import flexsim
    
def get_avocado_volume_stats(obj):
    mat_counts = obj.get_stats()
        
    thr = 10**-2
    air_ratio = mat_counts[3] / mat_counts.sum()
    sample_class = 0
    if air_ratio > thr:
        sample_class = 1
        
    stat_line = "{},{}".format(",".join(str(num) for num in mat_counts), sample_class)
    return stat_line

def avocado_generation(config_fname, input_folder, out_nofo, out_fo, vol_counts):
    '''
    '''
    config = flexsim.utils.read_config(config_fname)
    
    obj_folder = input_folder
    obj_vol_folder = obj_folder / "segm"
    out_nofo = Path(config['Paths']['out_folder']) / out_nofo
    out_nofo.mkdir(exist_ok=True)
    out_fo = Path(config['Paths']['out_folder']) / out_fo
    out_fo.mkdir(exist_ok=True)
    
    num_angles = config['Simulation']['num_angles']
    energy_bins = config['Simulation']['energy_bins']
    
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])
    
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_flexray_volume(obj_vol_folder)
    fruit_volume = vol_counts.sum()
    
    # Perform affine transformation of the whole fruit
    scale = np.random.uniform(0.8, 1.2, size=(3,))
    shear = np.random.uniform(-0.2, 0.2, size=(3,))
    rotation = (0., 0., 0.)
    translation = (0., 0., 0.)
    print("Main object transformation: {}, {}, {}, {}".format(scale, shear, rotation, translation))
    kwargs = {'mat_num' : 4, 'scale' : scale, 'shear' : shear, 'rotation' : rotation, 'translation' : translation}
    obj.modify_volume(flexsim.transform.affine_volume, kwargs)
    obj.save_volume(out_fo)
    volume_save = np.copy(obj.get_volume())
    
    # Generate a sample with a bit of air
    tg_percentage = np.random.uniform(0., 0.005)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 20
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    obj.save_volume(out_nofo)
    
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360), 4)
    proj.create_projection(0, out_nofo, 90)
    stat_line = get_avocado_volume_stats(obj)
    with open(out_nofo / "volume_info.csv", "w") as f:
        f.write("Peel,Meat,Seed,Air,Sample_class\n")
        f.write(stat_line)
        
    # Reset volume and generate a sample with the same main object geometry but different amount of air_ratio
    obj = flexsim.ObjectCreator(obj_shape, mat)
    obj.set_volume(volume_save)
    tg_percentage = np.random.uniform(0.01, 0.05)
    tg_count = int(fruit_volume * tg_percentage)
    total_clusters = 6
    kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.drop_few_clusters, 'verbose' : True}
    obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
    
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360), 4)
    proj.create_projection(0, out_fo, 90)
    stat_line = get_avocado_volume_stats(obj)
    with open(out_fo / "volume_info.csv", "w") as f:
        f.write("Peel,Meat,Seed,Air,Sample_class\n")
        f.write(stat_line)
    
def avocado_basic_augmentation(config_fname, input_folder, out_subfolder, remove_air = False, vol_counts = np.array([0, 1, 0, 0])):
    '''
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
    
    if remove_air:
        remove_all_flag = np.random.randint(0, 2)
        if remove_all_flag == 0:
            obj.replace_material(4, 2)
        else:
            fruit_volume = vol_counts.sum()
            print(vol_counts)
            print(fruit_volume)
            tg_percentage = np.random.uniform(0., 0.005)
            tg_count = int(fruit_volume * tg_percentage)
            print(tg_percentage, tg_count)
            total_clusters = 20
            kwargs = {'src_num' : 4, 'dest_num' : 2, 'num_classes' : total_clusters, 'tg_count' : tg_count, 'pick_func' : flexsim.transform.keep_few_clusters, 'verbose' : True}
            obj.modify_volume(flexsim.transform.remove_air_clusters, kwargs)
        
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])
    proj.read_flexray_geometry(obj_folder, (0, 360), 4)
        
    proj.create_projection(0, out_folder, 90)
    stat_line = get_avocado_volume_stats(obj)
    with open(out_folder / "volume_info.csv", "w") as f:
        f.write("Peel,Meat,Seed,Air,Sample_class\n")
        f.write(stat_line)
        
def batch_generation(config_fname, generation_base_samples):
    config = flexsim.utils.read_config(config_fname)
    # The samples will be generated in pairs, so //2
    augmentation_samples = config['Simulation']['augmentation_samples'] // 2
    
    input_root = Path("../../../Data/Generation/Avocado/Training/")
    subfolders = []
    vol_counts = {}
    for sample in generation_base_samples:
        path = input_root / sample
        data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)
        sample_class = data[-1]
        assert sample_class == 1
        vol_counts[sample] = data[:-1]
        
    for sample in generation_base_samples:
        sample_basename = "{}{}".format(sample.split('_')[0], sample.split('_')[1])
        print(sample_basename)
        for i in tqdm(range(augmentation_samples)):
            aug_name = "{}mod{:03d}".format(sample_basename, i)
            avocado_generation(config_fname, input_root / sample, "{}_nofo".format(aug_name), "{}_fo".format(aug_name), vol_counts[sample])
    
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
    
    for subfolder in subfolders:
        print(subfolder)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_fo".format(subfolder), False)
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_nofo".format(subfolder), True, vol_counts[subfolder])
        
def batch_replication(config_fname):
    input_root = Path("../../../Data/Generation/Avocado/Training/")
    subfolders = []
    for path in input_root.iterdir():
        if path.is_dir():
            subfolders.append(path.name)
    subfolders = sorted(subfolders)
    print(subfolders)
    
    for subfolder in subfolders[1:2]:
        avocado_basic_augmentation(config_fname, input_root / subfolder, "{}_noisy".format(subfolder), False)
            
if __name__ == "__main__":
    config_fname = "avocado.ini"
    # use different seeds for training and validation
    random_seed = 6
    np.random.seed(seed = random_seed)
    random.seed(random_seed)
    
    # Generate simulated projections based on real volume
    #batch_replication(config_fname)
    
    # Generate new volumes by removing air
    #batch_basic_augmentation(config_fname)
    
    # Generate new volumes using affine transformations
    generation_base_samples = ['s05_d09']
    batch_generation(config_fname, generation_base_samples)
