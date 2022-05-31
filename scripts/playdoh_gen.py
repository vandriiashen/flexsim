import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
import skimage
from tqdm import tqdm
import shutil
import flexsim
    
def write_playdoh_volume_stats(obj, out_folder):
    mat_counts = obj.get_stats()
    
    labels, nfeatures = ndimage.label(obj.volume == 2)
    sample_class = nfeatures
        
    stat_line = "{},{},{}".format(",".join(str(num) for num in mat_counts), nfeatures, sample_class)
    with open(out_folder / "volume_info.csv", "w") as f:
        f.write("Playdoh,Pebble,Pebble_objects,Sample_class\n")
        f.write(stat_line)
        
def remove_stone_objects(obj, remove_fo):
    '''Simple object modification function that removes FOs from the volume.
    '''
    kwargs = {'src_num' : 2, 'dest_num' : 1, 'num_clusters' : remove_fo}
    obj.modify_volume(flexsim.transform.replace_material_cluster, kwargs)
    
def playdoh_basic_augmentation(config_fname, input_folder, out_subfolder, remove_fo = 0):
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
    proj.read_flexray_geometry(input_folder, (0, 360), 2)

    obj.set_flexray_volume(obj_vol_folder)
    if remove_fo != 0:
        remove_stone_objects(obj, remove_fo)
        
    proj.create_projection(0, out_folder, 90)
    obj.save_volume(out_folder)
    write_playdoh_volume_stats(obj, out_folder)
    
def modify_main_object(obj):
    ''' Object modification function that performs affine transformation of the whole object
    '''
    scale = np.random.uniform(0.8, 1.2, size=(3,))
    shear = np.random.uniform(-0.2, 0.2, size=(3,))
    rotation = np.random.uniform(-90., 90., size=(3,))
    translation = (0., 0., 0.)
    print("Main object transformation: {}, {}, {}, {}".format(scale, shear, rotation, translation))
    kwargs = {'mat_num' : 3, 'scale' : scale, 'shear' : shear, 'rotation' : rotation, 'translation' : translation}
    obj.modify_volume(flexsim.transform.affine_volume, kwargs)
    
def modify_duplicate_foreign_object(obj):
    '''Object modification function that creates two FOs by performing affine transformations on one real FO.
    '''
    
    #size = 6 because we generate parameters for two tranformations
    scale = np.random.uniform(0.6, 1.8, size=(6,))
    shear = np.random.uniform(-0.2, 0.2, size=(6,))
    rotation = np.random.uniform(-10., 10., size=(6,))
    translation1 = np.random.uniform(20., 60., size=(3,))
    translation2 = np.random.uniform(-60., -20., size=(3,))
    translation = np.concatenate((translation1, translation2))
    print("Foreign object transformation #1: {}, {}, {}, {}".format(scale[:3], shear[:3], rotation[:3], translation[:3]))
    print("Foreign object transformation #2: {}, {}, {}, {}".format(scale[3:], shear[3:], rotation[3:], translation[3:]))
    kwargs = {'scale' : scale, 'shear' : shear, 'rotation' : rotation, 'translation' : translation}
    obj.modify_volume(flexsim.transform.duplicate_affine_pebble, kwargs)
    
    
def playdoh_triple_generation(config_fname, input_folder, output_subfolders):
    ''' Create artificial volumes and the corresponding X-ray projections.
    This function is used for complex generation, it supports MO and FO modification
    '''
    
    print("Triple generation")
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
    proj.read_flexray_geometry(input_folder, (0, 360), 2)
    
    obj.set_flexray_volume(obj_vol_folder)

    modify_main_object(obj)
    # Make two foreign object by applying affine transformation twice. Make sure that two separate objects are generated
    num_fo = 1
    volume_save = obj.get_volume()
    while num_fo != 2:
        obj.set_volume(volume_save)
        modify_duplicate_foreign_object(obj)
        labels, nfeatures = ndimage.label(obj.volume == 2)
        props = skimage.measure.regionprops(labels)
        print("Number of FO = {}".format(nfeatures))
        for prop in props:
            print(prop.area)
        num_fo = nfeatures
        if num_fo != 2:
            print("Wrong number of foreign objects. Need to generate different parameters.")
    
    volume_save = obj.get_volume()
    for i in range(out_samples):
        obj.set_volume(volume_save)
        if i != 0:
            remove_stone_objects(obj, i)
        proj.create_projection(0, out_folders[i], 90)
        obj.save_volume(out_folders[i])
        write_playdoh_volume_stats(obj, out_folders[i])
        
def batch_generation(input_root, config_fname, generation_base_samples):
    '''Create new X-ray images by modifying the main and foreign object.
    The number of artificial volumes to generate is taken from the config file.
    '''
    
    config = flexsim.utils.read_config(config_fname)
    # The samples will be generated in groups of 3, so //3
    augmentation_samples = config['Simulation']['augmentation_samples'] // 3
    
    subfolders = []
    for sample in generation_base_samples:
        path = input_root / sample
        data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)
        sample_class = data[-1]
        assert sample_class == 1
        
    for sample in generation_base_samples:
        sample_basename = sample.split('_')[0]
        print(sample_basename)
        for i in tqdm(range(augmentation_samples)):
            aug_name = "{}mod{:03d}".format(sample_basename, i)
            out_folders = ["{}_2fo".format(aug_name), "{}_1fo".format(aug_name), "{}_0fo".format(aug_name)]
            playdoh_triple_generation(config_fname, input_root / sample, out_folders)
    
def batch_basic_augmentation(input_root, config_fname):
    '''Create new X-ray projections by only modifying the foreign object.
    For every real sample, three artificial objects are created. The original sample is from Class 2.
    '''
    
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
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_noisy".format(subfolder), 0)
    
if __name__ == "__main__":
    config_fname = "playdoh.ini"
    input_root = Path("/path/to/data")
    np.random.seed(seed = 6)
    
    # Generate simulated projections based on real volume
    #batch_replication(input_root, config_fname)
    
    # Generate new volumes by removing clusters of stone
    #batch_basic_augmentation(input_root, config_fname)
    
    # Generate new volumes using affine transformations
    generation_base_samples = ['Object10_Scan20W']
    batch_generation(input_root, config_fname, generation_base_samples)
