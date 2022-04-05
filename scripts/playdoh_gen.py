import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import shutil

import flexsim
    
def get_playdoh_volume_stats(obj):
    mat_counts = obj.get_stats()
    
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
        kwargs = {'src_num' : 2, 'dest_num' : 1, 'num_clusters' : remove_fo}
        obj.modify_volume(flexsim.transform.replace_material_cluster, kwargs)
        
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
    
    for subfolder in subfolders[:1]:
        playdoh_basic_augmentation(config_fname, input_root / subfolder, "{}_noisy".format(subfolder), 0)
    
if __name__ == "__main__":
    config_fname = "playdoh.ini"
    np.random.seed(seed = 6)
    
    # Generate simulated projections based on real volume
    batch_replication(config_fname)
    
    # Generate new volumes by removing clusters of stone
    #batch_basic_augmentation(config_fname)
