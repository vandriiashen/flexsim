import numpy as np
from pathlib import Path
from configparser import ConfigParser
import imageio
import cupy
import cupyx.scipy.ndimage

def get_volume_properties(obj_folder):
    '''Gets volume properties.
    
    :param obj_folder: Path to the folder containing slices of the segmentation.
    :type obj_folder: :class:`pathlib.Path`
    :return: Object shape (height, width, width)
    :rtype: :class:`np.ndarray`
    
    '''
    height = len(list(obj_folder.glob("*.tiff")))
    sl = imageio.imread(obj_folder / "slice_{:06d}.tiff".format(0))
    obj_shape = (height, *sl.shape)
    
    return obj_shape

def read_volume(obj_folder):
    '''Reads slices from the folder and creates np.ndarray containing the object volume
    
    :param obj_folder: Path to the folder containing slices of the segmentation.
    :type obj_folder: :class:`pathlib.Path`
    :return: Object volume
    :rtype: :class:`np.ndarray`
    
    '''
    obj_shape = get_volume_properties(obj_folder)
    vol = np.zeros(obj_shape)
    for i in range(obj_shape[0]):
        vol[i,:] = imageio.imread(obj_folder / "slice_{:06d}.tiff".format(i))
    
    return vol

def read_config(fname):
    '''Reads the config and converts strings with numbers into number types
    
    :param fname: Path to the .ini config flie
    :type fname: :class:`string`
    :return: Dictionary of configuration parameters
    :rtype: class:`dict`
    
    '''
    parser = ConfigParser()
    parser.read(fname)
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    
    sim_config = config['Simulation']
    sim_config['num_angles'] = int(parser['Simulation'].get('num_angles', 1))
    sim_config['augmentation_samples'] = int(parser['Simulation'].get('augmentation_samples', 1))
    sim_config['energy_bins'] = int(parser['Simulation'].get('energy_bins', 1))
    sim_config['noise'] = parser['Simulation'].getboolean('noise')
    sim_config['save_noiseless'] = parser['Simulation'].getboolean('save_noiseless')
    
    mat_config = config['Materials']
    mat_config['material_count'] = int(parser['Materials'].get('material_count', 0))
    for i in range(mat_config['material_count']):
        par_name = 'lac_{}'.format(i+1)
        mat_config[par_name] = float(parser['Materials'].get(par_name, 0.))
        
    noise_config = config['Noise']
    noise_config['Flatfield'] = float(parser['Noise'].get('Flatfield', 1.))
    noise_config['Poisson_scaling'] = float(parser['Noise'].get('Poisson_scaling', 1.))
    noise_config['Gaussian_std'] = float(parser['Noise'].get('Gaussian_std', 0.))
    noise_config['Blur_width'] = float(parser['Noise'].get('Blur_width', 1.))
    
    return config

def apply_median_filter(vol):
    ''' Applies median filter to the volume using GPU acceleration (`cupyx.scipy.ndimage.median_filter`).
        
    :param vol: Array containing the object's model
    :type vol: :class:`np.ndarray`
    :return: Filtered volume
    :rtype: :class:`np.ndarray`
        
    '''
    vol_gpu = cupy.asarray(vol)
    vol_gpu = cupyx.scipy.ndimage.median_filter(vol_gpu, 3)
    vol_cpu = vol_gpu.get()
        
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
        
    return vol_cpu

def generate_affine_matrix(zoom, a, b, c, dist):
    zoom_mat = np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, zoom]])
    a_mat = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    b_mat = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    c_mat = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    
    dist_mat = np.zeros((3,3))
    main_el = np.random.uniform(1-dist, 1+dist, size=(3,))
    off_el = np.random.uniform(-dist, dist, size=(6,))
    dist_mat[0,0] = main_el[0]
    dist_mat[1,1] = main_el[1]
    dist_mat[2,2] = main_el[2]
    dist_mat[0,1] = off_el[0]
    dist_mat[0,2] = off_el[1]
    dist_mat[1,2] = off_el[2]
    dist_mat[1,0] = off_el[3]
    dist_mat[2,0] = off_el[4]
    dist_mat[2,1] = off_el[5]
    
    res = np.matmul(zoom_mat, a_mat)
    res = np.matmul(res, b_mat)
    res = np.matmul(res, c_mat)
    res = np.matmul(res, dist_mat)
    
    return res
