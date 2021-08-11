import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from configparser import ConfigParser

from ObjectCreator import ObjectCreator, get_volume_properties
from MaterialHandler import MaterialHandler
from NoiseModel import NoiseModel
from Projector import Projector
        
def default_process_augment(obj_folder, out_folder, aug_samples, aug_angles, config):
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins)
    obj = ObjectCreator(obj_shape, energy_bins)
    vol = obj.create_flexray_volume(model_fname, [0.025, 0.07])
    
    num_angles = aug_angles
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, config)
    '''
    zooms = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    shift_x = [0., 60., -20., 0., 0., 0., 0.]
    shift_y = [0., 0., 0., 30., -30., 0., 0.]
    shift_z = [0., 0., 0., 0., 0., -100., 100.]
    '''
    sample_size = aug_samples
    main_x = np.random.uniform(0.8, 1.2, sample_size)
    main_y = np.random.uniform(0.8, 1.2, sample_size)
    main_z = np.random.uniform(0.8, 1.2, sample_size)
    zooms = np.random.uniform(0.5, 1.1, sample_size)
    shift_x = np.random.uniform(-20., 60., sample_size)
    shift_y = np.random.uniform(-30., 30., sample_size)
    shift_z = np.random.uniform(-100., 100., sample_size)
    
    for i in range(sample_size):
        proj.read_flexray_geometry(obj_folder, (i, 360+i))
        proj.transform_main_geometry([main_x[i], main_y[i], main_z[i]])
        proj.transform_foreign_geometry(zooms[i], [shift_x[i]*main_x[i], shift_y[i]*main_y[i], shift_z[i]*main_z[i]])
        print(proj.geom.parameters)
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        
def default_process_fod(obj_folder, out_folder, config):
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins)
    obj = ObjectCreator(obj_shape, energy_bins)
    vol = obj.create_flexray_volume(model_fname, [0.025, 0.07])
    
    num_angles = 450
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, config)
    
    for i in range(4):
        proj.read_flexray_geometry(obj_folder, (90*i, 90*i+90))
        print(proj.geom.parameters)
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        
def default_process_single(obj_folder, out_folder, config):
    model_fname = obj_folder / "recon" / "volume.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins)
    obj = ObjectCreator(obj_shape, energy_bins)
    vol = obj.create_flexray_volume(model_fname, [0.025, 0.07])
    
    num_angles = 1
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, config)
    
    proj.read_flexray_geometry(obj_folder, (0, 360))
    proj.create_projection(0, our_folder, 90)
    proj.create_gt(0, our_folder)
    
def convert(inp_folder, out_folder, obj_name):
    gt_folder = convert_folder / "gt_{}".format(obj_name)
    poly_folder = convert_folder / "poly_{}".format(obj_name)
    noisy_folder = convert_folder / "noisy_{}".format(obj_name)
    
    gt_folder.mkdir(exist_ok=True)
    poly_folder.mkdir(exist_ok=True)
    noisy_folder.mkdir(exist_ok=True)
    for i in range(1800):
        noisy = imageio.imread(inp_folder / "90" / "Log" / "{}.tiff".format(i))
        poly = imageio.imread(inp_folder / "90" / "Noiseless" / "{}.tiff".format(i))
        gt = imageio.imread(inp_folder / "GT" / "{}.tiff".format(i))
        or_shape = noisy.shape
        tg_shape = (128,128)
        zoomed_noisy = ndimage.zoom(noisy, [tg_shape[0]/or_shape[0], tg_shape[1]/or_shape[1]])
        zoomed_poly = ndimage.zoom(poly, [tg_shape[0]/or_shape[0], tg_shape[1]/or_shape[1]])
        zoomed_gt = ndimage.zoom(gt, [tg_shape[0]/or_shape[0], tg_shape[1]/or_shape[1]])
        imageio.imsave(noisy_folder / "{:04d}.tiff".format(i), zoomed_noisy.astype(np.float32))
        imageio.imsave(poly_folder / "{:04d}.tiff".format(i), zoomed_poly.astype(np.float32))
        imageio.imsave(gt_folder / "{:04d}.tiff".format(i), zoomed_gt.astype(np.int32))

if __name__ == "__main__":
    parser = ConfigParser()
    parser.read("config.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    
    obj_folder = Path(config['Paths']['obj_folder'])
    out_folder = Path(config['Paths']['out_folder'])
    convert_folder = Path(config['Paths']['convert_folder'])
    convert_name = Path(config['Paths']['convert_name'])
    
    sim_config = config['Simulation']
    sim_config['noise'] = parser['Simulation'].getboolean('noise')
    sim_config['save_noiseless'] = parser['Simulation'].getboolean('save_noiseless')
    
    aug_samples = int(parser['Extra'].get('augmentation_samples', 1))
    aug_angles = int(parser['Extra'].get('augmentation_angles', 1))

    np.random.seed(seed = 3)
    #default_process_fod(obj_folder, out_folder, sim_config)
    default_process_augment(obj_folder, out_folder, aug_samples, aug_angles, sim_config)
    convert(out_folder, convert_folder, convert_name)
