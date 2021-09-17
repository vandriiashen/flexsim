import numpy as np
import imageio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
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
    
    num_angles = aug_angles
    sample_size = aug_samples
    
    mo_matrix = np.zeros((sample_size, 9))
    for i in [0, 4, 8]:
        mo_matrix[:, i] = np.random.uniform(0.8, 1.1, sample_size)
    for i in [1, 2, 3, 5, 6, 7]:
        mo_matrix[:, i] = np.random.uniform(-0.3, 0.3, sample_size)
    mo_matrix = np.resize(mo_matrix, (sample_size, 3, 3))

    moz_x = np.random.uniform(0.5, 1.1, sample_size)
    moz_y = np.random.uniform(0.5, 1.1, sample_size)
    moz_z = np.random.uniform(0.5, 1.1, sample_size)
    foz_x = np.random.uniform(0.5, 1.1, sample_size)
    foz_y = np.random.uniform(0.5, 1.1, sample_size)
    foz_z = np.random.uniform(0.5, 1.1, sample_size)
    fos_x = np.random.uniform(-150., 150., sample_size)
    fos_y = np.random.uniform(-200., 300., sample_size)
    fos_z = np.random.uniform(-150., 150., sample_size)
    
    for i in tqdm(range(sample_size)):
        obj = ObjectCreator(obj_shape, energy_bins)
        print(mo_matrix[i,:,:])
        vol = obj.create_flexray_volume(model_fname, [0.025, 0.07], (fos_x[i], fos_y[i], fos_z[i]), (foz_x[i], foz_y[i], foz_z[i]), mo_matrix[i,:,:])
        noise = NoiseModel((760, num_angles, 972))
        proj = Projector(obj, mat, noise, num_angles, config)
        proj.read_flexray_geometry(obj_folder, (i, 360+i))
    
        proj.create_projection(i*num_angles, out_folder, 90)
        proj.create_gt(i*num_angles, out_folder)
        
def default_process_fod(obj_folder, out_folder, sim_config, mat_config):
    model_fname = obj_folder / "segm.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins, mat_config)
    obj = ObjectCreator(obj_shape, energy_bins, mat)
    #vol = obj.create_artificial_volume()
    
    param = {'n_x' : 0., 'n_y' : 0., 'n_z' : 1.0,
             'a_x' : 500., 'a_y' : 500., 'a_z' : 300.,
             'radius' : 50.}
    vol = obj.create_flexray_volume(model_fname, param)
    obj.save_volume(out_folder)
    
    num_angles = 501
    noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
    proj = Projector(obj, mat, noise, num_angles, sim_config)
    
    for i in range(1):
        proj.read_flexray_geometry(obj_folder / "good", (90*i, 90*i+90))
        print(proj.geom.parameters)
        log = proj.create_projection(i*num_angles, out_folder, 70)
        #proj.create_reconstruction(log, i*num_angles, out_folder, 70)
        proj.create_gt(i*num_angles, out_folder)
        
def walnut_tunnel(obj_folder, out_folder, aug_samples, aug_angles, sim_config, mat_config):
    model_fname = obj_folder / "segm.npy"
    obj_shape = get_volume_properties(model_fname)
    energy_bins = 100
    mat = MaterialHandler(energy_bins, mat_config)
    
    num_angles = aug_angles
    sample_size = aug_samples
    
    n_x = np.random.uniform(0., 1.0, sample_size)
    n_y = np.random.uniform(0., 1.0, sample_size)
    n_z = np.random.uniform(0., 1.0, sample_size)
    a_x = np.random.uniform(400., 600.0, sample_size)
    a_y = np.random.uniform(400., 600.0, sample_size)
    a_z = np.random.uniform(300., 300.0, sample_size)
    radius = np.random.uniform(30., 80.0, sample_size)
    
    for i in tqdm(range(sample_size)):
        obj = ObjectCreator(obj_shape, energy_bins, mat)
        param = {'n_x' : n_x[i], 'n_y' : n_y[i], 'n_z' : n_z[i],
                 'a_x' : a_x[i], 'a_y' : a_y[i], 'a_z' : a_z[i],
                 'radius' : radius[i]}
        vol = obj.create_flexray_volume(model_fname, param)
        noise = NoiseModel((obj_shape[0], num_angles, obj_shape[2]))
        proj = Projector(obj, mat, noise, num_angles, sim_config)
        proj.read_flexray_geometry(obj_folder / "good", (i, 360+i))
    
        proj.create_projection(i*num_angles, out_folder, 70)
        proj.create_gt(i*num_angles, out_folder)
    
def convert(inp_folder, out_folder, obj_name, aug_samples, aug_angles):
    gt_folder = convert_folder / "gt_{}".format(obj_name)
    poly_folder = convert_folder / "poly_{}".format(obj_name)
    noisy_folder = convert_folder / "noisy_{}".format(obj_name)
    
    gt_folder.mkdir(exist_ok=True)
    poly_folder.mkdir(exist_ok=True)
    noisy_folder.mkdir(exist_ok=True)
    for i in range(aug_samples*aug_angles):
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
    
    mat_config = config['Materials']
    mat_config['material_count'] = int(parser['Materials'].get('material_count', 0))
    for i in range(mat_config['material_count']):
        par_name = 'lac_{}'.format(i+1)
        mat_config[par_name] = float(parser['Materials'].get(par_name, 0.))
    
    aug_samples = int(parser['Extra'].get('augmentation_samples', 1))
    aug_angles = int(parser['Extra'].get('augmentation_angles', 1))

    np.random.seed(seed = 3)
    #default_process_fod(obj_folder, out_folder, sim_config, mat_config)
    walnut_tunnel(obj_folder, out_folder, aug_samples, aug_angles, sim_config, mat_config)
    #default_process_augment(obj_folder, out_folder, aug_samples, aug_angles, sim_config)
    #convert(out_folder, convert_folder, convert_name, aug_samples, aug_angles)
