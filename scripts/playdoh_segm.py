import numpy as np
from pathlib import Path
from configparser import ConfigParser
import imageio
import shutil
from flexdata import display
from flexdata import data
from flextomo import projector
from flexcalc import process
from flexcalc import analyze
import cupy
import cupyx.scipy.ndimage
import skimage
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def apply_median_filter(vol, size):
    ''' Applies median filter to the volume using GPU acceleration (`cupyx.scipy.ndimage.median_filter`).
    '''
    vol_gpu = cupy.asarray(vol)
    vol_gpu = cupyx.scipy.ndimage.median_filter(vol_gpu, size)
    vol_cpu = vol_gpu.get()
        
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
        
    return vol_cpu

def reconstruct(input_folder, output_folder, bh_correction = True, compound = 'H2O', density = 0.6):
    '''Reconstructs the volume using flexbox. Supports beam-hardening correction, and that is crucial for good segmentation.
    '''
    path = output_folder
    if bh_correction == True:
        save_path = path / "recon_bh"
    else:
        save_path = path / "recon"
    proj, geom = process.process_flex(input_folder, sample = 2, skip = 1, correct='cwi-flexray-2019-04-24')

    save_path.mkdir(exist_ok=True)

    geom.parameters['src_ort'] += (7-5.5) #see below for flexbox hard coded values
    geom['det_roll'] -= 0.25
    
    vol = projector.init_volume(proj)
    projector.FDK(proj, vol, geom)
    
    if bh_correction == True:
        density = density
        compound = compound
        energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound = compound, density = density, verbose = 2, plot_path = save_path)   
        proj_cor = process.equivalent_density(proj, geom, energy, spec, compound = compound, density = density, preview=False)
        vol_rec = np.zeros_like(vol)
        projector.FDK(proj_cor, vol_rec, geom)
        vol = vol_rec
    
    data.write_stack(save_path, 'slice', vol, dim = 0)
    
def segment(input_folder, output_folder, otsu_classes):
    '''Performs segmentation of the reconstructed volume.
    The number of classes for otsu should be explicitly given by the user.
    '''
    path = output_folder
    recon_path = path / "recon_bh"
    segm_path = path / "segm"
    segm_path.mkdir(exist_ok=True)
    
    height = len(list(recon_path.glob("*.tiff")))
    im_shape = (imageio.imread(recon_path / "slice_{:06d}.tiff".format(0))).shape
    vol = np.zeros((height, *im_shape))
    segm_vol = np.zeros_like(vol, dtype=np.uint8)
    for i in range(height):
        vol[i,:] = imageio.imread(recon_path / "slice_{:06d}.tiff".format(i))
        
    vol -= vol.min()
    vol /= vol.max()
    vol = skimage.img_as_ubyte(vol)
    vol = apply_median_filter(vol, 3)
    
    thr = skimage.filters.threshold_multiotsu(vol, classes=otsu_classes)
    
    if otsu_classes == 3:
        print("Playdoh/Stone thresholds: {:.2f} / {:.2f}".format(*thr))
        segm_vol[vol > thr[0]] = 1
        segm_vol[vol > thr[1]] = 2
    if otsu_classes == 2:
        print("Playdoh threshold: {:.2f}".format(thr[0]))
        segm_vol[vol > thr[0]] = 1
        
    for i in range(height):
        imageio.imwrite(segm_path / "slice_{:06d}.tiff".format(i), segm_vol[i,:])
        
def preprocess_proj(input_folder, output_folder, skip_proj):
    '''Applies darkfield- and flatfield-correction to projections and saves them to a separate folder.
    '''
    path = input_folder
    out = output_folder
    log_path = out / "log"
    log_path.mkdir(exist_ok=True)
    
    proj, flat, dark, geom = data.read_flexray(path, sample = 2, skip = 1, correct='cwi-flexray-2019-04-24')
    proj = process.preprocess(proj, flat, dark)
    proj = np.flip(proj, 0)
    
    for i in range(0,proj.shape[1],skip_proj):
        imageio.imwrite(log_path / "scan_{:06d}.tiff".format(i), proj[:,i,:])
        
def check_intensity(output_folder):
    '''Computes mean value and standard deviations for materials based on the segmentation.
    '''
    path = Path(output_folder)
    recon_path = path / "recon"
    segm_path = path / "segm"
    
    height = len(list(recon_path.glob("*.tiff")))
    im_shape = (imageio.imread(recon_path / "slice_{:06d}.tiff".format(0))).shape
    vol = np.zeros((height, *im_shape), dtype=np.float32)
    segm = np.zeros((height, *im_shape), dtype=np.int16)
    for i in range(height):
        vol[i,:] = imageio.imread(recon_path / "slice_{:06d}.tiff".format(i))
        segm[i,:] = imageio.imread(segm_path / "slice_{:06d}.tiff".format(i))
        
    playdoh_mean = vol[segm == 1].mean()
    playdoh_std = vol[segm == 1].std()
    stone_mean = vol[segm == 2].mean()
    stone_std = vol[segm == 2].std()
    
    print("Playdoh mean intensity = {}".format(playdoh_mean))
    print("Playdoh intensity std = {}".format(playdoh_std))
    print("Stone mean intensity = {}".format(stone_mean))
    print("Stone intensity std = {}".format(stone_std))
        
def multiple_objects_process():
    input_root = Path('../../../Data/Real/AutomatedFOD/')
    #input_root = Path('../../../Data/Real/Playdoh_extra/')
    #input_root = Path('../../../Data/Real/Playdoh_nofo/')
    output_root = Path('../../../Data/Generation/Playdoh/Simulation')
    sub_folders = []
    
    for path in input_root.iterdir():
        if path.is_dir():
            sub_folders.append(path.name)

    '''
    check_num = [10]
    for num in check_num:
        sub_folders.append('Object{}_Scan20W'.format(num))
    '''
        
    sub_folders = sorted(sub_folders)
    
    input_folders = [input_root / sub_folder for sub_folder in sub_folders]
    output_folders = [output_root / sub_folder for sub_folder in sub_folders]
    assert len(input_folders) == len(output_folders)
    
    for i in range(len(input_folders)):
        print(input_folders[i])
        output_folders[i].mkdir(exist_ok=True)
        
        preprocess_proj(input_folders[i], output_folders[i], 5)
        shutil.copy(input_folders[i] / 'scan settings.txt', output_folders[i])
        reconstruct(input_folders[i], output_folders[i], bh_correction=False, compound='H2O', density=0.6)
        
        # Change the number of otsu classes depending on the presence of pebble stone in the sample
        segment(input_folders[i], output_folders[i], 3)
        #segment(input_folders[i], output_folders[i], 2)

if __name__ == "__main__":
    multiple_objects_process()
    
