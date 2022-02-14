import numpy as np
from pathlib import Path
from configparser import ConfigParser
import imageio
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

def reconstruct(input_folder, bh_correction):
    '''Reconstructs the volume using flexbox. Slices will be written to recon/ subfolder.
    '''
    path = Path(input_folder)
    if bh_correction == True:
        save_path = path / "recon_bh"
    else:
        save_path = path / "recon"
    proj, geom = process.process_flex(path, sample = 2, skip = 1)

    save_path.mkdir(exist_ok=True)

    geom.parameters['src_ort'] += (7-5.5) #see below for flexbox hard coded values
    geom['det_roll'] -= 0.25
    
    vol = projector.init_volume(proj)
    projector.FDK(proj, vol, geom)
    
    if bh_correction == True:
        density = 0.6
        energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound = 'H2O', density = density)   
        proj_cor = process.equivalent_density(proj, geom, energy, spec, compound = 'H2O', density = density, preview=False)
        vol_rec = np.zeros_like(vol)
        projector.FDK(proj_cor, vol_rec, geom)
        vol = vol_rec
    
    data.write_stack(save_path, 'slice', vol, dim = 0)
    
def segment(input_folder, otsu_classes):
    '''Segments every slice with thresholding. Segmented slices will be written to segm/ subfolder.
    '''
    path = Path(input_folder)
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
        
def preprocess_proj(input_folder, skip_proj):
    '''Applies darkfield- and flatfield-correction to projections and saves them to a separate folder.
    '''
    path = Path(input_folder)
    log_path = path / "log"
    log_path.mkdir(exist_ok=True)
    
    proj, flat, dark, geom = data.read_flexray(path, sample = 2, skip = 1)
    proj = process.preprocess(proj, flat, dark)
    proj = np.flip(proj, 0)
    
    for i in range(0,proj.shape[1],skip_proj):
        imageio.imwrite(log_path / "scan_{:06d}.tiff".format(i), proj[:,i,:])
        
def count_materials(input_folder):
    '''Computes the number of voxels filled with different materials
    '''
    path = Path(input_folder)
    segm_path = path / "segm"
    
    height = len(list(segm_path.glob("*.tiff")))
    im_shape = (imageio.imread(segm_path / "slice_{:06d}.tiff".format(0))).shape
    segm = np.zeros((height, *im_shape), dtype=np.int16)
    for i in range(height):
        segm[i,:] = imageio.imread(segm_path / "slice_{:06d}.tiff".format(i))
        
    playdoh_count = np.count_nonzero(segm == 1)
    pebble_count = np.count_nonzero(segm == 2)
    
    print("Playdoh, Pebble")
    print("{},{}".format(playdoh_count, pebble_count))
        
def check_intensity(input_folder):
    '''Computes mean value and standard deviations for materials based on the segmentation.
    '''
    path = Path(input_folder)
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
    
def single_object_process():
    parser = ConfigParser()
    parser.read("playdoh.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    input_folder = config['Paths']['obj_folder']
    
    mode = "Preprocess projections"
        
    if mode == "Preprocess projections":
        preprocess_proj(input_folder, 30)
    if mode == "Segment":
        reconstruct(input_folder, True)
        segment(input_folder, 3)
        print(input_folder)
        count_materials(input_folder)
    if mode == "Evaluate intensity":
        reconstruct(input_folder, False)
        check_intensity(input_folder)
        
def multiple_objects_process():
    folders = ['/export/scratch2/vladysla/Data/Real/AutomatedFOD/Object14_Scan20W']
    
    for input_folder in folders:
        #reconstruct(input_folder, True)
        #segment(input_folder, 2)
        preprocess_proj(input_folder, 40)
        
if __name__ == "__main__":
    multiple_objects_process()
    
