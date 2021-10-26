import numpy as np
from pathlib import Path
from configparser import ConfigParser
import imageio
from flexdata import display
from flexdata import data
from flextomo import projector
from flexcalc import process
from flexcalc import analyze
import scipy.ndimage
import scipy.stats
import cupy
import cupyx.scipy.ndimage
import skimage.segmentation
import skimage.morphology
import skimage.filters
import skimage.util
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
        density = 0.4
        energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound = 'H2O', density = density)   
        proj_cor = process.equivalent_density(proj, geom, energy, spec, compound = 'H2O', density = density, preview=False)
        vol_rec = np.zeros_like(vol)
        projector.FDK(proj_cor, vol_rec, geom)
        vol = vol_rec
    
    data.write_stack(save_path, 'slice', vol, dim = 0)
    
def segment(input_folder):
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
    
    vol[:,100:700,200:850] = apply_median_filter(vol[:,100:700,200:850], 8)
    print("Meat/Background segmentation")
    thr_avocado = skimage.filters.threshold_otsu(vol)
    print("Avocado meat threshold = {}".format(thr_avocado))
    obj = (vol > thr_avocado).astype(np.uint8)
    
    print("Dual space segmentation")
    boundary = skimage.segmentation.find_boundaries(obj).astype(np.uint8)
    background = skimage.segmentation.flood_fill(boundary, (1, 1, 1), 1, connectivity=1)
    air_gaps =  skimage.util.invert((background+obj)>0).astype(np.uint8)
    
    print("Seed segmentation")
    thr_seed = 0.4
    seed_mask = (vol > thr_seed).astype(np.uint8)
    print("Avocado meat threshold = {}".format(thr_seed))
    erosion_num = 3
    dilation_num = 4
    for i in range(erosion_num):
        seed_mask = skimage.morphology.erosion(seed_mask)
    for i in range(dilation_num):
        seed_mask = skimage.morphology.dilation(seed_mask)
    
    # avocado meat
    segm_vol[obj == 1] = 2
    # avocado peel
    segm_vol[boundary == 1] = 1
    # avocado seed
    segm_vol[seed_mask == 1] = 3
    # air gaps
    segm_vol[air_gaps == 1] = 4
    
    for i in range(height):
        imageio.imwrite(segm_path / "slice_{:06d}.tiff".format(i), segm_vol[i,:])
        
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
        
    peel_mean = vol[segm == 1].mean()
    peel_std = vol[segm == 1].std()
    avocado_mean = vol[segm == 2].mean()
    avocado_std = vol[segm == 2].std()
    seed_mean = vol[segm == 3].mean()
    seed_std = vol[segm == 3].std()
    air_mean = vol[segm == 4].mean()
    air_std = vol[segm == 4].std()
    
    print("Peel intensity = {} +- {} (should be multiplied by 2 to account for binning)".format(peel_mean, peel_std))
    print("Avocado intensity = {} +- {}".format(avocado_mean, avocado_std))
    print("Seed intensity = {} +- {}".format(seed_mean, seed_std))
    print("Air gap intensity = {} +- {}".format(air_mean, air_std))
    
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
        
def count_air(input_folder, proj_numbers):
    '''Counts air voxels in the segmented volume and air pixels on projection specified by proj_numbers
    '''
    path = Path(input_folder)
    proj_path = path / "gt"
    slice_path = path / "segm"
    
    slice_air = 0
    slice_meat = 0
    height = len(list(slice_path.glob("*.tiff")))
    for i in range(height):
        sl = imageio.imread(slice_path / "slice_{:06d}.tiff".format(i))
        unique, counts = np.unique(sl, return_counts=True)
        val_dict = dict(zip(unique, counts))
        if 1 in unique:
            slice_meat += val_dict[1]
        if 2 in unique:
            slice_air += val_dict[2]
    slice_air /= (slice_meat+slice_air)
    
    slice_proj = np.zeros((len(proj_numbers)), dtype=np.float32)
    meat_proj = np.zeros((len(proj_numbers)), dtype=np.float32)
    for i in range(len(proj_numbers)):
        proj = imageio.imread(proj_path / "{:06d}.tiff".format(proj_numbers[i]))
        unique, counts = np.unique(proj, return_counts=True)
        val_dict = dict(zip(unique, counts))
        if 2 in unique:
            slice_proj[i] = val_dict[2]
            meat_proj[i] = val_dict[1]
            slice_proj[i] /= (meat_proj[i]+slice_proj[i])
        
    print("Volume,Proj_{},Proj_{},Proj_{},Proj_{},Proj_{},Proj_{}".format(*proj_numbers))
    print("{},{},{},{},{},{},{}".format(slice_air, *slice_proj))
        
if __name__ == "__main__":
    parser = ConfigParser()
    parser.read("avocado.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    input_folder = config['Paths']['obj_folder']
    
    #preprocess_proj(input_folder, 16)
    reconstruct(input_folder, True)
    segment(input_folder)
    #check_intensity(input_folder)
