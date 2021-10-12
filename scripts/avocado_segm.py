import numpy as np
from pathlib import Path
from configparser import ConfigParser
import imageio
from flexdata import display
from flexdata import data
from flextomo import projector
from flexcalc import process
import scipy.ndimage
import scipy.stats
import skimage.segmentation
import skimage.morphology
import skimage.filters
import skimage.util
import matplotlib.pyplot as plt
from tqdm import tqdm

def reconstruct(input_folder):
    '''Reconstructs the volume using flexbox. Slices will be written to recon/ subfolder.
    '''
    path = Path(input_folder)
    save_path = path / "recon"
    proj, geom = process.process_flex(path, sample = 2, skip = 1)

    save_path.mkdir(exist_ok=True)

    geom.parameters['src_ort'] += (7-5.5) #see below for flexbox hard coded values
    geom['det_roll'] -= 0.25

    vol = projector.init_volume(proj)
    projector.FDK(proj, vol, geom)
    data.write_stack(save_path, 'slice', vol, dim = 0)
    
def segment(input_folder):
    '''Segments every slice with thresholding. Segmented slices will be written to segm/ subfolder.
    '''
    path = Path(input_folder)
    recon_path = path / "recon"
    segm_path = path / "segm"
    segm_path.mkdir(exist_ok=True)
    
    height = len(list(recon_path.glob("*.tiff")))
    im_shape = (imageio.imread(recon_path / "slice_{:06d}.tiff".format(0))).shape
    
    for i in tqdm(range(height)):
        
        sl = imageio.imread(recon_path / "slice_{:06d}.tiff".format(i))
        sl = skimage.filters.median(sl,skimage.morphology.square(5))
        
        # ===================
        # Dual space analysis by Robert van Liere
        # ===================
        
        #thr = skimage.filters.threshold_otsu(sl)
        # We will not use otsu to avoid mistakes in the bottom and top parts of the avocado
        thr = 0.015
        obj = (sl > thr).astype(np.uint8)
        
        boundary = skimage.segmentation.find_boundaries(obj).astype(np.uint8)
        background = skimage.segmentation.flood_fill(boundary, (1, 1), 1)
        air_gaps =  skimage.util.invert((background+obj)>0).astype(np.uint8)
        
        segm_tmp = np.zeros_like(sl, dtype=np.int8)
        segm_tmp[obj == 1] = 1
        segm_tmp[air_gaps == 1] = 2
                
        imageio.imwrite(segm_path / "slice_{:06d}.tiff".format(i), segm_tmp)
        
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
        
    avocado_mean = vol[segm == 1].mean()
    avocado_std = vol[segm == 1].std()
    air_mean = vol[segm == 2].mean()
    air_std = vol[segm == 2].std()
    
    print("Avocado mean intensity = {} (should be multiplied by 2 to account for binning)".format(avocado_mean))
    print("Avocado intensity std = {}".format(avocado_std))
    print("Air gap mean intensity = {}".format(air_mean))
    print("Air gap intensity std = {}".format(air_std))
    
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
    
    if True:
        preprocess_proj(input_folder, 16)
        reconstruct(input_folder)
        segment(input_folder)
        check_intensity(input_folder)
    else:
        count_air(input_folder, [0, 8, 16, 24, 32, 40])
