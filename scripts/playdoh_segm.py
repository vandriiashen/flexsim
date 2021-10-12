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
import matplotlib.pyplot as plt
from tqdm import tqdm

def reconstruct(input_folder):
    '''Reconstructs the volume using flexbox. Slices will be written to recon/ subfolder.
    '''
    path = Path(input_folder)
    save_path = path / "recon"
    proj, geom = process.process_flex(path, sample = 1, skip = 1)

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
        sl = scipy.ndimage.median_filter(sl, 5)
        
        segm_tmp = np.zeros(im_shape, dtype=np.int8)
        # Playdoh intensity is around 0.04 for 90 kV
        segm_tmp[sl > 0.03] = 1
        # Stone intensity is around 0.08 for 90 kV
        segm_tmp[sl > 0.06] = 2
                
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
        
    playdoh_mean = vol[segm == 1].mean()
    playdoh_std = vol[segm == 1].std()
    stone_mean = vol[segm == 2].mean()
    stone_std = vol[segm == 2].std()
    
    print("Playdoh mean intensity = {}".format(playdoh_mean))
    print("Playdoh intensity std = {}".format(playdoh_std))
    print("Stone mean intensity = {}".format(stone_mean))
    print("Stone intensity std = {}".format(stone_std))
    
def extract_noise_properties(input_folder):
    '''Gets an approximate value of noise properties based on a projection and a flatfield.
    Computes statistic properties of the intensity in a window around every pixel. Then it is assumed that the relation between mean value and std is sigma^2 = alpha mu + sigma_gaussian^2.
    '''
    path = Path(input_folder)
    ff = imageio.imread(path / "io{:06d}.tif".format(0))
    df = imageio.imread(path / "di{:06d}.tif".format(0))
    proj = imageio.imread(path / "scan_{:06d}.tiff".format(0))
    proj -= df
    ff -= df
    log = -np.log(np.divide(proj, ff))
    
    background_mask = log < 0.4
    object_mask = np.logical_not(background_mask)
    
    flatfield_value = ff.mean()
    
    mean_proj = np.zeros_like(proj)
    sigma_proj = np.zeros_like(proj)
    window_size = 9
    hs = window_size // 2
    for i in tqdm(range(400, 450)):
        for j in range(400, 450):
            mean_proj[i,j] = proj[i-hs:i+hs,j-hs:j+hs].mean()
            sigma_proj[i,j] = np.power(proj[i-hs:i+hs,j-hs:j+hs].std(), 2)
    
    #plt.imshow(sigma_proj)
    plt.scatter(mean_proj[400:450,400:450].ravel(), sigma_proj[400:450,400:450].ravel())
    result = scipy.stats.linregress(mean_proj[400:450,400:450].ravel(), sigma_proj[400:450,400:450].ravel())
    print(result.slope)
    print(result.intercept)
    
    plt.show()
    
    print("Flatfield value = {}".format(flatfield_value))
        
if __name__ == "__main__":
    parser = ConfigParser()
    parser.read("playdoh.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    input_folder = config['Paths']['obj_folder']
    
    #reconstruct(input_folder)
    #segment(input_folder)
    #check_intensity(input_folder)
    extract_noise_properties(input_folder)
