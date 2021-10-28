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
import scipy.optimize
import cupy
import cupyx.scipy.ndimage
import skimage
import skimage.segmentation
import skimage.morphology
import skimage.filters
import skimage.measure
import skimage.transform
import skimage.util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

def sphere_fitfunc(p, coords):
    z0, y0, x0, R = p
    z, y, x = coords.T
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

def get_bounding_box(mask):
    z = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    x = np.any(mask, axis=(0, 1))
    
    zmin, zmax = np.where(z)[0][[0,-1]]
    ymin, ymax = np.where(y)[0][[0,-1]]
    xmin, xmax = np.where(x)[0][[0,-1]]
    
    return (zmin, zmax, ymin, ymax, xmin, xmax)

def show3(title, vol) :
    shape = vol.shape

    midZ = (int)(shape[0]/2.)
    midY = (int)(shape[1]/2.)
    midX = (int)(shape[2]/2.)

    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    fig.suptitle(title, fontsize=16)

    ax[0].set_title("YZ")
    ax[0].imshow(vol[:,:,midX], cmap=cm.gray)

    ax[1].set_title("XZ")
    ax[1].imshow(vol[:,midY,:], cmap=cm.gray)

    ax[2].set_title("XY")
    ax[2].imshow(vol[midZ,:,:], cmap=cm.gray)

    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    
def invert_volume(vol):
    return (~(vol.astype(bool))).astype(vol.dtype)

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
    
def segment(input_folder, verbose = True):
    '''Dual-space analysis by Robert van Liere.
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
    vol = skimage.img_as_ubyte(vol)
        
    thr_holder = skimage.filters.threshold_otsu(vol)
    vol = ((skimage.morphology.binary_opening(vol > thr_holder, np.ones((5,5,5), np.uint8)).astype(np.uint8))) * vol
    if verbose:
        print("Holder removed")
        show3("without_holder", vol)
        
    peel0 = skimage.morphology.binary_dilation(skimage.morphology.binary_dilation(vol>0)) * invert_volume(vol)
    labels, nfeatures = scipy.ndimage.label(peel0)
    props = skimage.measure.regionprops(labels)
    if verbose:
        print ("Peel: nfeatures", nfeatures, 'props', len(props))
    propL = []
    for prop in props:
        propL.append ((prop.area, prop))
    propL = sorted (propL, key=lambda r:r[0], reverse=True)
    area, prop = propL[0]
    peel = (labels == prop.label).astype(np.int8)
    if verbose:
        print("Peel segmented")
        show3("peel", peel)
    
    meatpit = vol * invert_volume (peel)
    meatpit = apply_median_filter(meatpit, 3)
    thr_meatpit = skimage.filters.threshold_multiotsu(vol)
    labels, nfeatures = scipy.ndimage.label((meatpit>thr_meatpit[-1]).astype(np.uint8))
    props = skimage.measure.regionprops(labels)
    if verbose:
        print ('Pit: nfeatures0', nfeatures, 'props', len(props))
    propL = []
    for prop in props:
        propL.append ((prop.area, prop))
    propL = sorted (propL, key=lambda r:r[0], reverse=True)
    area, prop = propL[0]
    pit = (labels == prop.label).astype(np.int8)
    num_erosion = 3
    num_dilation = 3
    for i in range(num_erosion):
        pit = skimage.morphology.binary_erosion(pit)
    for i in range(num_dilation):
        pit = skimage.morphology.binary_dilation(pit)
    if verbose:
        print("Pit segmented")
        show3("pit", pit)
        
    pit_boundary = skimage.segmentation.find_boundaries(pit)
    pit_boundary_points = np.nonzero(pit_boundary)
    point_coords = np.zeros((pit_boundary_points[0].shape[0], 3))
    for i in range(3):
        point_coords[:,i] = pit_boundary_points[i]
    error_func = lambda p, x: sphere_fitfunc(p, x) - p[3]
    p0 = np.array([0.,0.,0.,1.])
    print(point_coords.shape)
    p1, cost = scipy.optimize.leastsq(error_func, p0, args=(point_coords,))
    margin = 1.05
    p1[3] *= margin
    sphere_mask = np.zeros_like(pit, dtype=bool)
    sphere_mask[int(p1[0]-p1[3]):int(p1[0]+p1[3]),
                int(p1[1]-p1[3]):int(p1[1]+p1[3]),
                int(p1[2]-p1[3]):int(p1[2]+p1[3])] = True
    pit[~sphere_mask] = 0
    if verbose:
        print("Pit sphere fit:")
        print(p1)
        show3("pit_sphere", pit)
        
    meatpit = (meatpit>thr_meatpit[0]).astype(np.uint8)
    meat = meatpit - pit
    if verbose:
        print("Meat segmented")
        show3("meat", meat)
    
    zmin, zmax, ymin, ymax, xmin, xmax = get_bounding_box(peel)
    peel_subvolume = peel[zmin:zmax, ymin:ymax, xmin:xmax]
    shape_before_downscale = peel_subvolume.shape
    peel_subvolume = skimage.transform.downscale_local_mean(peel_subvolume, (4,4,4))
    peel_subvolume = (peel_subvolume>0).astype(np.uint8)
    if verbose:
        print("Convex hull of peel: ", zmin, zmax, ymin, ymax, xmin, xmax)
        print("Volume size reduction factor = ", float(peel_subvolume.size) / peel.size)
    peel_subvolume_convex = skimage.morphology.convex_hull.convex_hull_image(peel_subvolume)
    peel_subvolume_convex = skimage.transform.resize(peel_subvolume, shape_before_downscale)
    peel_subvolume_convex = (peel_subvolume_convex>0).astype(np.uint8)
    peel_convex = np.zeros_like(peel, dtype=np.uint8)
    peel_convex[zmin:zmax, ymin:ymax, xmin:xmax] = peel_subvolume_convex
    if verbose:
        print("Peel convex hull constructed")
        show3("peel_convex", peel_convex)
    
    background = skimage.segmentation.flood_fill(peel_convex, (1, 1, 1), 1, connectivity=1)
    air_gaps = skimage.util.invert((background+meatpit+peel)>0).astype(np.uint8)
    air_gaps = skimage.morphology.binary_dilation(skimage.morphology.binary_erosion(air_gaps))
    if verbose:
        print("Air gaps segmented")
        show3("air_gaps", air_gaps)
    
    segm_vol[peel == 1] = 1
    segm_vol[meat == 1] = 2
    segm_vol[pit == 1] = 3
    segm_vol[air_gaps == 1] = 4
    
    if verbose:
        print("Segmentation is complete")
        show3("segm", segm_vol)
    
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
    
    mode = "Segment"
    
    if mode == "Preprocess projections":
        preprocess_proj(input_folder, 16)
    if mode == "Segment":
        #reconstruct(input_folder, True)
        segment(input_folder)
    if mode == "Evaluate intensity":
        reconstruct(input_folder, False)
        check_intensity(input_folder)
