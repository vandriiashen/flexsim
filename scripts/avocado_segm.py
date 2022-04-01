import numpy as np
from pathlib import Path
from configparser import ConfigParser
import shutil
from tqdm import tqdm
import imageio
import cupy
import cupyx.scipy.ndimage
import skimage
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flexdata import display, data
from flextomo import projector
from flexcalc import process, analyze

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
    plt.savefig("./img/{}.png".format(title))
    
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

def reconstruct(input_folder, output_folder, bh_correction = True, compound = 'H2O', density = 0.6):
    '''Reconstructs the volume using flexbox. Slices will be written to recon/ subfolder.
    '''
    path = output_folder
    if bh_correction == True:
        save_path = path / "recon_bh"
    else:
        save_path = path / "recon"
    proj, geom = process.process_flex(input_folder, sample = 4, skip = 1, correct='cwi-flexray-2019-04-24')

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
    
def segment(input_folder, output_folder, verbose = True):
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
    # A small number of outliers is present in one sample
    vol[vol>1.] = 1.
    vol = skimage.img_as_ubyte(vol)
        
    thr_holder = skimage.filters.threshold_otsu(vol)
    vol = ((skimage.morphology.binary_opening(vol > thr_holder, np.ones((5,5,5), np.uint8)).astype(np.uint8))) * vol
    
    if verbose:
        print("Holder removed")
        show3("without_holder", vol)
    
    shape_before_downscale = vol.shape
    vol_downscaled = skimage.transform.downscale_local_mean(vol, (4,4,4))
    vol_downscaled = (vol_downscaled>0).astype(np.uint8)
    vol_downscaled_convex = skimage.morphology.convex_hull.convex_hull_image(vol_downscaled)
    vol_convex = skimage.transform.resize(vol_downscaled_convex, shape_before_downscale)
    vol_convex = (vol_convex>0).astype(np.uint8)
        
    #vol_convex should not include the boundary of the fruit, but needs to cover the air gaps inside
    num_erosion = 10
    for i in range(num_erosion):
        vol_convex = skimage.morphology.binary_erosion(vol_convex)
    #remove the top of the fruit to avoid problems with connection to branch
    #top_cut = 30
    top_cut = 5
    vol_bbox = get_bounding_box(vol_convex)
    print(vol_bbox[0])
    vol_convex[vol_bbox[0]:vol_bbox[0]+top_cut,:,:] = 0
    
    if verbose:
        print("Volume convex hull")
        show3("vol_convex", vol_convex)

    peel0 = skimage.morphology.binary_dilation(skimage.morphology.binary_dilation(vol>0)) * invert_volume(vol)
    peel0[vol_convex > 0] = 0
        
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
    
    meatpit = vol * invert_volume(peel)
    meatpit = apply_median_filter(meatpit, 3)
    
    plt.clf()
    plt.hist(meatpit[meatpit > 0].ravel(), bins=256)
    plt.savefig("./img/meatpit_hist.png")
    plt.clf()

    thr_meatpit = skimage.filters.threshold_multiotsu(meatpit[meatpit > 0])
    if verbose:
        print("Multiotsu thresholds = {}".format(thr_meatpit))
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

    meatpit = (meatpit>thr_meatpit[0]).astype(np.uint8)
    meat = meatpit - pit
    if verbose:
        print("Meat segmented")
        show3("meat", meat)

    background = (vol_convex == 0).astype(np.uint8)

    if verbose:
        print("Background constructed")
        show3("background", background)
    air_gaps = skimage.util.invert((background+meatpit+peel)>0).astype(np.uint8)

    if verbose:
        print("Air gaps segmented")
        show3("air_gaps", air_gaps)

    segm_vol[peel == 1] = 1
    segm_vol[meat == 1] = 2
    segm_vol[pit == 1] = 3
    segm_vol[air_gaps == 1] = 4

    if verbose:
        print("Segmentation is complete")
        show3("segm_extra_{}".format(path.name), segm_vol)
    
    for i in range(height):
        imageio.imwrite(segm_path / "slice_{:06d}.tiff".format(i), segm_vol[i,:])
        
def count_materials(input_folder, output_folder):
    '''Computes the number of voxels filled with different materials
    '''
    path = output_folder
    segm_path = path / "segm"
    
    height = len(list(segm_path.glob("*.tiff")))
    im_shape = (imageio.imread(segm_path / "slice_{:06d}.tiff".format(0))).shape
    segm = np.zeros((height, *im_shape), dtype=np.int16)
    for i in range(height):
        segm[i,:] = imageio.imread(segm_path / "slice_{:06d}.tiff".format(i))
        
    peel_count = np.count_nonzero(segm == 1)
    avocado_count = np.count_nonzero(segm == 2)
    seed_count = np.count_nonzero(segm == 3)
    air_count = np.count_nonzero(segm == 4)
    seed_com = scipy.ndimage.center_of_mass(segm == 3)
    
    #print("Peel,Avocado,Seed,Air,Seed_COM")
    print("{},{},{},{},{}".format(path.name, peel_count, avocado_count, seed_count, air_count))
        
def check_intensity(output_folder):
    '''Computes mean value and standard deviations for materials based on the segmentation.
    '''
    path = output_folder
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
    peel_count = np.count_nonzero(segm == 1)
    avocado_mean = vol[segm == 2].mean()
    avocado_std = vol[segm == 2].std()
    avocado_count = np.count_nonzero(segm == 2)
    seed_mean = vol[segm == 3].mean()
    seed_std = vol[segm == 3].std()
    seed_count = np.count_nonzero(segm == 3)
    air_mean = vol[segm == 4].mean()
    air_std = vol[segm == 4].std()
    air_count = np.count_nonzero(segm == 4)
    
    print("Peel intensity =         {:0.4f} +- {:0.4f} ({} voxels)".format(peel_mean, peel_std, peel_count))
    print("Avocado intensity =      {:0.4f} +- {:0.4f} ({} voxels)".format(avocado_mean, avocado_std, avocado_count))
    print("Seed intensity =         {:0.4f} +- {:0.4f} ({} voxels)".format(seed_mean, seed_std, seed_count))
    print("Air gap intensity =      {:0.4f} +- {:0.4f} ({} voxels)".format(air_mean, air_std, air_count))
    
def preprocess_proj(input_folder, output_folder, skip_proj):
    '''Applies darkfield- and flatfield-correction to projections and saves them to a separate folder.
    '''
    path = input_folder
    out = output_folder
    log_path = out / "log"
    log_path.mkdir(exist_ok=True)
    
    proj, flat, dark, geom = data.read_flexray(path, sample = 4, skip = 1, correct='cwi-flexray-2019-04-24')
    proj = process.preprocess(proj, flat, dark)
    proj = np.flip(proj, 0)
    
    for i in range(0,proj.shape[1],skip_proj):
        imageio.imwrite(log_path / "scan_{:06d}.tiff".format(i), proj[:,i,:])    
        
def multiple_objects_process():
    input_root = Path('/export/scratch2/vladysla/Data/Real/Avocado_extra/')
    #input_root = Path('/export/scratch2/vladysla/Data/Real/AvocadoSet/')
    #input_root = Path('/export/scratch2/vladysla/Data/Real/AvocadoScans/')
    output_root = Path('/export/scratch2/vladysla/Data/Generation/Avocado/Training')
    
    sub_folders = []
    
    fruit_numbers = list(range(1, 13))
    for num in fruit_numbers:
        sub_folders.append('s{:02d}_d01'.format(num))
        sub_folders.append('s{:02d}_d06'.format(num))
        sub_folders.append('s{:02d}_d09'.format(num))
    
    sub_folders.extend(['s13_d01', 's13_d07', 's13_d09', 's14_d01', 's14_d07', 's14_d09', 's15_d01'])
    remove_obj = ['s01_d01', 's03_d06', 's03_d09', 's06_d01', 's08_d01', 's09_d06', 's10_d06']
    for obj in remove_obj:
        sub_folders.remove(obj)
    
    print(sub_folders)
    print(len(sub_folders))
    input_folders = [input_root / sub_folder for sub_folder in sub_folders]
    output_folders = [output_root / sub_folder for sub_folder in sub_folders]
    assert len(input_folders) == len(output_folders)
    
    for i in range(len(input_folders)):
        print(input_folders[i])
        output_folders[i].mkdir(exist_ok=True)
        
        preprocess_proj(input_folders[i], output_folders[i], 2)
        shutil.copy(input_folders[i] / 'scan settings.txt', output_folders[i])
        reconstruct(input_folders[i], output_folders[i], bh_correction=True, compound='H2O', density=0.6)
        segment(input_folders[i], output_folders[i])
        
        # Check average attenuation for every label
        #reconstruct(input_folders[i], output_folders[i], bh_correction=False)
        #check_intensity(output_folders[i])
        
if __name__ == "__main__":
    multiple_objects_process()
