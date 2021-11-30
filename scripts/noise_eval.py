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
import scipy.signal
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

def evaluate_noise(input_folder, verbose=True):
    path = Path(input_folder)
    proj, flat, dark, geom = data.read_flexray(path, sample = 2, skip = 1)
    proj = np.flip(proj, 0)
    
    num_angles = proj.shape[1]
    x = 200
    y = 400
    real_data = proj[y,:,x]
    avg_line = scipy.signal.medfilt(proj[y,:,x], kernel_size = 11)
    std_data = np.zeros_like(real_data)
    window = 11
    for i in range(window//2, num_angles - window//2):
        std_data[i] = real_data[i-window//2:i+window//2].std()
    std_data = np.power(std_data, 2)
    
    avg_diff = np.zeros_like(avg_line)
    avg_diff[1:] = np.diff(avg_line)
    diff_thr = 10
    select = np.abs(avg_diff) < diff_thr
    
    if verbose:
        plt.plot(range(num_angles), real_data, label = "Real data")
        plt.plot(range(num_angles), avg_line, label = "Median filtered")
        plt.legend()
        plt.savefig("pixel_val.png")
        plt.clf()
        
    if verbose:
        plt.plot(range(num_angles), avg_diff)
        plt.savefig("diff.png")
        plt.clf()
        
    if verbose:
        plt.scatter(avg_line[select], std_data[select])
        plt.savefig("noise_level.png")
        plt.clf()
        
if __name__ == "__main__":
    parser = ConfigParser()
    parser.read("avocado.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    input_folder = config['Paths']['obj_folder']
    
    evaluate_noise(input_folder)
