'''
This file contains application-specific functions to change the object volume
Every function receives volume as an argument and returns volume after modification
'''
import numpy as np
from voltools import transform
import skimage
            
def replace_material(volume, src_num, dest_num):
    '''Changes material in voxels from source to dest.
        
    :param src_num: ID of material that should be removed
    :type src_num: :class:`int`
    :param dest_num: ID of material that should be used instead
    :type dest_num: :class:`int`
    '''
    volume[volume == src_num] = dest_num
    return volume
        
def replace_material_cluster(volume, src_num, dest_num, num_clusters):
    '''Changes material in voxels from source to dest based on cluster structure.
    '''
    labels, nfeatures = scipy.ndimage.label(self.volume == src_num)
    assert num_clusters <= nfeatures
    for i in range(num_clusters):
        volume[labels == i+1] = dest_num
        
    return volume
        
def remove_material_clusters(volume, src_num, dest_num, num_keep):
    '''Compute clusters of voxels filled with a certain material.
    '''
    labels, nfeatures = scipy.ndimage.label(volume == src_num)
    props = skimage.measure.regionprops(labels)
        
    propL = []
    for prop in props:
        propL.append ((prop.area, prop))
    propL = sorted (propL, key=lambda r:r[0], reverse=True)
        
    for i in range(min(num_keep, len(propL))):
        area, prop = propL[i]
        volume[labels == prop.label] = src_num
    
    return volume
        
def keep_few_clusters(class_sizes, tg_count):
    sorted_sizes = sorted(class_sizes.items(), key = lambda x: x[1], reverse=True)
    cluster_seq = []
        
    max_class_num = 3
    for i in range(max_class_num):
        for k, v in sorted_sizes:
            if v < tg_count:
                cluster_seq.append(k)
                tg_count -= v
                sorted_sizes.remove((k, v))
                break
            
    return cluster_seq

def keep_few_clusters(class_sizes, tg_count):
    sorted_sizes = sorted(class_sizes.items(), key = lambda x: x[1], reverse=True)
    cluster_seq = []
        
    max_class_num = 3
    for i in range(max_class_num):
        for k, v in sorted_sizes:
            if v < tg_count:
                cluster_seq.append(k)
                tg_count -= v
                sorted_sizes.remove((k, v))
                break
            
    return cluster_seq

def drop_few_clusters(class_sizes, tg_count):
    sorted_sizes = sorted(class_sizes.items(), key = lambda x: x[1], reverse=True)
    total_air_count = 0
    for k, v in sorted_sizes:
        total_air_count += v
    drop_air_voxels = total_air_count - tg_count
    drop_seq = []
        
    max_class_num = 3
    for i in range(max_class_num):
        for k, v in sorted_sizes:
            if v < drop_air_voxels:
                drop_seq.append(k)
                drop_air_voxels -= v
                sorted_sizes.remove((k, v))
                print(k, v)
                break
            
    # Function should return clusters to keep, so need to invert drop_seq
    cluster_seq = [k for k in class_sizes.keys() if k not in drop_seq]
            
    return cluster_seq
        
def remove_air_clusters(volume, src_num, dest_num, num_classes, tg_count, pick_func, verbose=True):
    points = np.nonzero(volume == src_num)
    points_num = points[0].shape[0]
    coords = np.zeros((points_num, 3), dtype=np.int32)
    for i in range(3):
        coords[:,i] = points[i]
        
    seq = np.arange(points_num)
    np.random.shuffle(seq)
        
    dist_map = np.zeros((points_num, num_classes), dtype=np.float32)
    for i in range(num_classes):
        dist_map[:,i] = np.power(np.subtract(coords, coords[seq[i],:]), 2).sum(axis=1)
    class_map = np.argmin(dist_map, axis=1)
    class_sizes = {}
    for i in range(num_classes):
            class_sizes[i] = np.count_nonzero(class_map == i)
                
    if verbose:
        print("Sizes of classes:")
        print(class_sizes)
            
    keep_classes = pick_func(class_sizes, tg_count)
    drop_classes = [i for i in range(num_classes) if i not in keep_classes]
        
    if verbose:
        print("Chosen classes:")
        total_size = 0
        for i in keep_classes:
            total_size += class_sizes[i]
            print(i, class_sizes[i])
        print("Target voxel number = {}, generated {}".format(tg_count, total_size))
                    
    for i in drop_classes:
        select = coords[class_map==i,:]
        volume[select[:,0], select[:,1], select[:,2]] = dest_num 
        
    return volume

def affine_volume(volume, mat_num, scale, shear, rotation, translation):
    res_vol = np.zeros_like(volume, dtype=int)
    
    # Direct affine transformation of volume leads to many interpolation artifacts. Thus, we will apply it to every material separately and apply dilation
    for i in range(1, mat_num+1):
        tmp_vol = np.zeros_like(volume, dtype=bool)
        tmp_vol[volume == i] = True
        tmp_res = transform(tmp_vol, interpolation='linear', device='cpu', 
                            scale=scale, shear=shear, translation=translation, rotation=rotation, rotation_units='deg', rotation_order='rzxz')
        tmp_res = tmp_res.astype(bool)
        tmp_res = skimage.morphology.binary_dilation(tmp_res)
        res_vol[tmp_res] = i
            
    return res_vol
