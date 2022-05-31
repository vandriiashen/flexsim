# flexsim
This package can be used to reconstruct scanned objects, change the internal structure and reproject the augmented volume. The goal is to generate new X-ray projections similar but not identical to the acquired data
# Installation
CT reconstruction and forward projection uses flexbox that has to be installed from sources
```
conda create -n flexbox -c astra-toolbox -c conda-forge -c defaults numpy pyqtgraph matplotlib tqdm imageio psutil toml scipy scikit-image simpleitk xraylib networkx pygraphviz numpy-stl astra-toolbox cupy
conda activate flexbox
pip install git+https://github.com/cicwi/flexDATA
pip install git+https://github.com/cicwi/flexTOMO
pip install git+https://github.com/cicwi/flexCALC
pip install voltools
pip install -e .
```
# Usage
scripts/ subfolder contains code for processing of raw data and generation of simulated projections.

avocado_segm.py and playdoh_segm.py were used to generate binned logarithmed projections, reconstruct object volumes and segment them. Beam hardening correction is performed during reconstruction, material parameters are chosen manually. Without beam hardening correction, otsu segmentation is not working sufficiently well (labels boundary of playdoh as stone and poorly segments pit in avocado). It is possible to run reconstruction without bh correction and use *check_intensity* function to estimate attenuation intensity for all materials.

avocado_gen.py and playdoh_gen.py are used for data generation. The scripts contain code implementations for multiple tasks:
1. ***batch_replication***

Generate simulated projections for multiple segmented objects without changing anything in its inner structure. This function can be used to replace acquired projections with simulated images and check how it affects machine learning algorithm.

2. ***batch_basic_augmentation***

Generate simulated projections after basic modifications of the segmented object volume. In case of playdoh, voxels labeled as stone are replaced with playdoh. Thus, a sample with two pieces of stone can be transformed into a sample with one and zero foreign objects. For avocado, 3D distribution of air voxels is split into a number of clusters. Clusters are selected using nearest neighbour approach, cluster centers are chose randomly among air voxels. Then we drop a certain number of clusters to achieve the desired ratio of air.

3. ***avocado_pair_generation*** and ***playdoh_triple_generation***

Generate artificial objects based on a single base sample. Affine transformations are used to modify the main object. Then a foreign object is modified separately similarly to ***batch_basic_augmentation***.

