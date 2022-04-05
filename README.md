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
