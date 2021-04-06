# MCHClustering
Clustering and Fitting implemetation
## Directories
- C: C++/C source code and the Python interface
- PyTests: Python scripts to test C++/C Implementation
- Jupyter: Jupyter notes to test and run
- Analyses: perform analysis (like compute residuals)
- Data: MC data and reco Data from AliRoot release and results of python analysis
- Doc: future documentation
- Util: utilities (plot, IO, ...)
- TrashCan: Sandbox of old python scripts
- Macros: Root macros used to extract MC & reco Data

## Install
- cd MCHClustering/src/C
- ./compil.sh

### Install with Conda

If you have [Conda](https://docs.conda.io), just do, for the root of this repo : 

```
conda create
conda activate mch-clustering
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
```

Should have created a `libExternalC` in `build` directory

If you don't have Conda, you should [give it a try](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-in-silent-mode) :wink:

## Play from Jupyter 

Should work out of the box if using the Conda env mch-clustering created above.

```
cd src
jupyter notebook
```

Then pick the `Play with pads` notebook.

## Run the test examples
- adjust PYTHONPATH with xx/yy/MCHClustering/src
- cd MCHClustering/src/PyTests
- python src/PyTests/clusterProcessOnMCData_t.py 
