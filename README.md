# MCHClustering
Clustering and Fitting implemetation

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

## Run the test examples
- adjust PYTHONPATH with xx/yy/MCHClustering/src
- cd MCHClustering/src/PyTests
- python src/PyTests/clusterProcessOnMCData_t.py 
