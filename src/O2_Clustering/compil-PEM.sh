OPT="-O2 -g"
OPT="-O3 -g"
OPT="-O0 -g -Wunused-variable"
OPT="-O0 -g -Wunused-function"
g++ -c -fPIC $OPT -Iinclude -Isrc src/PadsPEM.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/dataStructure.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathieson.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/poissonEM.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/padProcessing.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathiesonFit.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathUtil.cxx
g++ -c -fPIC $OPT -Iinclude src/gaussianEM.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/clusterProcessing.cxx -o clusterProcessing.o 
g++ -shared -fPIC -Wl,-soname,libExternalC.so PadsPEM.o dataStructure.o mathUtil.o poissonEM.o gaussianEM.o mathieson.o mathiesonFit.o \
     padProcessing.o clusterProcessing.o \
    -o libExternalC.so -lgsl -lgslcblas
