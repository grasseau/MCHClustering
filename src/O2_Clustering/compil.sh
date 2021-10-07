OPT="-O0 -g"
OPT="-O2 -g"
OPT="-O3 -g"
g++ -c -fPIC $OPT -Iinclude src/dataStructure.cxx 
g++ -c -fPIC $OPT -Iinclude src/mathieson.cxx 
g++ -c -fPIC $OPT -Iinclude src/mathiesonFit.cxx
g++ -c -fPIC $OPT -Iinclude src/mathUtil.cxx
g++ -c -fPIC $OPT -Iinclude src/gaussianEM.cxx
g++ -c -fPIC $OPT -Iinclude src/padProcessing.cxx
#g++ -c -fPIC $OPT clusterProcessing-v1.cpp -o clusterProcessing.o 
g++ -c -fPIC $OPT -Iinclude src/clusterProcessing.cxx -o clusterProcessing.o 
g++ -shared -fPIC -Wl,-soname,libExternalC.so dataStructure.o mathUtil.o gaussianEM.o mathieson.o mathiesonFit.o \
     padProcessing.o clusterProcessing.o \
    -o libExternalC.so -lgsl -lgslcblas
