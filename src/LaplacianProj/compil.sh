OPT="-O3 -g"
OPT="-O1 -g"
g++ -c -fPIC $OPT dataStructure.cxx
g++ -c -fPIC $OPT mathieson.cxx
g++ -c -fPIC $OPT mathiesonFit.cxx
g++ -c -fPIC $OPT mathUtil.cxx
g++ -c -fPIC $OPT gaussianEM.cxx
g++ -c -fPIC $OPT padProcessing.cxx
#g++ -c -fPIC $OPT clusterProcessing-v1.cpp -o clusterProcessing.o 
g++ -c -fPIC $OPT clusterProcessing.cxx -o clusterProcessing.o 
g++ -shared -fPIC -Wl,-soname,libExternalC.so dataStructure.o mathUtil.o gaussianEM.o mathieson.o mathiesonFit.o \
     padProcessing.o clusterProcessing.o \
    -o libExternalC.so -lgsl -lgslcblas
