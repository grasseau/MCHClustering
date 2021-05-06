OPT="-O3 -g"
OPT="-O0 -g"
g++ -c -fPIC $OPT dataStructure.cpp 
g++ -c -fPIC $OPT mathieson.cpp 
g++ -c -fPIC $OPT mathiesonFit.cpp
g++ -c -fPIC $OPT mathUtil.cpp
g++ -c -fPIC $OPT gaussianEM.cpp
g++ -c -fPIC $OPT padProcessing.cpp
g++ -c -fPIC $OPT clusterProcessing-v1.cpp -o clusterProcessing.o 
g++ -shared -fPIC -Wl,-soname,libExternalC.so dataStructure.o mathUtil.o gaussianEM.o mathieson.o mathiesonFit.o \
     padProcessing.o clusterProcessing.o \
    -o libExternalC.so -lgsl -lgslcblas
