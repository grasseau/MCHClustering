set -x

rm *.o


OPT="-O3 -g -pg"
OPT="-O0 -g -Wunused-function"
OPT="-O3 -g"
OPT="-O2 -g"
OPT="-O0 -g"

g++ -c -fPIC $OPT -Iinclude -Isrc src/PadsPEM.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/InspectModel.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/ClusterPEM.cxx 
# g++ -c -fPIC $OPT -Iinclude -Isrc src/dataStructure.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathieson.cxx 
g++ -c -fPIC $OPT -Iinclude -Isrc src/poissonEM.cxx 
# g++ -c -fPIC $OPT -Iinclude -Isrc src/padProcessing.cxx
## g++ -c -fPIC $OPT -Iinclude -Isrc src/padProcessing.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathiesonFit.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/mathUtil.cxx
## g++ -c -fPIC $OPT -Iinclude src/gaussianEM.cxx
g++ -c -fPIC $OPT -Iinclude -Isrc src/clusterProcessing.cxx -o clusterProcessing.o 
# g++ -shared -fPIC -Wl,-soname,libExternalC.so PadsPEM.o dataStructure.o mathUtil.o poissonEM.o gaussianEM.o mathieson.o mathiesonFit.o \
#     padProcessing.o clusterProcessing.o \
#    -o libExternalC.so -lgsl -lgslcblas
g++ -shared -fPIC $OPT -Wl,-soname,libExternalC.so PadsPEM.o  mathUtil.o poissonEM.o mathieson.o mathiesonFit.o \
     ClusterPEM.o clusterProcessing.o InspectModel.o \
    -o libExternalC.so -lgsl -lgslcblas

