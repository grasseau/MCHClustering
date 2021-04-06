cmake_minimum_required(VERSION 3.19)

project(MCHClustering LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(GSL REQUIRED)

add_library(ExternalC SHARED)

target_sources(ExternalC PUBLIC src/C/dataStructure.cpp src/C/mathieson.cpp
src/C/mathUtil.cpp src/C/gaussianEM.cpp src/C/padProcessing.cpp src/C/clusterProcessing.cpp
src/C/mathiesonFit.cpp)

target_link_libraries(ExternalC PUBLIC GSL::gsl)



