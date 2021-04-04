cmake_minimum_required(VERSION 3.19)

project(MCHClustering LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(GSL CONFIG REQUIRED)

add_library(ExternalC SHARED)

target_sources(ExternalC PUBLIC C/dataStructure.cpp C/mathieson.cpp
C/mathUtil.cpp C/gaussianEM.cpp C/padProcessing.cpp C/clusterProcessing.cpp
C/mathiesonFit.cpp)

target_link_libraries(ExternalC PUBLIC gsl::libgsl)



