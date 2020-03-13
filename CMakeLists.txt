# cmake minimum version
cmake_minimum_required(VERSION 2.8)

# Start configuration
project(optmize_contraction CXX)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

# Flags and definitions
set(CMAKE_CXX_FLAGS "-std=c++11")# -fopenmp -fdiagnostics-color")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wshadow -Wextra -Weffc++ -pedantic -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
add_definitions("-DVERSION_MAJOR=${VERSION_MAJOR}")
add_definitions("-DVERSION_MINOR=${VERSION_MINOR}")


# display configuration when cmake is run
message("Build type: ${CMAKE_BUILD_TYPE}")
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message ("Flags = " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_DEBUG})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  message ("Flags = " ${CMAKE_CXX_FLAGS} " " ${CMAKE_CXX_FLAGS_RELEASE})
endif()

# Compile library
set (LIBSRC abstract_tensor.cpp)
#add_library(PEPS_lib ${LIBSRC})
link_libraries(PEPS_lib)

# Linking and include info
#link_libraries(-llapack)
#include_directories(...)
#target_link_libraries(ising ising_lib)

# Create executables
add_executable(main main.cpp)

