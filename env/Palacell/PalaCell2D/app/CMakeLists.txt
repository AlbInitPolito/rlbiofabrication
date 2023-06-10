cmake_minimum_required(VERSION 2.8.12)

project(palaCell2D)
enable_language(CXX)
set (CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_NAME "palaCell")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    #set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
endif()
message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
message(${CMAKE_BUILD_TYPE})

# Compiler flags
# Append flags: set(CMAKE_XXX_FLAGS "${CMAKE_XXX_FLAGS} ...")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    message("GCC.")
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wnon-virtual-dtor -Wno-deprecated")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DPLB_DEBUG -O0")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message("Clang.")
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wnon-virtual-dtor")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DPLB_DEBUG -O0")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    message("MSVC.")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /Ot /GS- /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/DPLB_DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message( FATAL_ERROR "CXX compiler not recognized. CMake will quit." )
endif()

option(ENABLE_MPI "Enable MPI" ON)
if(ENABLE_MPI)
    message("Enabling MPI")
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        #set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
        include_directories(${MPI_CXX_INCLUDE_PATH})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
        add_definitions(-DPLB_MPI_PARALLEL)
    endif()
endif()

if(WIN32)
    option(ENABLE_POSIX "Enable POSIX" OFF)
else()
    option(ENABLE_POSIX "Enable POSIX" ON)
endif()

if(ENABLE_POSIX)
    message("Enabling POSIX")
    add_definitions(-DPLB_USE_POSIX)
endif()

if(APPLE)
    add_definitions(-DPLB_MAC_OS_X)
endif()

if(WIN32 OR CYGWIN)
    add_definitions(-DPLB_WINDOWS)
endif()

###############################################################################
# Palabos Library
###############################################################################

include_directories("../external/palabos/src")
include_directories("../external/palabos/externalLibraries")
include_directories("../external/palabos/externalLibraries/Eigen3")

find_library(PALABOS_LIB palabos ../external/palabos/build)
if(NOT PALABOS_LIB)
    message("Palabos library not found. Build again from source.")
    file(GLOB_RECURSE PALABOS_SRC "../external/palabos/src/*.cpp")
    file(GLOB_RECURSE EXT_SRC "../external/palabos/externalLibraries/tinyxml/*.cpp")
    add_library(palabos STATIC ${PALABOS_SRC} ${EXT_SRC})
    set(PALABOS_LIB palabos)
endif()

###############################################################################
# tinyxml2 Library
###############################################################################

include_directories("../external/tinyxml2")

find_library(TINYXML2_LIB tinyxml2 ../external/tinyxml2/build)
if(NOT TINYXML2_LIB)
    message("Tinyxml2 library not found. Build again from source.")
    file(GLOB_RECURSE TINYXML2_SRC "../external/tinyxml2/*.cpp")
    add_library(tinyxml2 STATIC ${TINYXML2_SRC})
    set(TINYXML2_LIB tinyxml2)
endif()

###############################################################################

add_executable(${EXECUTABLE_NAME} "./${EXECUTABLE_NAME}.cpp")
target_link_libraries(${EXECUTABLE_NAME} PUBLIC stdc++fs)

# Link with the following libraries
target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${PALABOS_LIB})
target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${TINYXML2_LIB})
if(ENABLE_MPI)
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${MPI_CXX_LIBRARIES})
endif()
