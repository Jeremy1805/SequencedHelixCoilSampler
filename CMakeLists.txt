cmake_minimum_required(VERSION 3.16)
project(HelixCoilSampler VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Doxygen QUIET)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

# Test executables - include source files from src directory
add_executable(bernoulli_scan 
    tests/BernoulliScan.cpp
    src/Utilities.cpp
    src/EquilibriumPartitionMapGenerator.cpp
    src/FoldModels.cpp
)

add_executable(random_param_scan 
    tests/RandomParamErrorScan.cpp
    src/Utilities.cpp
    src/EquilibriumPartitionMapGenerator.cpp
    src/FoldModels.cpp
)

# Add include directory so source files can find each other
foreach(target bernoulli_scan random_param_scan)
    target_include_directories(${target} PRIVATE src)
    target_link_libraries(${target} Eigen3::Eigen OpenMP::OpenMP_CXX)
endforeach()

# Enable testing
enable_testing()
add_test(NAME bernoulli_scan_test COMMAND bernoulli_scan)
add_test(NAME random_param_scan_test COMMAND random_param_scan)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Eigen3 version: ${Eigen3_VERSION}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "  OpenMP: REQUIRED but not found!")
endif()
if(DOXYGEN_FOUND)
    message(STATUS "  Doxygen: Found")
else()
    message(STATUS "  Doxygen: Not found")
endif()
message(STATUS "")
