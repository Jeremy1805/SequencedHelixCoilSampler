cmake_minimum_required(VERSION 3.12)
project(HelixCoilSampler 
    VERSION 1.0.0 
    DESCRIPTION "Statistical Mechanics Simulation for Protein Helix-Coil Transitions"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Wall /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find OpenMP (optional but recommended)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel computing enabled")
else()
    message(WARNING "OpenMP not found - parallel computing disabled")
endif()

# Create library with core functionality
add_library(helixcoil_core INTERFACE)
target_include_directories(helixcoil_core INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)

# Link Eigen3
target_link_libraries(helixcoil_core INTERFACE Eigen3::Eigen)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(helixcoil_core INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(helixcoil_core INTERFACE OPENMP_ENABLED)
endif()

# Bernoulli scan executable
add_executable(bernoulli_scan BernoulliScan.cpp)
target_link_libraries(bernoulli_scan PRIVATE helixcoil_core)
set_target_properties(bernoulli_scan PROPERTIES
    OUTPUT_NAME "bernoulli_scan"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Arc sample executable  
add_executable(arc_sample ArcSample.cpp)
target_link_libraries(arc_sample PRIVATE helixcoil_core)
set_target_properties(arc_sample PROPERTIES
    OUTPUT_NAME "arc_sample"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Example executable (if you want to add a simple example)
# add_executable(example Example.cpp)
# target_link_libraries(example PRIVATE helixcoil_core)

# Installation rules
include(GNUInstallDirs)

# Install executables
install(TARGETS bernoulli_scan arc_sample
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install headers
install(FILES 
    FoldModels.cpp
    EquilibriumPartitionMapGenerator.cpp
    Utilities.cpp
    CustomMatrix.cpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/helixcoil)

# Install library
install(TARGETS helixcoil_core
    EXPORT HelixCoilSamplerTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export configuration
install(EXPORT HelixCoilSamplerTargets
    FILE HelixCoilSamplerTargets.cmake
    NAMESPACE HelixCoilSampler::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HelixCoilSampler)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HelixCoilSamplerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HelixCoilSamplerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HelixCoilSamplerConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HelixCoilSampler)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HelixCoilSamplerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HelixCoilSamplerConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HelixCoilSampler)

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Optional: Enable testing
option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    # add_subdirectory(tests)
endif()

# Optional: Documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    option(BUILD_DOCUMENTATION "Build documentation" OFF)
    if(BUILD_DOCUMENTATION)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Eigen3 version: ${Eigen3_VERSION}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "  OpenMP: Not found")
endif()
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")