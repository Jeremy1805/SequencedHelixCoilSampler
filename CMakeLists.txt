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
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MUPARSER REQUIRED muparser)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

# Main executable
add_executable(main
    src/main.cpp
    src/Utilities.cpp
    src/EquilibriumPartitionMapGenerator.cpp
    src/FoldModels.cpp
    src/ConfigurableScanner.cpp
)

# Add include directory and link libraries
target_include_directories(main PRIVATE include ${MUPARSER_INCLUDE_DIRS})
target_link_libraries(main
    Eigen3::Eigen 
    OpenMP::OpenMP_CXX 
    nlohmann_json::nlohmann_json
    ${MUPARSER_LIBRARIES}
)
target_compile_options(main PRIVATE ${MUPARSER_CFLAGS_OTHER})

# Enable testing
enable_testing()

# Add tests that run main with different config files
add_test(NAME bernoulli_test 
    COMMAND main test/config/bernoulli_test.json
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_test(NAME error_test 
    COMMAND main test/config/error_test.json
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

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
