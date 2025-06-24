## Summary

• **Added comprehensive build system** - Introduced CMake configuration for cross-platform compilation with dependency management for Eigen3 and OpenMP
• **Enhanced scientific applications** - Added BernoulliScan.cpp for systematic parameter scanning analysis of helix-coil transition models  
• **Improved code documentation** - Added extensive inline documentation and scientific context to existing applications
• **Modernized codebase structure** - Enhanced CustomMatrix.cpp with detailed documentation and scientific explanations

## Changes Made

### New Files
- `CMakeLists.txt` - Complete build system with modern CMake practices, dependency management, and installation rules
- `BernoulliScan.cpp` - Parameter scanning application for analyzing Bernoulli sequence distributions vs equilibrium

### Enhanced Files  
- `ArcSample.cpp` - Added comprehensive documentation
- `CustomMatrix.cpp` - Added comprehensive documentation
- `README.md` - Added comprehensive documentation
-  FoldModels.cpp - Added comprehensive documentation
-  EquilibriumPartitionMapGenerator.cpp - Added comprehensive documentation

### Removed Files
- `NearEquilibriumScan.cpp
- `RandomParamErrorScan.cpp

## Technical Details

**Build System**: CMake 3.12+ with automatic dependency detection, cross-platform compiler flags, and optional OpenMP support

**Scientific Applications**: New Bernoulli scanning enables systematic analysis of information-theoretic quantities (KL divergences, entropies) across energy and probability parameter spaces

**Documentation**: Added extensive Doxygen-style comments explaining the statistical mechanics background and mathematical formulations

## Test Plan

- [ ] Verify CMake build system works on Linux/macOS/Windows
- [ ] Test BernoulliScan executable generates expected TSV output
- [ ] Confirm ArcSample runs without errors and produces test.tsv
- [ ] Validate OpenMP integration improves performance on multi-core systems
- [ ] Check that all existing functionality remains intact after refactoring
