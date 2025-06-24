# HelixCoilSampler

**Advanced Statistical Mechanics Simulation for Protein Helix-Coil Transitions**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.0%2B-green.svg)](https://eigen.tuxfamily.org/)

## Overview

HelixCoilSampler is a high-performance C++ implementation for analyzing protein secondary structure transitions using statistical mechanical models. The software provides exact solutions for helix-coil equilibrium dynamics through transfer matrix methods and enables comprehensive analysis of sequence-dependent folding patterns.

### Key Features

- **Transfer Matrix Methods**: Exact partition function calculations for small to medium-sized systems
- **Novel Matrix Fraction Arithmetic**: Original method for calculations of marginalized statistics in quenched ising chains.
- **Monte Carlo Sampling**: Efficient sampling of sequence-structure ensembles
- **Information Theory Analysis**: KL divergence, mutual information, and entropy calculations
- **Parallel Computing**: OpenMP optimization for large-scale computations
- **Flexible Sequence Generation**: Bernoulli, template-based, and custom distributions
- **Multiple Model Types**: Ising models with 2-state and 3-fold variants

## Scientific Background

This implementation is based on the statistical mechanical treatment of protein folding, particularly the helix-coil transition theory developed by Zimm-Bragg and Lifson-Roig. The software uses:

- **Transfer matrix formalism** for exact equilibrium calculations
- **Partition function methods** for thermodynamic property evaluation  
- **Information theoretic measures** for analyzing sequence-structure relationships
- **Quenched disorder models** for sequence-dependent folding analysis
- **Novel Matrix Fraction Methods** for exact entropy estimation in long polymer systems (original research)

## Installation

### Dependencies

- **C++17 compatible compiler** (GCC 7+, Clang 6+, MSVC 2017+)
- **Eigen3** (3.3+) - Linear algebra library
- **OpenMP** (optional) - Parallel computing support
- **CMake** (3.12+) - Build system

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libomp-dev
```

### macOS
```bash
brew install cmake eigen libomp
```

### Windows (vcpkg)
```bash
vcpkg install eigen3 openmp
```

### Building from Source

```bash
git clone https://github.com/Jeremy1805/HelixCoilSampler.git
cd HelixCoilSampler
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Example

```cpp
#include "FoldModels.cpp"

int main() {
    // Create Ising model with specified parameters
    Ising2 model(2.0, 2.0, 1.0, 1.0, 1.0, 10);  // w00, w11, c0, c1, v, length
    
    // Calculate equilibrium properties
    model.CalcAllPartition();
    model.CalcAllEigen();
    
    // Generate equilibrium distribution
    model.GetEquilibrumTable();
    
    // Analyze with Bernoulli sequence distribution
    auto bernoulli_map = IsingVar::GenerateBernoulliMap(0.5, 10);
    auto results = model.Results(bernoulli_map);
    
    // Print results
    model.printAll();
    
    return 0;
}
```

### Parameter Scanning

```cpp
// Example: Scan over energy parameters
for (double epsilon = 0.0; epsilon < 4.0; epsilon += 0.1) {
    double w_helix = exp(epsilon);
    Ising2 model(w_helix, w_helix, 1.0, 1.0, 1.0, length);
    
    model.GetEquilibrumTable();
    auto results = model.Results(sequence_distribution);
    
    // Extract thermodynamic quantities
    double kl_divergence = std::get<0>(results);
    double avg_helicity = std::get<4>(results);
    // ... process results
}
```

## Core Components

### FoldModels.cpp
- **GFold**: Base class for general folding models
- **IsingVar**: Ising model variants with sequence mapping
- **Ising2**: Two-state helix-coil model implementation
- **Ising2S3F**: Extended model with three fold states

### EquilibriumPartitionMapGenerator.cpp
- High-performance partition function enumeration
- Optimized recursive generation of state sequences
- Memory-efficient state space exploration

### CustomMatrix.cpp
- Template-based matrix operations with advanced slicing
- **Novel MatrixFraction and VectorFraction classes** for exact calculations of marginalized probability
- **Original entropy estimation methods** for long polymer systems
- Automatic consolidation of rational expressions for numerical stability

### Utilities.cpp
- File I/O operations (TSV export)
- Matrix manipulation utilities
- Statistical sampling functions
- Performance timing utilities

## Model Parameters

### Ising2 Model
- **w00, w11**: Helix-helix interaction energies
- **c0, c1**: Coil state energies  
- **v**: Boundary/end effects parameter
- **L**: Polymer chain length

### Physical Interpretation
- Energy parameters are in units of kT (thermal energy)
- Higher w values favor helix formation
- c parameters control coil state preferences
- v handles chain end effects

## Output Formats

### Equilibrium Tables
```
s,w             s               w               P(s,w)
00              00              00              0.123456e-02
01              00              01              0.234567e-03
...
```

### Statistical Analysis
The `Results()` function returns a tuple containing:
1. KL divergence D(P_map(s,w)||P_eq(s,w))
2. KL divergence D(P_map(s)||P_eq(s))  
3. KL divergence D(P_map(w)||P_eq(w))
4. Expected helicity (equilibrium)
5. Expected helicity (mapped distribution)
6. Average energy (equilibrium)
7. Average energy (mapped distribution)
8. Conditional entropy H(s|w) (equilibrium)
9. Conditional entropy H(s|w) (mapped)
10. Fold entropy H(w) (equilibrium)
11. Fold entropy H(w) (mapped)
12. Sequence entropy H(s) (equilibrium)
13. Sequence entropy H(s) (mapped)

## Performance Considerations

### Memory Usage
- Equilibrium tables scale as O(alphabet_size^length)
- Practical limits: ~15-20 residues for full enumeration
- Matrix methods enable analysis of longer sequences

### Computational Complexity
- Partition function: O(alphabet_size^2 × length)
- Full enumeration: O(alphabet_size^length)
- Parallel scaling: Near-linear with OpenMP

### Optimization Tips
- Enable OpenMP for multi-core systems
- Consider approximate methods for very long sequences

## Applications

### Research Use Cases

Thermodynamics of joint copying-folding systems.

## Research Contributions

### Novel Matrix Fraction Methods

This software introduces **original research methods** for  the estimation of marginalized statistics in long ising chains, using MatrixFraction and VectorFraction.
This method is especially helpful for estimating fold entropy and other information-theoretic quantities on the fold distribution, which normally have no unbiased estimator. However, the exact calculation of p(fold) enables entropy estimation via E[log(p(fold))].

## Validation
- Analytical solutions for simple cases
- Published results from helix-coil literature
- Thermodynamic consistency checks
- Conservation law verification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow C++17 best practices
- Add unit tests for new functionality
- Document all public interfaces
- Maintain backwards compatibility

## Contact

**Jeremy Guntoro**
- GitHub: [@Jeremy1805](https://github.com/Jeremy1805)
- Email: [jeremy.ebg@gmail.com]

## Acknowledgments

- Statistical mechanics formalism based on Zimm-Bragg and Lifson-Roig theories
- Eigen library for high-performance linear algebra
- OpenMP for parallel computing support

---
