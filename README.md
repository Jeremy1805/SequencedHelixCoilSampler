# SequencedHelixCoilSampler

**Statistical Mechanics Simulation for Protein Folding with Sequence Information**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.0%2B-green.svg)](https://eigen.tuxfamily.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-orange.svg)](https://www.openmp.org/)

## Overview

A C++ implementation for analyzing the information thermodynamics of sequence-to-structure mapping using helix-coil models (specifically the Zimm-Bragg model) as a testbed. This software combines statistical mechanics with information theory to understand fundamental principles of work-information tradeoffs in systems where a copied (quenched) ensemble of sequences adopts an ensemble of folds. The Zimm-Bragg model is used as a toy model for its simplicity.  

## Scientific Background

### Information Thermodynamics of Joint Copying-folding Systems
Polymer copying is fundamentally nonequilibrium (see Ouldridge 2017 and Poulton 2019 to understand why). On the other hand, polymer polfing is often envisioned as anequilibrium process. The question then, is as follows: **How is the nonequilibrium work exerted during copying utilized in folding?**

### Zimm-Bragg Model
We use the Zimm-Bragg helix-coil folding model as a toy model with which to investigate the above question. Briefly, the Zimm-Bragg model has the following features:
- **1D lattice model**: For a polymer of length L, assume L lattice sites arranged in 1D. 
- **Sequence-dependence**: Each lattice site i has sequence s_i and fold ω_i degrees of freedom. Let s = s_1s_2...s_L and ω = ω_1ω_2...ω_L.
- **Nearest-neighbour interactions**: Only interactions between consecutive lattice sites allowed, with energy function U(s_i,ω_i,s_{i+1},ω_{i+1}).
- **State Space**: We are interested in the information-thermodynamic quantities for the equilibrium probability distribution p_eq(s,ω) as well as the quenched copy probability distribution p_map(s,ω), along with associated marginalized distributions (eg. p_eq(s), p_map(ω), ...).  

### Key information-thermodynamic quantities
Without getting into the weeds of the thermodynamics, we list some key quantities we calculate using this program (units of k_BT assumed)
- **Stored Free Energy (minimal work required to shift the ensemble from p_eq to p_map)**: G_copy-G_eq = D(p_map(s,ω)||p_eq(s,ω))
- **KL-Divergence in the marginalized fold distribution**: D(p_map(ω)||p_eq(ω))
- **Entropies in fold-marginalized distributions**: H[p_map(ω)] and H[p_eq(ω)]
- **Entropies in conditional distributions**: <H[p_map(s|ω)]> and <H[p_eq(s|ω)]>, averaged over p_map(ω) and p_eq(ω) respectively
- **Equilibrium and quenched energies**: <U(s,ω)>, with average over p_eq(s) and p_map(s)

### Measures of useful work
Some of the work G_copy-G_eq is clearly 'wasted' on shifting the distribution of s instead of ω. Some measures of useul work:
- **D(p_map(ω)||p_eq(ω))**
- **H[p_eq(ω)]-H[p_map(ω)]**

## Two Complementary Approaches

### Transfer Matrix Framework for Short Polymers
Ising-like dynamics allow the use of transfer matrices to calculate p_eq(s,ω) exactly. Then, given p_map(s), p_map(s,ω) can be calculated using Bayes' theorem, so that: 
```
p_map(s,ω) = p_eq(s,ω)/p_eq(s)p_map(s)
```

## Novel Method for Long Polymers
- **Unbiased entropy sampling**: Generally, no unbiased estimator of entropy, making sampling difficult for long polymers
- **Reciprocal matrices**: Enable precise estimation of p(ω), hence H[[(ω)] = <log p(ω)> is an unbiased estimator.

## Installation

### Dependencies
```bash
# Required
C++17 compiler (GCC 7+, Clang 6+)
Eigen3 (3.3+)
CMake (3.12+)
nlohmann/json
muParser

# Optional  
OpenMP (parallel computing)
```

### Quick Install (Ubuntu/Debian)
```bash
sudo apt-get install build-essential cmake libeigen3-dev libmuparser-dev nlohmann-json3-dev libomp-dev
git clone https://github.com/Jeremy1805/SampledHelixCoilSampler.git
cd SampledHelixCoilSampler
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Usage

### JSON Configuration System
The software uses JSON files to configure parameter scans:

```json
{
    "scan_type": "error",
    "fold_model": "Ising2", 
    "length": 11,
    "x_param_range": {
        "name": "bernoulli_prob",
        "start": -8, "end": -1, "step": 0.5
    },
    "y_param_range": {
        "name": "eps", 
        "start": 0.0, "end": 4.0, "step": 0.1
    },
    "energy_matrix": [
        ["exp(-8.6*x)", "exp(13.5*x)", "exp(13.5*x)", "exp(13.5*x)"],
        ["exp(13.5*x)", "exp(13.5*x)", "exp(13.5*x)", "exp(13.5*x)"],
        ["exp(2.4*x)", "exp(2.4*x)", "exp(7.3*x)", "exp(2.4*x)"],
        ["exp(2.4*x)", "exp(2.4*x)", "exp(2.4*x)", "exp(2.4*x)"]
    ],
    "templates": ["11100100111","10100100010"]
    "start_vector": ["1.0", "1.0", "1.0", "1.0"],
    "end_vector": ["exp(13.5*x)", "exp(13.5*x)", "exp(2.4*x)", "exp(2.4*x)"],
    "x_var_is_log": true
}
```
'bernoulli' assumes a bernoulli-distributed p_map(s). 'error' assumes a starting ensemble given by 'templates', corrupted by some error rate. error and bernoulli parameters can be made logarithmic with "x_var_is_log".

### Running Scans
```bash
./configurable_scanner configs/bernoulli_scan.json
```

### Usage
```cpp
main /path/to/config/file.config /path/to/output/directory
```
or, with default ./results directory
```cpp
main /path/to/config/file.config
```

## Model Types

### Ising2: Two-State Helix-Coil Model
Each residue can be in helix (ω=0) or coil (ω=1) conformation.

### Ising2S3F: Extended Three-Fold Model  
Implementation pending. 

## Output and Analysis

### Scan Results (TSV format)
```
bernoulli_prob    eps    D(pmap(s,w)||peq(s,w))    D(pmap(s)||peq(s))    D(pmap(w)||peq(w))    ...
-8.0             0.0    2.345e-02                  1.234e-02             8.765e-03             ...
-8.0             0.1    2.456e-02                  1.245e-02             8.876e-03             ...
```

## Performance and Scaling

### Computational Complexity
- **Exact enumeration**: O(alphabet^length) - feasible to ~13 residues under typical memory and runtime constraints. Up to ~15/16 feasible in HPC environments.
- **Matrix fractions**: O(length) at fixed precision.

### Parallel Performance
OpenMP parallelization available.

## Support and Contact

- **GitHub Issues**: Bug reports and feature requests
- **Email**: jeremy.ebg@gmail.com
- **Research Group**: [Ouldridge Lab, Imperial College London](https://www.imperial.ac.uk/principles-of-biomolecular-systems/)
