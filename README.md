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
- **Sequence-dependence**: Each lattice site i has sequence s_i and fold ω_i degrees of freedom.
- **Nearest-neighbour interactions**: Only interactions between consecutive lattice sites allowed, with energy function E(s_i,ω_i,s_{i+1},ω_{i+1}).
- **State Space**: We are interested in the information-thermodynamic quantities for the equilibrium probability distribution p_eq(s_1s_2...s_L,ω_1ω_2...ω_L) as well as the quenched copy probability distribution p_copy(s_1s_2...s_L,ω_1ω_2...ω_L), along with associated marginalized distributions.  

### Key information-thermodynamic quantities
We list a few key quantities that will be useful

The free energy stored in the quenched distribution relative to the equilibrium distribution can be decomposed as follows:
```
G_copy - G_eq = U_copy - U_eq - k_BT( H_copy(S|Ω) - H_eq(S|Ω) + H_copy(Ω) - H_eq(Ω))
```

#### 1. KL Divergence as Stored Free Energy
The Kullback-Leibler divergence D(p||p_eq) directly measures the **excess free energy** stored in a non-equilibrium distribution:
```
F_excess = kT × D(p_map(ω)||p_eq(ω))
```
This represents the maximum work that could be extracted by allowing the fold distribution to relax from p_map(ω) back to equilibrium p_eq(ω).

**Physical interpretation**: If you have a system with fold distribution p_map(ω) instead of the equilibrium p_eq(ω), you have stored kT × D(p_map(ω)||p_eq(ω)) of free energy that could drive other processes.

#### 2. Entropy Reduction as Organization Work
The reduction in entropy relative to equilibrium measures the **organizational work** done:
```
W_organization = kT × (H_eq(ω) - H_map(ω))
```
This quantifies how much more "organized" (lower entropy) the fold distribution has become relative to equilibrium.

**Physical interpretation**: Creating a more organized (lower entropy) distribution requires work input. The entropy reduction measures this organizational work.

#### Efficiency and the Data Processing Inequality
These measures are related by fundamental inequalities from information theory. For the sequence-to-structure mapping T → S → Ω:
```
D(p_map(s)||p_eq(s)) ≥ D(p_map(ω)||p_eq(ω))
```
This gives an efficiency measure:
```
η = D(p_map(ω)||p_eq(ω)) / D(p_map(s)||p_eq(s)) ≤ 1
```
**Physical meaning**: The folding process can never create more deviation from equilibrium in the structure than existed in the sequence. Folding acts as a "lossy compression" that necessarily reduces the stored free energy.

#### Energy-Entropy Decomposition
The total excess free energy can be decomposed into energetic and entropic contributions:
```
kT × D(p_map(ω)||p_eq(ω)) = ⟨U⟩_map - ⟨U⟩_eq - kT(H_map(ω) - H_eq(ω))
```
This separates the contribution from **energetic bias** (average energy difference) and **entropic organization** (entropy reduction).

### Transfer Matrix Framework

The transfer matrix method provides an exact solution for nearest-neighbor statistical mechanical models by decomposing the partition function into local contributions:

```
Z = v_start^T × M^(L-1) × v_end
```

where:
- **M**: Transfer matrix encoding local sequence-fold interactions E(s_i,ω_i,s_{i+1},ω_{i+1})
- **v_start, v_end**: Boundary condition vectors
- **L**: Polymer length

#### Two Complementary Approaches

**1. Sequence-Free (Annealed) Models**
In traditional statistical mechanics, both sequence and structure equilibrate simultaneously:
```
p_eq(s,ω) = exp(-E(s,ω)/kT) / Z_total
```
This gives the equilibrium distribution over all possible (sequence, structure) pairs, treating the sequence as a thermodynamic variable that can change to minimize free energy.

**2. Sequenced (Quenched) Models**  
In biological systems, the sequence is often fixed (quenched disorder) while structure equilibrates:
```
p_eq(ω|s) = exp(-E(s,ω)/kT) / Z(s)
p(ω) = Σ_s p_copy(s) × p_eq(ω|s)
```
Here, we have a distribution p_copy(s) over sequences (from copying/evolution), and each sequence s folds according to its own equilibrium p_eq(ω|s). The overall fold distribution p(ω) results from marginalizing over the sequence distribution.

#### Mathematical Structure
For a combined state space with N_s sequence states and N_ω fold states, the transfer matrix M is (N_s × N_ω) × (N_s × N_ω), with elements:
```
M[(s_i,ω_i), (s_{i+1},ω_{i+1})] = exp(-E(s_i,ω_i,s_{i+1},ω_{i+1})/kT)
```

**Sequence-free case**: Standard eigenvalue analysis gives thermodynamic properties
**Sequenced case**: Requires marginalization techniques, including our novel reciprocal matrix methods

## Key Features

### Exact Methods (Short Polymers)
- **Transfer Matrix Calculations**: Exact partition functions and probabilities
- **Full State Enumeration**: Complete (s,ω) distribution for systems up to ~15 residues
- **Thermodynamic Consistency**: Rigorous statistical mechanical framework

### Novel Computational Methods (Long Polymers)
- **Reciprocal Matrix Arithmetic**: Breakthrough method for exact p(ω) calculation in quenched disorder
- **VectorFraction Classes**: Automatic consolidation prevents exponential scaling
- **Unbiased Entropy Estimation**: Calculate H[p(ω)] exactly using E[ln p(ω)]

### Sequence Distribution Generation
- **Bernoulli Models**: Independent site probabilities
- **Template-Based**: Error-prone copying from template sequences  
- **Custom Distributions**: User-defined sequence probabilities

### Analysis Capabilities
- **Information-Thermodynamic Quantities**: Free energy differences, efficiency measures, work calculations
- **Thermodynamic Properties**: Partition functions, average energies, heat capacities
- **Entropy Analysis**: Sequence entropy, fold entropy, conditional entropies

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
git clone https://github.com/Jeremy1805/HelixCoilSampler.git
cd HelixCoilSampler
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Usage

### JSON Configuration System
The software uses JSON files to configure parameter scans:

```json
{
    "scan_type": "bernoulli",
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
    "start_vector": ["1.0", "1.0", "1.0", "1.0"],
    "end_vector": ["exp(13.5*x)", "exp(13.5*x)", "exp(2.4*x)", "exp(2.4*x)"],
    "x_var_is_log": true
}
```

### Running Scans
```bash
./configurable_scanner configs/bernoulli_scan.json
```

### Programming Interface
```cpp
#include "FoldModels.h"

// Create 2-state Ising model (helix-coil)
Ising2 model(w_helix, w_coil, c_helix, c_coil, end_effect, length);

// Calculate equilibrium
model.GetEquilibrumTable();

// Analyze with sequence distribution  
auto bernoulli_seq = IsingVar::GenerateBernoulliMap(0.3, length);
auto results = model.Results(bernoulli_seq);

// Extract key quantities
double kl_joint = std::get<0>(results);      // D(p_map(s,ω)||p_eq(s,ω))
double kl_sequence = std::get<1>(results);   // D(p_map(s)||p_eq(s)) 
double kl_fold = std::get<2>(results);       // D(p_map(ω)||p_eq(ω))
double efficiency = kl_fold / kl_sequence;   // Information transfer efficiency
```

## Model Types

### Ising2: Two-State Helix-Coil Model
Each residue can be in helix (ω=0) or coil (ω=1) conformation:
- **Parameters**: w_helix, w_coil, c_helix, c_coil, boundary_effects
- **Applications**: Basic information transfer studies, phase transition analysis
- **Computational limit**: ~20 residues for exact enumeration

### Ising2S3F: Extended Three-Fold Model  
Each residue can adopt helix, antihelix, or coil conformations:
- **Parameters**: w00, w11, a01, a10, v (antihelix couplings)
- **Applications**: More complex information channels, multi-state transitions
- **Computational limit**: ~15 residues for exact enumeration

### Custom Matrix Models
Define arbitrary transfer matrices through JSON configuration:
- **Flexible energy functions**: Mathematical expressions in parameters
- **Variable alphabets**: 2-16 states per position  
- **Research applications**: Test novel information processing hypotheses

## Output and Analysis

### Scan Results (TSV format)
```
bernoulli_prob    eps    D(pmap(s,w)||peq(s,w))    D(pmap(s)||peq(s))    D(pmap(w)||peq(w))    ...
-8.0             0.0    2.345e-02                  1.234e-02             8.765e-03             ...
-8.0             0.1    2.456e-02                  1.245e-02             8.876e-03             ...
```

### Key Outputs Explained
- **D(p||p_eq)**: KL divergence measuring stored free energy (excess work)
- **Efficiency η**: Ratio of fold to sequence KL divergences (≤1 by data processing inequality)
- **Entropies H(s), H(ω)**: Information content and organizational work
- **Energies ⟨U⟩**: Average folding energies under different distributions

## Performance and Scaling

### Computational Complexity
- **Exact enumeration**: O(alphabet^length) - feasible to ~15-20 residues
- **Transfer matrix**: O(alphabet^2 × length) - efficient for thermodynamic properties
- **Matrix fractions**: O(length) growth for special parameter regimes

### Memory Requirements  
- **Short systems** (L≤15): ~1-100 MB
- **Medium systems** (L≤20): ~1-10 GB  
- **Long systems** (L>20): Matrix fraction methods scale favorably

### Parallel Performance
OpenMP parallelization provides near-linear scaling for:
- Equilibrium table generation
- Statistical analysis loops
- Parameter scanning

## Applications and Research Uses

### Information Thermodynamics Studies
- **Information-energy trade-offs**: How physical constraints limit information processing
- **Efficiency optimization**: Design principles for optimal information transfer
- **Thermodynamic bounds**: Fundamental limits on information processing in physical systems

### Channel Capacity Analysis  
- **Sequence-to-structure channels**: Maximum information transferable through folding
- **Noise analysis**: How thermal fluctuations affect information transmission
- **Error correction**: Physical mechanisms for robust information transfer

### Phase Transition Studies
- **Information flow across transitions**: How phase behavior affects information processing
- **Critical phenomena**: Information-theoretic signatures of phase transitions
- **Order parameters**: Information content as a probe of system organization

### Fundamental Physics Applications
- **Test bed for theory**: Validate information-thermodynamic principles
- **Scaling laws**: How information processing scales with system size
- **Universal behavior**: Identify general principles across physical systems

## Novel Research Contributions

### Reciprocal Matrix Arithmetic
This software implements the **first computational method** for exact calculation of marginalized probabilities p(ω) in quenched disorder systems. Key innovations:

1. **Mathematical Framework**: Objects of form `numerator/matrix_denominator` with novel arithmetic rules
2. **Vector Consolidation**: Automatic merging of parallel directions prevents exponential growth  
3. **Exact Entropy Estimation**: Enables E[ln p(ω)] calculation for unbiased H[p(ω)] estimation
4. **Computational Breakthrough**: Makes tractable previously impossible calculations for long polymers

## Support and Contact

- **GitHub Issues**: Bug reports and feature requests
- **Email**: jeremy.ebg@gmail.com
- **Research Group**: [Ouldridge Lab, Imperial College London](https://www.imperial.ac.uk/principles-of-biomolecular-systems/)

## License

MIT License - see LICENSE file for details.

---

*This software represents ongoing research in information thermodynamics and statistical mechanics. We appreciate feedback and collaboration from the scientific community.*
