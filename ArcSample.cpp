/**
 * @file ArcSample.cpp
 * @brief Arc walking application for systematic probability space exploration
 * 
 * This application demonstrates advanced probability space exploration using
 * arc walks on the probability simplex. It generates random tangent directions
 * from equilibrium and explores nearby probability distributions to analyze
 * how small perturbations affect thermodynamic and information-theoretic properties.
 * 
 * The arc walking method enables:
 * - Systematic exploration of probability space
 * - Analysis of sensitivity to perturbations
 * - Study of information geometry in protein folding
 * - Generation of diverse test distributions
 * 
 * Results are suitable for studying the geometric structure of sequence-folding
 * relationships and can inform sequence design strategies.
 */

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <string>
#include <unordered_map>
#include <tuple>
#include "FoldModels.cpp"
#include <omp.h>

/**
 * @brief Main arc sampling loop for probability space exploration
 * @return Exit code (0 for success)
 * 
 * Performs systematic exploration of probability space using arc walks:
 * 
 * 1. Creates multiple independent random tangent vectors from equilibrium
 * 2. For each tangent vector, performs arc walks at different angles
 * 3. Analyzes thermodynamic and information-theoretic properties along each arc
 * 4. Saves complete results for statistical analysis
 * 
 * The exploration reveals how the geometric structure of probability space
 * relates to physical properties like energy and entropy, providing insights
 * into the information geometry of protein folding.
 */
int main() {
    Timer t1("Full operation");
    
    /// Results table storing all computed quantities for each arc walk step
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> tbl;
    
    /// Polymer length for analysis
    int l = 6;
    
    /// Random device for generating different tangent directions
    std::random_device rd;
    
    /// Fixed energy parameter for this exploration
    double pwr = 2.0;
    
    /**
     * Outer loop: Generate multiple independent tangent directions
     * Each trial creates a new random tangent vector for exploration
     */
    for (int trials = 0; trials < 10; trials++){
        double whom1 = exp(pwr);  ///< Helix-helix interaction strength for state 0
        double whom2 = exp(pwr);  ///< Helix-helix interaction strength for state 1
        double ahet1 = exp(pwr);  ///< Unused in current Ising2 model
        double ahet2 = exp(pwr);  ///< Unused in current Ising2 model
        
        /// Create symmetric Ising2 model
        Ising2 ISInst(whom1, whom2, 1, 1, 1, l);
        
        /// Generate equilibrium distribution (starting point for arc walks)
        ISInst.GetEquilibrumTable();
        
        /// Generate random seed for this trial
        unsigned int seed = rd();
        
        /// Create random tangent vector orthogonal to equilibrium distribution
        std::unordered_map<std::string,double> SqrtTangentDirs = ISInst.EquilibriumTangentSample(seed);
        
        /**
         * Inner loop: Walk along arc in tangent direction
         * 
         * Currently set to sample near π (angle ≈ 3.14) with fine resolution.
         * The range can be adjusted to explore different regions of the simplex.
         * 
         * Arc walking formula: P(θ) = (√P_eq * cos(θ) + tangent * sin(θ))²
         * where θ is the arc angle parameter.
         */
        for (double angle = 3.14; angle < 3.15; angle += 6.28/20) {
            std::cout << "SW Equilibrium Validation: " << ISInst.EquilibriumValidation << std::endl;
            
            /// Generate new probability distribution at this arc position
            std::unordered_map<std::string,double> SCopyMap = ISInst.ArcWalk(SqrtTangentDirs,angle);
            
            /// Analyze properties of this distribution vs equilibrium
            auto entry = ISInst.Results(SCopyMap);
            
            /// Store results with seed and angle parameters
            tbl.push_back(std::tuple_cat(std::make_tuple(seed),std::make_tuple(angle),entry));
        }
    }
   
    /**
     * Save complete arc walk results to TSV file
     * 
     * Columns include:
     * - seed: Random seed used for tangent vector generation
     * - angle: Arc angle parameter (radians)
     * - Various KL divergences, entropies, and thermodynamic quantities
     * 
     * The data can be analyzed to understand:
     * - How information-theoretic quantities vary along arcs
     * - Sensitivity of physical properties to probability perturbations  
     * - Geometric structure of the probability simplex
     * - Optimal directions for sequence design
     */
    saveTuplesToCSV(tbl, "test.tsv",
        {"seed","angle","D(pmap(s,w)||peq(s,w))",
        "D(pmap(s)||peq(s))",
        "D(pmap(w)||peq(w))",
        "NHelixeq",
        "NHelixMap",
        "<U>eq",
        "<U>map",
        "Heq(s|w)",
        "Hmap(s|w)",
        "Heq(w)",
        "Hmap(w)",
        "Heq(s)",
        "Hmap(s)"});

    return 0;
}