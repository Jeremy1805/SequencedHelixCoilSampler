/**
 * @file BernoulliScan.cpp
 * @brief Parameter scanning application for Bernoulli sequence distributions
 * 
 * This application performs systematic parameter scans over energy parameters
 * and Bernoulli probabilities to analyze the relationship between sequence
 * bias and folding preferences in helix-coil transition models.
 * 
 * The scan generates data suitable for analyzing:
 * - Information-theoretic quantities (KL divergences, entropies)
 * - Thermodynamic properties (average energies, helicity)
 * - Sequence-structure relationships
 * 
 * Output is saved as tab-separated values for analysis in R, Python, or Excel.
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
#include "FoldModels.h"
#include <omp.h>

/**
 * @brief Main scanning loop for Bernoulli parameter analysis
 * @return Exit code (0 for success)
 * 
 * Performs a nested loop scan over:
 * - Bernoulli probability parameters (log scale from 2^-8 to 2^-0.5)
 * - Energy parameters (exponential scale from 1 to exp(4))
 * 
 * For each parameter combination:
 * 1. Creates Ising2 model with specified energy parameters
 * 2. Generates equilibrium distribution
 * 3. Creates Bernoulli sequence distribution
 * 4. Analyzes information-theoretic and thermodynamic properties
 * 5. Saves results to TSV file
 * 
 * The scan demonstrates how sequence bias affects folding preferences
 * and can be used to optimize sequence design for desired structures.
 */
int main() {
    Timer t1("Full operation");
    
    /// Results table storing all computed quantities for each parameter combination
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> tbl;
    
    /// Polymer length for analysis
    int l = 3;

    // Example template sequences (currently using single template)
    std::vector<std::string> template_11 = {"10010110101"};
    /*template_11 = {"10010110101",
        "00111001011",
        "11001010001",
        "01011100110",
        "10100010111"};*/
    
    /**
     * Outer loop: Scan over Bernoulli probability parameters
     * Range: log₂(p) from -8 to -0.5 in steps of 0.5
     * This covers Bernoulli probabilities from ~0.004 to ~0.35
     */
    for (double lerr = -8; lerr < -0.51; lerr = lerr+0.5) {
        /// Generate Bernoulli sequence distribution with probability 2^lerr
        std::unordered_map<std::string,double> SCopyMap = IsingVar::GenerateBernoulliMap(pow(2.0,lerr),l);
        
        /**
         * Inner loop: Scan over energy parameters  
         * Range: exp(pwr) from 1 to exp(4) ≈ 54.6 in steps of 0.1
         * This covers a wide range of helix stability preferences
         */
        for (double pwr = 0.0; pwr < 4.05; pwr = pwr+0.1) {
            double whom1 = exp(pwr);  ///< Helix-helix interaction strength for state 0
            double whom2 = exp(pwr);  ///< Helix-helix interaction strength for state 1  
            double ahet1 = exp(pwr);  ///< Unused in current Ising2 model
            double ahet2 = exp(pwr);  ///< Unused in current Ising2 model
            
            /// Create symmetric Ising2 model with equal helix preferences
            Ising2 ISInst(whom1, whom2, 1, 1, 1, l);
            
            // Alternative: 3-fold state model (commented out)
            //Ising2S3F ISInst(whom1, whom2, ahet1, ahet2, 1, l);
            
            /// Generate complete equilibrium distribution table
            ISInst.GetEquilibrumTable();
            
            // Debug output (commented out)
            // ISInst.printAll();
            
            std::cout << "SW Equilibrium Validation: " << ISInst.EquilibriumValidation << std::endl;
            //std::cout << SCopyMap["11111111111"] << std::endl;
            
            /// Analyze sequence distribution vs equilibrium
            auto entry = ISInst.Results(SCopyMap);
            
            /// Store results with parameter values prepended
            tbl.push_back(std::tuple_cat(std::make_tuple(lerr),std::make_tuple(pwr),entry));
        }
    }

    /**
     * Save complete results table to TSV file
     * 
     * Columns include:
     * - bernoulli: log₂(Bernoulli probability)
     * - eps: energy parameter (log scale)
     * - Various KL divergences, entropies, and thermodynamic quantities
     */
    saveTuplesToCSV(tbl, "symmetric_bern_log_scan_3.tsv",
        {"bernoulli","eps","D(pmap(s,w)||peq(s,w))",
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
}
