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
#include "EquilibriumPartitionMapGenerator.cpp"
#include "Utilities.cpp"
#include <omp.h>
#include "CustomMatrix.cpp"
#include <random>

#ifndef DEFAULT_EPSILON_DEFINED
#define DEFAULT_EPSILON_DEFINED

inline constexpr double DEFAULT_EPSILON = 1e-300;

#endif // DEFAULT_EPSILON_DEFINED

/**
 * @class GFold
 * @brief Base class for general protein folding models using statistical mechanics
 * 
 * This class provides the fundamental framework for analyzing protein folding
 * using transfer matrix methods and statistical mechanical approaches. It handles
 * equilibrium calculations, partition functions, and various probability distributions.
 */
class GFold {
public:
    /// Character conversion arrays for efficient integer-to-character mapping
    static constexpr std::array<char, 16> INT_TO_CHAR = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f'
    };
    
    /// Character-to-integer lookup table for fast conversion
    static constexpr std::array<int, 128> CHAR_TO_INT = []() {
        std::array<int, 128> arr{};
        for (int i = 0; i < 10; ++i) arr['0' + i] = i;
        for (int i = 0; i < 6; ++i) arr['a' + i] = 10 + i;
        return arr;
    }();

    // Weight Matrices for Equilibrium and Quenched Disorder
    Eigen::MatrixXd EqSWMatrix;                                     ///< Equilibrium sequence-weight transfer matrix
    std::vector<std::vector<Eigen::MatrixXd>> QuenchedWeightMatrix; ///< Quenched disorder weight matrices
    
    // Start and end vectors, eigenvalues, partition func
    Eigen::RowVectorXd EqSWstart;  ///< Starting probability vector for transfer matrix
    Eigen::VectorXd EqSWend;       ///< Ending probability vector for transfer matrix
    double SWeigenMax;             ///< Maximum eigenvalue of transfer matrix
    double SWZnum;                 ///< Numerical partition function

    /// Equilibrium probability matrices for sequence-weight states
    std::vector<Eigen::MatrixXd> EqProbList;
    /// Conditional probability maps for sequences given previous states
    std::vector<std::vector<Eigen::MatrixXd>> SeqProbMapsCond;
    /// Single-site probability maps for sequences
    std::vector<Eigen::RowVectorXd> SeqProbMapsSS;
    /// Equilibrium probability matrices for weight states
    std::vector<Eigen::MatrixXd> WeqProbList;
    /// Conditional probability maps for weights given previous states
    std::vector<std::vector<Eigen::MatrixXd>> WeqProbMapsCond;
    /// Single-site probability maps for weights
    std::vector<Eigen::RowVectorXd> WeqProbMapsSS;

    // For Bernoulli p(w) evaluations using matrix fractions
    Matrix<VectorFractionList> start_vec_frac; ///< Starting vector fractions for matrix calculations
    Matrix<MatrixFraction> indep_mat_frac;     ///< Independent matrix fractions
    Matrix<MatrixFraction> end_vec_frac;       ///< Ending vector fractions

    // Start and End for Quenched Disorder
    Eigen::RowVectorXd Qstart;  ///< Starting vector for quenched disorder calculations
    Eigen::VectorXd Qend;       ///< Ending vector for quenched disorder calculations

    int L; ///< Real polymer length (number of monomers)
    
    /// Complete equilibrium table with all sequence-weight combinations
    std::vector<std::tuple<std::string, std::string, std::string, double, double>> EquilibriumTable;
    std::unordered_map<std::string, double> SEquilibriumMap; ///< Sequence probability map
    std::unordered_map<std::string, double> WEquilibriumMap; ///< Weight (fold) probability map
    double EquilibriumValidation; ///< Sum of all probabilities for validation

    /**
     * @brief Calculate the dominant eigenvalue of a weight matrix
     * @param weightMatrix The transfer matrix to analyze
     * @return The maximum real eigenvalue
     * 
     * Uses Eigen's ComplexEigenSolver to find eigenvalues and returns the
     * maximum real part, which corresponds to the dominant eigenvalue
     * in the Perron-Frobenius sense for positive matrices.
     */
    double GetEigen(Eigen::MatrixXd weightMatrix) {
        Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces(weightMatrix);
        Eigen::VectorXcd eigenvals = ces.eigenvalues();
        double maxReal = eigenvals.real().maxCoeff();
        return(maxReal);
    }

    /**
     * @brief Calculate numerical partition function using matrix multiplication
     * @param start Starting probability vector
     * @param end Ending probability vector  
     * @param weightMatrix Transfer matrix
     * @return Partition function value
     * 
     * Computes Z = start * weightMatrix^(L-1) * end by successive matrix
     * multiplications. More accurate than eigenvalue method for finite systems.
     */
    double getNumPartition(Eigen::RowVectorXd start, Eigen::VectorXd end, Eigen::MatrixXd weightMatrix) {
        Eigen::RowVectorXd run = start;
        for (int i = 0; i < L-1; i++) {
            //L-1 because start already counts the first monomer. end does NOT correspond to a monomer
            run = run * weightMatrix;
        }
        return(run.dot(end));
    }

    /**
     * @brief Calculate partition function using dominant eigenvalue approximation
     * @return Eigenvalue-based partition function
     * 
     * Returns λ_max^L where λ_max is the dominant eigenvalue.
     * Valid in the thermodynamic limit (large L).
     */
    double eigenPartition() const {
        return std::pow(SWeigenMax, L);
    }

    /**
     * @brief Calculate free energy per monomer from eigenvalue
     * @return Free energy G = ln(λ_max)
     */
    double eigenG() const {
        return std::log(SWeigenMax);
    }

    /**
     * @brief Calculate free energy per monomer from numerical partition function
     * @return Free energy G = ln(Z)/L
     */
    double Gnum() const {
        return std::log(SWZnum)/L;
    }

    /**
     * @brief Calculate all partition functions for the current model
     * 
     * Computes the numerical partition function using the current
     * transfer matrix and start/end vectors.
     */
    void CalcAllPartition() {
        SWZnum = getNumPartition(EqSWstart,EqSWend, EqSWMatrix);
    }

    /**
     * @brief Calculate all eigenvalues for the current model
     * 
     * Computes the dominant eigenvalue of the transfer matrix.
     */
    void CalcAllEigen() {
        SWeigenMax = GetEigen(EqSWMatrix); 
    }

    /**
     * @brief Generate probability matrices for sequence-weight calculations
     * 
     * Creates conditional probability matrices P(state_i | state_{i-1}) for
     * each position in the chain using backward iteration from the end vector.
     * These matrices enable efficient calculation of sequence probabilities.
     */
    void getPSWMatrices() {
        Eigen::MatrixXd CondProb_L_gvn_Lm1 = RowNormalize(multiplytoColVec(EqSWMatrix,EqSWend));
        Eigen::VectorXd backward_iter_vec = EqSWMatrix*EqSWend;

        EqProbList.resize(L);
        
        EqProbList[L-1] = CondProb_L_gvn_Lm1;
        std::cout << EqProbList[L-1] << std::endl;
        std::cout <<std::endl;
        for (int i = L-2; i > 0; i--) {
            EqProbList[i] = RowNormalize(multiplytoColVec(EqSWMatrix,backward_iter_vec));
            backward_iter_vec = EqSWMatrix*backward_iter_vec;
            std::cout << EqProbList[i] << std::endl;
            std::cout <<std::endl;
        }

        EqProbList[0] = RowNormalize(multiplytoColVec(EqSWstart,backward_iter_vec));
        std::cout << EqProbList[0] << std::endl;
        std::cout <<std::endl;
    }

    /**
     * @brief Calculate probability of a sequence-weight combination using precomputed matrices
     * @param seq Sequence string in internal alphabet
     * @return Probability P(sequence, weight)
     * 
     * Uses the probability matrices generated by getPSWMatrices() to efficiently
     * calculate the joint probability of any sequence-weight combination.
     */
    double CalcPSWfromMatrices(std::string seq) {
        double prob = EqProbList[0](CHAR_TO_INT[seq[0]]);
        for (int i = 1; i < L; i++) {
            prob = prob*EqProbList[i](CHAR_TO_INT[seq[i-1]],CHAR_TO_INT[seq[i]]);
        }

        return(prob);
    }

    /**
     * @brief Calculate probability of a sequence using sequence mapping matrices
     * @param seq Sequence string
     * @return Marginal probability P(sequence)
     * 
     * Computes the probability of observing a particular sequence by summing
     * over all possible weight (fold) configurations using sequence mapping matrices.
     */
    double CalcSfromMatrices(std::string seq) {
        Eigen::RowVectorXd prob = (EqProbList[0].array()*SeqProbMapsSS[CHAR_TO_INT[seq[0]]].array()).matrix();
        for (int i = 1; i < L; i++) {
            Eigen::MatrixXd mapped_matrix = (EqProbList[i].array()*SeqProbMapsCond[CHAR_TO_INT[seq[i-1]]][CHAR_TO_INT[seq[i]]].array()).matrix();
            prob = prob*mapped_matrix;
        }

        return(prob.sum());
    }

    /**
     * @brief Calculate probability of a weight (fold) configuration using weight mapping matrices
     * @param seq Sequence string (used to map to weight states)
     * @return Marginal probability P(weight)
     * 
     * Computes the probability of observing a particular fold configuration
     * by summing over all possible sequences using weight mapping matrices.
     */
    double CalcWfromMatrices(std::string seq) {
        Eigen::RowVectorXd prob = (EqProbList[0].array()*WeqProbMapsSS[CHAR_TO_INT[seq[0]]].array()).matrix();
        for (int i = 1; i < L; i++) {
            Eigen::MatrixXd mapped_matrix = (EqProbList[i].array()*WeqProbMapsCond[CHAR_TO_INT[seq[i-1]]][CHAR_TO_INT[seq[i]]].array()).matrix();
            prob = prob*mapped_matrix;
        }

        return(prob.sum());
    }

};

/**
 * @class IsingVar
 * @brief Ising model variant with sequence-to-fold mapping functionality
 * 
 * Extends GFold to provide specific implementations for Ising-type models
 * with explicit mappings between sequence states (S) and weight/fold states (W).
 * Includes methods for generating equilibrium distributions and analyzing
 * sequence-structure relationships.
 */
class IsingVar : public GFold {
    public:
    std::unordered_map<char, char> IsingSlookup; ///< Maps combined SW states to sequence states
    std::unordered_map<char, char> IsingWlookup; ///< Maps combined SW states to weight states
        // SSlices and Wslices are more convenient when calculating probabilities with 
            // Matrix Fractions 
    std::unordered_map<char, std::vector<size_t>> IsingSSlices; ///< Sequence state index slices
    std::unordered_map<char, std::vector<size_t>> IsingWSlices; ///< Weight state index slices

    /**
     * @brief Extract sequence from combined sequence-weight string
     * @param SW Combined sequence-weight string
     * @return Pure sequence string
     */
    std::string GetS(std::string SW){
        std::string ans = "";
        for (char c : SW) {
            ans = ans + IsingSlookup.at(c);
        }
        return(ans);
    }

    /**
     * @brief Extract weight (fold) configuration from combined sequence-weight string
     * @param SW Combined sequence-weight string  
     * @return Pure weight string
     */
    std::string GetW(std::string SW){
        std::string ans = "";
        for (char c : SW) {
            ans = ans + IsingWlookup.at(c);
        }
        return(ans);
    }

    /**
     * @brief Generate complete equilibrium distribution table
     * 
     * Creates the full equilibrium table containing all possible sequence-weight
     * combinations with their probabilities and energies. Also generates marginal
     * distributions for sequences and weights separately.
     */
    void GetEquilibrumTable() {
        EquilibriumPartitionMapGenerator::generateWithPartition(EquilibriumTable,SEquilibriumMap,WEquilibriumMap,EquilibriumValidation,
            L, EqSWMatrix, EqSWstart, EqSWend, SWZnum, EqSWMatrix.rows(), IsingSlookup, IsingWlookup);
    }

    /**
     * @brief Print the complete equilibrium table to console
     * 
     * Displays all sequence-weight combinations with their probabilities
     * and energies in a formatted table.
     */
    void printEquilibriumTable() {
        // Print results
        std::cout << std::left 
                << std::setw(15) << "s,w" << std::setw(15) << "s" << std::setw(15) << "w"
                << std::setw(20) << "P(s,w)" << std::endl;
        std::cout << std::string(35, '-') << std::endl;
        
        for (const auto& [str, str1, str2, value, energy] : EquilibriumTable) {
            std::cout << std::left 
                    << std::setw(15) << str << std::setw(15) << str1 << std::setw(15) << str2
                    << std::scientific << std::setprecision(6) 
                    << value << std::setw(15) << energy << std::endl;
        }
    }

    /**
     * @brief Print sequence marginal distribution to console
     * 
     * Displays the probability distribution over all possible sequences
     * and validates that probabilities sum to 1.
     */
    void printSEquilibriumMap() {
        std::cout << std::left 
                << std::setw(15) << "s" 
                << std::setw(20) << "P(s)" << std::endl;
        std::cout << std::string(35, '-') << std::endl;
        double sum = 0.0;
        for (const auto& [str, value] :  SEquilibriumMap) {
            std::cout << std::left 
                    << std::setw(15) << str 
                    << std::scientific << std::setprecision(6) 
                    << value << std::endl;
            sum += value;
        }
        std::cout << sum << std:: endl;
    }

    /**
     * @brief Print weight (fold) marginal distribution to console
     * 
     * Displays the probability distribution over all possible fold configurations
     * and validates that probabilities sum to 1.
     */
    void printWEquilibriumMap() {
        std::cout << std::left 
              << std::setw(15) << "w" 
              << std::setw(20) << "P(w)" << std::endl;
        std::cout << std::string(35, '-') << std::endl;
        double sum = 0.0;
        for (const auto& [str, value] :  WEquilibriumMap) {
            std::cout << std::left 
                    << std::setw(15) << str 
                    << std::scientific << std::setprecision(6) 
                    << value << std::endl;
            sum += value;
        }

        std::cout << "Sum: " << sum << std::endl;
    }

    /**
     * @brief Print all equilibrium distributions and validation
     * 
     * Convenience method that prints the complete equilibrium table,
     * sequence distribution, and weight distribution with validation sums.
     */
    void printAll() {
        printEquilibriumTable();
        std::cout << "Sum: " << EquilibriumValidation << std::endl;
        std::cout << std::endl;
        printSEquilibriumMap();
        std::cout << std::endl;
        printWEquilibriumMap();
        std::cout << std::endl;
    }

    /**
     * @brief Count number of helix residues in a fold configuration
     * @param fold Fold string where '0' represents helix
     * @return Number of helix residues
     * 
     * Utility function for analyzing secondary structure content.
     * Assumes '0' represents helix state and '1' represents coil state.
     */
    static int countHelix(std::string fold) {
        int helix_count = 0;
        for (char c : fold) {
            if (c == '0'){
                helix_count++;
            }
        }
        return(helix_count);
    }

    /**
     * @brief Calculate Bernoulli probability for a binary sequence
     * @param Prob0 Probability of observing '0' 
     * @param sequence Binary sequence string
     * @return Probability under Bernoulli model
     * 
     * Computes P(sequence) = ∏ᵢ p^{s_i} (1-p)^{1-s_i} where p = Prob0.
     */
    static double GetBernoulliProb(double Prob0, std::string sequence){
        double prob = 1;
        for (char c : sequence) {
            if (c == '0'){
                prob = prob*Prob0;
            } else {
                prob = prob*(1-Prob0);
            }
        }
        return(prob);
    }

    /**
     * @brief Generate probability distribution over all binary sequences using Bernoulli model
     * @param Prob0 Probability of observing '0' at each position
     * @param len Length of sequences
     * @return Map from sequences to their Bernoulli probabilities
     * 
     * Creates a complete probability distribution over all 2^len possible
     * binary sequences using independent Bernoulli trials at each position.
     */
    static std::unordered_map<std::string,double> GenerateBernoulliMap(double Prob0,int len) {
        std::unordered_map<std::string,double>  SCopyMap;
        // Total number of strings will be 2^L
        int n_strings = 1 << len;  // Same as 2^L
        SCopyMap.reserve(n_strings);
        // Generate each number from 0 to 2^L - 1
        for(int i = 0; i < n_strings; i++) {
            std::string binary;
            
            // Convert to binary by checking each bit
            for(int j = len-1; j >= 0; j--) {
                // Check if jth bit is set
                binary += (i & (1 << j)) ? '1' : '0';
            }
            
            SCopyMap[binary] = GetBernoulliProb(Prob0,binary);
        }
        return SCopyMap;
    }

    /**
     * @brief Generate random probability distribution over binary sequences
     * @param seed Random seed for reproducibility
     * @param len Length of sequences
     * @return Map from sequences to random probabilities (properly normalized)
     * 
     * Creates a random probability distribution by drawing from exponential
     * distribution and normalizing. Useful for testing and random sampling.
     */
    static std::unordered_map<std::string,double> SampleRandomProb(unsigned int seed, int len) {
        std::unordered_map<std::string,double>  SCopyMap;
        
         // Total number of strings will be 2^L

        int n_strings = 1 << len;  // Same as 2^L
        SCopyMap.reserve(n_strings);
        
        std::mt19937_64 gen(seed);

        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double normalization = 0.0;
         // Generate each number from 0 to 2^L - 1
        for(int i = 0; i < n_strings; i++) {
            std::string binary;
            
            // Convert to binary by checking each bit
            for(int j = len-1; j >= 0; j--) {
                // Check if jth bit is set
                binary += (i & (1 << j)) ? '1' : '0';
            }
            
            SCopyMap[binary] = -log(dist(gen));
            normalization += SCopyMap[binary];
        }

        for (const auto& pair : SCopyMap) {
            SCopyMap[pair.first] = SCopyMap[pair.first]/normalization;
        }
       
        return SCopyMap;
    }

    /**
     * @brief Generate probability distribution from template sequences with error model
     * @param template_ls List of template sequences
     * @param err_rate Error rate for deviations from templates
     * @return Map from sequences to template-based probabilities
     * 
     * Creates a probability distribution where sequences similar to the templates
     * have higher probability. The probability decreases exponentially with
     * the number of differences from the nearest template.
     */
    static std::unordered_map<std::string,double> GenerateFromUniformTemplate(std::vector<std::string> template_ls, double err_rate) {
        std::unordered_map<std::string,double>  SCopyMap;
        double template_prob = 1/double(template_ls.size());
        // Total number of strings will be 2^L
        int len = template_ls[0].size();
        int n_strings = 1 << len;  // Same as 2^L
        SCopyMap.reserve(n_strings);
        // Generate each number from 0 to 2^L - 1
        for(int i = 0; i < n_strings; i++) {
            std::string binary;
            
            // Convert to binary by checking each bit
            for(int j = len-1; j >= 0; j--) {
                // Check if jth bit is set
                binary += (i & (1 << j)) ? '1' : '0';
            }
            
            for (auto template_str: template_ls) {
                int differences = 0;
                for (int i = 0; i < template_str.length(); i++) {
                    if (template_str[i] != binary[i]) {
                        differences++;
                    }
                }
                //std::cout << template_prob << " " << pow(err_rate,differences) << " " << pow(1-err_rate,len-differences) << std::endl;
                SCopyMap[binary] += template_prob*pow(err_rate,differences)*pow(1-err_rate,len-differences);
            } 
            //std::cout << binary << " " << SCopyMap[binary] << std::endl;
        }
        return SCopyMap;
    }

    /**
     * @brief Calculate folding energy for a sequence-weight configuration
     * @param foldseq Combined sequence-weight string
     * @return Total folding energy
     * 
     * Computes the energy as the negative log of the statistical weights:
     * E = -∑ᵢ ln(w_{i,i+1}) - ln(w_end)
     */
    double CalcFoldEnergy(std::string foldseq) {
        double energy = 0.0;
        for (int i = 1; i < L; i++) { 
                //minus sign because we used positive energy
            energy = energy - log(EqSWMatrix(CHAR_TO_INT[foldseq[i-1]],CHAR_TO_INT[foldseq[i]]));
        }
        energy = energy-log(EqSWend[CHAR_TO_INT[foldseq[L-1]]]);

        return energy;
    }

    /**
     * @brief Comprehensive analysis of sequence distribution vs equilibrium
     * @param SCopyMapIn Input sequence probability distribution
     * @return Tuple containing 13 thermodynamic and information-theoretic quantities
     * 
     * Performs detailed comparison between an input sequence distribution and
     * the equilibrium distribution, computing:
     * - KL divergences for joint, sequence, and weight distributions
     * - Average helicity in equilibrium and mapped distributions  
     * - Average energies
     * - Conditional and marginal entropies
     * - Various information-theoretic measures
     * 
     * Returns tuple: (KL_SW, KL_S, KL_W, EWeq, EWuni, eqEnergy, uniEnergy, 
     *                eqCondEntropy, uniCondEntropy, HWeq, HWcopy, HSeq, HScopy)
     */
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> Results(std::unordered_map<std::string,double> SCopyMapIn) {
        double KL_SW = 0.0;
        double KL_S = 0.0;
        double KL_W = 0.0;

        double validation = 0.0;

        double EWuni = 0.0;
        double EWeq = 0.0;

        double eqEnergy = 0.0;
        double uniEnergy = 0.0;

        double eqCondEntropy = 0.0;
        double uniCondEntropy = 0.0;

        double HWeq = 0.0;
        double HWcopy = 0.0;

        double HSeq = 0.0;
        double HScopy = 0.0;

        std::string fold_idx;

        std::unordered_map<std::string,double> WCopyMap; 
            WCopyMap.reserve(WEquilibriumMap.size());
        std::unordered_map<std::string,double> SCopyMap = SCopyMapIn;

        for (const auto& pair : SEquilibriumMap) {
            validation += SCopyMap[pair.first];
            HScopy += -SCopyMap[pair.first]*log(SCopyMap[pair.first]);
            HSeq += -pair.second*log(pair.second);
            if (SCopyMap[pair.first] > 0) { // This zero handling matters when we are crossing out of the positive orthant
                KL_S += SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second);
            }
            // std::cout << pair.first << " " << SCopyMap[pair.first] << " "<< SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second) << " " << KL_S << std::endl;
        }
        std::cout << "S copy distribution validation: " << validation << std::endl;
        
        #ifdef _OPENMP
        std::cout << "OpenMP is enabled" << std::endl;
        #else
        std::cout << "OpenMP is not enabled" << std::endl;
        #endif

        KL_SW = 0.0;
        // Parallelize stuff that can be parallelized
        #pragma omp parallel for reduction(+:KL_SW,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy)
        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& s_state = std::get<1>(swdata);
            const auto& w_state = std::get<2>(swdata);
            const double foldEn = std::get<4>(swdata);
            const double s_copy_prob = SCopyMap.at(s_state);
            const double s_eq_prob = SEquilibriumMap.at(s_state);
            const double w_eq_prob = WEquilibriumMap[w_state];
            
            const double pcopySW = peqSW * s_copy_prob / s_eq_prob;

            if (pcopySW > 0) { // This zero handling matters when we are crossing out of the positive orthant
                KL_SW += pcopySW * log(pcopySW/peqSW);
                uniCondEntropy += -pcopySW * log(pcopySW);
            }
            
            eqEnergy += peqSW * foldEn;
            uniEnergy += pcopySW * foldEn;
            
            eqCondEntropy += -peqSW * log(peqSW/w_eq_prob);
        }
        
        // Then do WCopyMap updates serially
        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& s_state = std::get<1>(swdata);
            const auto& w_state = std::get<2>(swdata);
            const double s_copy_prob = SCopyMap[s_state];
            const double s_eq_prob = SEquilibriumMap[s_state];
            
            const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
            WCopyMap[w_state] += pcopySW;
        }
        
        validation = 0;
        for (const auto& pair : WEquilibriumMap) {
            validation += WCopyMap[pair.first];
            KL_W += WCopyMap[pair.first]*log(WCopyMap[pair.first]/pair.second);
            double helix_count = countHelix(pair.first);

            HWcopy += -WCopyMap[pair.first]*log(WCopyMap[pair.first]);
            HWeq += -pair.second*log(pair.second);

            EWuni += helix_count*WCopyMap[pair.first];
            EWeq += helix_count*pair.second;
        }
        uniCondEntropy = uniCondEntropy - HWcopy;
        std::cout << "W copy distribution validation: " << validation << std::endl;
        return( std::make_tuple(KL_SW,KL_S,KL_W,EWeq,EWuni,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy,HWeq,HWcopy,HSeq,HScopy) );
    }

    /**
     * @brief Same as Results() but also saves fold distributions to files
     * @param SCopyMapIn Input sequence probability distribution
     * @param filename Base filename for output files
     * @return Same tuple as Results()
     * 
     * Identical functionality to Results() but additionally saves the
     * equilibrium and mapped fold distributions to TSV files for later analysis.
     */
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> ResultsSaveFoldMap(std::unordered_map<std::string,double> SCopyMapIn, std::string filename) {
        double KL_SW = 0.0;
        double KL_S = 0.0;
        double KL_W = 0.0;

        double validation = 0.0;

        double EWuni = 0.0;
        double EWeq = 0.0;

        double eqEnergy = 0.0;
        double uniEnergy = 0.0;

        double eqCondEntropy = 0.0;
        double uniCondEntropy = 0.0;

        double HWeq = 0.0;
        double HWcopy = 0.0;

        double HSeq = 0.0;
        double HScopy = 0.0;
 
        std::string fold_idx;

        std::unordered_map<std::string,double> WCopyMap; 
            WCopyMap.reserve(WEquilibriumMap.size());
        std::unordered_map<std::string,double> SCopyMap = SCopyMapIn;

        for (const auto& pair : SEquilibriumMap) {
            validation += SCopyMap[pair.first];
            HScopy += -SCopyMap[pair.first]*log(SCopyMap[pair.first]);
            HSeq += -pair.second*log(pair.second);
            KL_S += SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second);
        }
        std::cout << "S copy distribution validation: " << validation << std::endl;
        
        #ifdef _OPENMP
        std::cout << "OpenMP is enabled" << std::endl;
        #else
        std::cout << "OpenMP is not enabled" << std::endl;
        #endif

        KL_SW = 0.0;
        // Parallelize stuff that can be parallelized
        #pragma omp parallel for reduction(+:KL_SW,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy)
        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& s_state = std::get<1>(swdata);
            const auto& w_state = std::get<2>(swdata);
            const double foldEn = std::get<4>(swdata);
            const double s_copy_prob = SCopyMap.at(s_state);
            const double s_eq_prob = SEquilibriumMap.at(s_state);
            const double w_eq_prob = WEquilibriumMap[w_state];
            
            const double pcopySW = peqSW * s_copy_prob / s_eq_prob;

            KL_SW += pcopySW * log(pcopySW/peqSW);
            
            eqEnergy += peqSW * foldEn;
            uniEnergy += pcopySW * foldEn;
            
            eqCondEntropy += -peqSW * log(peqSW/w_eq_prob);
            uniCondEntropy += -pcopySW * log(pcopySW);
        }
        
        // Then do WCopyMap updates serially
        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& s_state = std::get<1>(swdata);
            const auto& w_state = std::get<2>(swdata);
            const double s_copy_prob = SCopyMap[s_state];
            const double s_eq_prob = SEquilibriumMap[s_state];
            
            const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
            WCopyMap[w_state] += pcopySW;
        }
        
        validation = 0;
        for (const auto& pair : WEquilibriumMap) {
            validation += WCopyMap[pair.first];
            KL_W += WCopyMap[pair.first]*log(WCopyMap[pair.first]/pair.second);
            double helix_count = countHelix(pair.first);

            HWcopy += -WCopyMap[pair.first]*log(WCopyMap[pair.first]);
            HWeq += -pair.second*log(pair.second);

            EWuni += helix_count*WCopyMap[pair.first];
            EWeq += helix_count*pair.second;
        }
        uniCondEntropy = uniCondEntropy - HWcopy;
        
        saveMapToTSV(WEquilibriumMap, "FOLDEq"+filename);
        saveMapToTSV(WCopyMap, "FOLDMap"+filename);
        
        std::cout << "W copy distribution validation: " << validation << std::endl;
        return( std::make_tuple(KL_SW,KL_S,KL_W,EWeq,EWuni,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy,HWeq,HWcopy,HSeq,HScopy) );
    }

    /**
     * @brief Verify matrix calculation methods against direct enumeration
     * 
     * Cross-validates the matrix-based probability calculations against
     * the direct enumeration results. Throws runtime_error if discrepancies
     * exceed numerical precision tolerance (10^-10).
     */
    void VerifyMatrixApproach() {
        std::cout << "Setting up matrices..." << std::endl;
        
        getPSWMatrices();
        getSEqMatrices();
        getWEqMatrices();

        std::cout << "Verifying..." << std:: endl;
        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& sw_state = std::get<0>(swdata);
            
            if (abs(CalcPSWfromMatrices(sw_state)-peqSW) > pow(10.0,-10)) {
                throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                    + std::to_string(abs(CalcPSWfromMatrices(sw_state)-peqSW)) + " OBTAINED!");
            }
            //std::cout << sw_state << "/" << peqSW << "/" << 
            //CalcPSWfromMatrices(sw_state); 
            //<< std::endl; 
        }
        std::cout << "SW Equilibrium Verification succeeded"<< std::endl;
        for (const auto& sdata : SEquilibriumMap) {
            const double peqS = std::get<1>(sdata);
            const auto& s_state = std::get<0>(sdata);

            //std::cout << s_state << "/" << peqS << "/" << CalcSfromMatrices(s_state) << "/" << CalcSfromMatrices(s_state)-peqS << std::endl; 
    
            if (abs(CalcSfromMatrices(s_state)-peqS) > pow(10.0,-10)) {
                throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                    + std::to_string(abs(CalcSfromMatrices(s_state)-peqS)) + " OBTAINED!");
            }
        }
        std::cout << "S Equilibrium Verification succeeded"<< std::endl;
        for (const auto& wdata : WEquilibriumMap) {
            const double peqW = std::get<1>(wdata);
            const auto& w_state = std::get<0>(wdata);
            
            if (abs(CalcWfromMatrices(w_state)-peqW) > pow(10.0,-10)) {
                throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                    + std::to_string(abs(CalcWfromMatrices(w_state)-peqW)) + " OBTAINED!");
            }
            std::cout << w_state << "/" << peqW << "/" << CalcWfromMatrices(w_state) << std::endl; 
        }
        std::cout << "W Equilibrium Verification succeeded"<< std::endl;

    }

    /**
     * @brief Set up sequence probability mapping matrices
     * 
     * Creates mapping matrices that relate combined sequence-weight states
     * to pure sequence states. These matrices enable efficient calculation
     * of sequence marginal probabilities.
     */
    void getSEqMatrices() {
        // Find maximum s index
        int SWalphabetsize = IsingSlookup.size();
        int maxs = 0;
        
        for (int i = 0; i < SWalphabetsize; i++) {
            int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
            maxs = std::max(maxs,s);
        }
        maxs = maxs+1;

        // Initialize our data structures
        Eigen::MatrixXd default_condmatrix(SWalphabetsize, SWalphabetsize);
        default_condmatrix.setZero();  // Or any initialization
        Eigen::RowVectorXd default_SSvector(SWalphabetsize);
        default_SSvector.setZero();  // Or any initialization

        SeqProbMapsCond.assign( maxs, std::vector<Eigen::MatrixXd>(maxs, default_condmatrix));
        SeqProbMapsSS.assign( maxs, default_SSvector );

        // Define our maps
        for (int i = 0; i < SWalphabetsize; i++) {
            int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
            SeqProbMapsSS[s](i) = 1;
            for (int j = 0; j < SWalphabetsize; j++) {
                int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
                SeqProbMapsCond[s][s2](i,j) = 1;
            }
        }
    }

    /**
     * @brief Set up weight probability mapping matrices
     * 
     * Creates mapping matrices that relate combined sequence-weight states
     * to pure weight states. These matrices enable efficient calculation
     * of weight marginal probabilities.
     */
    void getWEqMatrices() {
        // Find maximum s index
        int SWalphabetsize = IsingWlookup.size();
        int maxw = 0;
        
        for (int i = 0; i < SWalphabetsize; i++) {
            int w = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[i]]];
            maxw = std::max(maxw,w);
        }
        maxw = maxw+1;

        // Initialize our data structures
        Eigen::MatrixXd default_condmatrix(SWalphabetsize, SWalphabetsize);
        default_condmatrix.setZero();  // Or any initialization
        Eigen::RowVectorXd default_SSvector(SWalphabetsize);
        default_SSvector.setZero();  // Or any initialization

        WeqProbMapsCond.assign( maxw, std::vector<Eigen::MatrixXd>(maxw, default_condmatrix));
        WeqProbMapsSS.assign( maxw, default_SSvector );

        // Define our maps
        for (int i = 0; i < SWalphabetsize; i++) {
            int w = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[i]]];
            WeqProbMapsSS[w](i) = 1;
            for (int j = 0; j < SWalphabetsize; j++) {
                int w2 = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[j]]];
                WeqProbMapsCond[w][w2](i,j) = 1;
            }
        }
    }

    /**
     * @brief Set up matrix fraction structures for independent sequence model
     * @param SWalphabetsize Size of combined sequence-weight alphabet
     * @param ssize Number of sequence states
     * @param wsize Number of weight states  
     * @param probs Probability vector for sequence states
     * @param epsilon Numerical precision parameter
     * 
     * Initializes matrix fraction data structures for calculating probabilities
     * under independent sequence models (e.g., Bernoulli distributions).
     * These structures enable exact arithmetic for probability calculations.
     */
    void getIndependentMatrixFractions(int SWalphabetsize,int ssize, int wsize, std::vector<double> probs, double epsilon = DEFAULT_EPSILON) {
        // Initialize the VectorFractionList
        std::vector<std::vector<VectorFractionList>> start_init = std::vector(1,std::vector(SWalphabetsize,VectorFractionList(1,2, epsilon)));
        start_vec_frac = Matrix<VectorFractionList>(start_init);
            // Generate Matrices and end vectors for the calculation of sequence weights
        Eigen::MatrixXd default_matrix = Eigen::MatrixXd::Zero(ssize,ssize);
        Eigen::VectorXd default_vector = Eigen::VectorXd::Zero(ssize);
        
        std::vector<std::vector<Eigen::MatrixXd>> seq_matrices = std::vector(ssize,std::vector(ssize,default_matrix));
        std::vector<Eigen::VectorXd> end_vectors = std::vector(ssize,default_vector);
        
        for (int i = 0; i < SWalphabetsize; i++) {
            char skey = IsingSlookup[INT_TO_CHAR[i]];
            int s = CHAR_TO_INT[skey];
            char wkey = IsingWlookup[INT_TO_CHAR[i]];
            int w = CHAR_TO_INT[wkey];
            IsingWSlices[wkey].push_back(i);
            IsingSSlices[skey].push_back(i);
            end_vectors[s](w) = EqSWend(i);  
            for (int j = 0; j < SWalphabetsize; j++) {
                int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
                char wkey2 = IsingWlookup[INT_TO_CHAR[j]];
                int w2 = CHAR_TO_INT[wkey2];
                seq_matrices[s][s2](w,w2) = EqSWMatrix(i,j);
            }
        }
            // Generate Matrices of MatrixFractions, as well as end MatrixFractions
        std::vector<std::vector<MatrixFraction>> mat_frac_init = std::vector(SWalphabetsize,std::vector<MatrixFraction>(SWalphabetsize));
        std::vector<std::vector<MatrixFraction>> end_frac_init = std::vector(SWalphabetsize,std::vector<MatrixFraction>(1));

        for (int i = 0; i < SWalphabetsize; i++) {
            int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
            end_frac_init[i][0] = MatrixFraction(probs[s]*EqSWend(i),end_vectors[s]);
            for (int j = 0; j < SWalphabetsize; j++) {
                int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
                mat_frac_init[i][j] = MatrixFraction(probs[s]*EqSWMatrix(i,j),seq_matrices[s][s2]);
            }
        }

        indep_mat_frac = Matrix<MatrixFraction>(mat_frac_init);
        end_vec_frac = Matrix<MatrixFraction>(end_frac_init);

        //std::cout << start_vec_frac.toString() << std::endl;
        //std::cout << indep_mat_frac.toString() << std::endl;
        //std::cout << end_vec_frac.toString() << std::endl;
    }

    /**
     * @brief Calculate weight probability using matrix fractions
     * @param seq Weight sequence string
     * @return Probability P(weight) under independent sequence model
     * 
     * Uses matrix fraction arithmetic to compute exact probabilities
     * for weight configurations under independent sequence distributions.
     * More numerically stable than direct floating-point arithmetic.
     */
    double CalcWCopyfromMatrixFrac(std::string seq) {
        
        Matrix<VectorFractionList> running = start_vec_frac.slice({0},IsingWSlices[seq[0]]);
        for (int i = 1; i < L; i++) { 
             running = running * indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]);
             // LENGTHSCALE
             // std::cout << seq[i-1] << " " << seq[i] << std::endl; 
             // std::cout << indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]).toString() << std::endl;
             //std::cout << std::endl;
             // std::cout << running.toString() << std::endl;
             //std::cout << "Matrix" << std::endl;
             //std::cout << indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]).toString() << std::endl;
             //std::cout << "running" << std::endl;
             //std::cout << running.toString() << std::endl;
        }
        std::cout << running(0,0).ls.size() << std::endl;
        //std::cout << running.toString() << std::endl;
        Matrix<MatrixFraction> sliced_end = end_vec_frac.slice(IsingWSlices[seq[L-1]],{0});
        double result = 0.0;
        for (int j = 0; j < sliced_end.rows; j++){
            result += running(0,j).Finisher(sliced_end(j,0));
        }
        return(result);
    }

    /**
     * @brief Calculate weight probability and vector complexity using matrix fractions
     * @param seq Weight sequence string
     * @return Tuple of (vector_complexity, probability)
     * 
     * Same as CalcWCopyfromMatrixFrac but also returns the complexity
     * (number of terms) in the vector fraction representation, useful
     * for monitoring computational efficiency.
     */
    std::tuple<int,double> CalcWCopyandVectorComplexity(std::string seq) {
        Matrix<VectorFractionList> running = start_vec_frac.slice({0},IsingWSlices[seq[0]]);
        for (int i = 1; i < L; i++) { 
             running = running * indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]);
        }
        Matrix<MatrixFraction> sliced_end = end_vec_frac.slice(IsingWSlices[seq[L-1]],{0});
        double result = 0.0;
        for (int j = 0; j < sliced_end.rows; j++){
            result += running(0,j).Finisher(sliced_end(j,0));
        }
        return(std::make_tuple(running(0,0).ls.size()+running(0,1).ls.size(),result));
    }

    /**
     * @brief Generate quenched probability matrices for a specific sequence
     * @param sequence Input sequence string
     * @return Vector of conditional probability matrices for each position
     * 
     * Creates position-specific probability matrices for quenched disorder
     * calculations where the sequence is fixed and we sample over fold
     * configurations. Uses backward iteration to ensure proper normalization.
     */
    std::vector<Eigen::MatrixXd> getQuenchedProbList(std::string sequence){
        std::vector<Eigen::MatrixXd> partition_list(L+1);
        std::vector<Eigen::MatrixXd> prob_list(L);

        partition_list[0] = sliceMatrix(EqSWstart,{0},IsingSSlices[sequence[0]]);
        
        for (int i = 1; i < L; i++) {
            partition_list[i]  = sliceMatrix(EqSWMatrix,IsingSSlices[sequence[i-1]],IsingSSlices[sequence[i]]);
        }
        partition_list[L] = sliceMatrix(EqSWend,IsingSSlices[sequence[L-1]],{0});

        Eigen::MatrixXd CondProb_end = RowNormalize(multiplytoColVec(partition_list[L-1],partition_list[L]));
        Eigen::VectorXd backward_iter_vec = partition_list[L-1]*partition_list[L];
        
        prob_list[L-1] = CondProb_end;
    
        for (int i = L-2; i > 0; i--) {
            prob_list[i] = RowNormalize(multiplytoColVec(partition_list[i],backward_iter_vec));
            backward_iter_vec = partition_list[i]*backward_iter_vec;
            backward_iter_vec = backward_iter_vec/backward_iter_vec.sum(); // Helps avoid blowing up 
        }
        prob_list[0] = RowNormalize(multiplytoColVec(partition_list[0],backward_iter_vec));

        return(prob_list);
    }

    /**
     * @brief Evaluate fold probability using quenched probability matrices
     * @param fold Fold configuration string
     * @param prob_list Precomputed probability matrices from getQuenchedProbList
     * @return Conditional probability P(fold|sequence)
     * 
     * Efficiently computes the conditional probability of a fold configuration
     * given a fixed sequence using precomputed probability matrices.
     */
    double evalQuenchedProbList(std::string fold, std::vector<Eigen::MatrixXd> prob_list){
        double prob = prob_list[0](CHAR_TO_INT[fold[0]]);
        for (int i = 1; i < L; i++) {
            prob = prob*prob_list[i](CHAR_TO_INT[fold[i-1]],CHAR_TO_INT[fold[i]]);
        }

        return(prob);
    }

    /**
     * @brief Sample sequence distribution close to equilibrium
     * @param tresh Relative threshold for perturbation (e.g., 0.1 for ±10%)
     * @param gen Random number generator
     * @return Perturbed sequence distribution map
     * 
     * Creates a sequence distribution by randomly perturbing the equilibrium
     * distribution within specified bounds. Useful for sensitivity analysis
     * and exploring near-equilibrium behavior.
     */
    std::unordered_map<std::string,double>  SampleCloseSequenceDist(double tresh, std::mt19937_64& gen){
        std::unordered_map<std::string,double> SCopyMap; 
        double newTotal = 0.0;
        std::uniform_real_distribution<double> dist(1-tresh, 1+tresh);
        
        for (const auto& pair : SEquilibriumMap) {
            SCopyMap[pair.first] = SEquilibriumMap[pair.first]*dist(gen);
            newTotal += SCopyMap[pair.first];
        }

        for (const auto& pair : SCopyMap) {
            SCopyMap[pair.first] = SCopyMap[pair.first]/newTotal;
        }

        return(SCopyMap);
    }

    /**
     * @brief Generate random tangent vector on probability simplex
     * @param seed Random seed for reproducibility
     * @return Random direction vector orthogonal to equilibrium distribution
     * 
     * Samples a random direction on the probability simplex that is orthogonal
     * to the equilibrium distribution. Used for systematic exploration of
     * probability space via arc walks.
     */
    std::unordered_map<std::string,double>  EquilibriumTangentSample(unsigned int seed){
        double dotProd = 0.0;

        std::unordered_map<std::string,double> initSamp;
        std::unordered_map<std::string,double> SqrtDirs; 
        std::normal_distribution<double> normal_dist(0,1);
        std::mt19937_64 gen(seed);

        for (const auto& pair : SEquilibriumMap) {
            initSamp[pair.first] = normal_dist(gen);
            dotProd += initSamp[pair.first]*sqrt(pair.second); 
        }

        double newMagSqr = 0.0;
        for (const auto& pair : SEquilibriumMap) {
            SqrtDirs[pair.first] = initSamp[pair.first] - dotProd*sqrt(pair.second);
            newMagSqr += pow(SqrtDirs[pair.first],2);
        }

        double magsqrValidate = 0.0;
        double dotValidate = 0.0;

        for (const auto& pair : SEquilibriumMap) {
            SqrtDirs[pair.first] = SqrtDirs[pair.first]/sqrt(newMagSqr); 
            dotValidate += SqrtDirs[pair.first]*sqrt(pair.second);
            magsqrValidate += pow(SqrtDirs[pair.first],2);
        }

        std::cout << "Orthogonality validation:" << dotValidate << std::endl;
        std::cout << "Sum validation:" << magsqrValidate << std::endl;

        return(SqrtDirs);
    }

    /**
     * @brief Perform arc walk on probability simplex
     * @param tangentVec Tangent direction vector (from EquilibriumTangentSample)
     * @param angle Arc angle in radians
     * @return New probability distribution after arc walk
     * 
     * Moves along a circular arc on the probability simplex starting from
     * equilibrium in the direction of tangentVec. Enables systematic
     * exploration of probability space while maintaining normalization.
     */
    std::unordered_map<std::string,double>  ArcWalk(std::unordered_map<std::string,double> tangentVec, double angle){
        std::unordered_map<std::string,double> NewProb;
        double Total = 0;
        for (const auto& pair : SEquilibriumMap) {
            NewProb[pair.first] = pow(sqrt(pair.second)*cos(angle) + tangentVec[pair.first]*sin(angle),2); 
            //std::cout << pair.second << " " << NewProb[pair.first] << std::endl;
            Total += NewProb[pair.first];
        }
        return(NewProb);
    }

    /**
     * @brief Find angular limits to stay in positive orthant
     * @param tangentVec Tangent direction vector
     * @return Pair of (lower_limit, upper_limit) angles in radians
     * 
     * Calculates the range of angles for arc walks that keep all probabilities
     * positive. Essential for ensuring physical validity during exploration.
     */
    std::pair<double,double>  PositiveOrthantLimits(std::unordered_map<std::string,double> tangentVec){
        std::unordered_map<std::string,double> NewProb;
        double lowlim = -3.142;
        double highlim = 3.142;

        double testlowlim;  
        double testhighlim;

        for (const auto& pair : SEquilibriumMap) {

            double test = atan(-sqrt(pair.second)/tangentVec[pair.first]);
            if (test<0) {
                testlowlim = test;
                testhighlim = test+3.142;
            } else {
                testhighlim = test;
                testlowlim = test-3.142;
            }
            if (testlowlim > lowlim){
                lowlim = testlowlim;
            }
            if (testhighlim < highlim){
                highlim = testhighlim;
            }
        }
        std::cout << lowlim << " " << highlim << std::endl;
        return(std::pair(lowlim,highlim));
    }

    /**
     * @brief Decompose probability distribution into equilibrium component and tangent vector
     * @param SCopyMap Input probability distribution
     * @return Pair of (tangent_vector, angle) for arc representation
     * 
     * Decomposes an arbitrary probability distribution into its projection
     * onto the equilibrium distribution and a tangent component. Enables
     * representation as an arc walk from equilibrium.
     */
    std::pair<std::unordered_map<std::string,double>,double>  GetVectorAndArc(std::unordered_map<std::string,double> SCopyMap){
        std::unordered_map<std::string,double> vector;
        double dotprod = 0.0;
        double eqMag = 0.0;
        double qMag = 0.0;
        for (const auto& pair : SEquilibriumMap) {
            dotprod += sqrt(pair.second)*sqrt(SCopyMap[pair.first]); 
            eqMag += pair.second;
            qMag += SCopyMap[pair.first];
        }
        eqMag = sqrt(eqMag);
        qMag = sqrt(qMag);
        double angle = acos(dotprod/(eqMag*qMag));
        //double checker = 0.0;
        //double checker2 = 0.0;
        for (const auto& pair : SEquilibriumMap) {
            vector[pair.first] = (sqrt(SCopyMap[pair.first])-sqrt(pair.second)*cos(angle))/sin(angle); 
            //checker += pow(vector[pair.first],2);  
            //checker2 +=  vector[pair.first]*sqrt(pair.second);
        }
        
        //std::cout << "SUM VALIDATION " << checker << std::endl;
        //std::cout << "DOT VALIDATION " << checker2 << std::endl;

        return(std::pair(vector,angle));
    }
};

/**
 * @class Ising2
 * @brief Two-state Ising model for helix-coil transitions
 * 
 * Implements a specific two-state model where each residue can be in
 * helix (0) or coil (1) conformation. Provides multiple constructor
 * variants for different parameterizations and includes specialized
 * methods for Bernoulli sequence analysis and quenched disorder studies.
 */
class Ising2 : public IsingVar {
    public:
    /**
     * @brief Constructor with explicit helix-helix and coil parameters
     * @param w00 Helix-helix interaction energy
     * @param w11 Coil-coil interaction energy  
     * @param w01 Helix-coil interaction energy
     * @param w10 Coil-helix interaction energy
     * @param v End effects parameter
     * @param l Polymer length
     */
    Ising2(double w00, double w11, double w01, double w10, double v, int l) {
        EqSWstart =  Eigen::Vector4d(1, 1, 1, 1); // Start just sums up, does not correspond to first monomer 
        EqSWend = Eigen::Vector4d(v, 1, v, 1); // End corresponds to last monomer
    
        Qstart =  Eigen::Vector2d(0, 0);
        Qend = Eigen::Vector2d(0, 0);

        EqSWMatrix.resize(4, 4);
 
        std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

        IsingSlookup = {
            {'0', '0'},
            {'1', '0'},
            {'2', '1'},
            {'3', '1'}
        };

        IsingWlookup = {
            {'0', '0'},
            {'1', '1'},
            {'2', '0'},
            {'3', '1'}
        };

        EqSWMatrix << w00, v, w01, v,
                      1,   1, 1,   1,
                      w10, v, w11, v,
                      1,   1, 1,   1;
        
        L = l;

        CalcAllEigen();
        CalcAllPartition();
    }

    /**
     * @brief Constructor with state-specific weights and couplings
     * @param w0 Weight for state 0 (helix)
     * @param w1 Weight for state 1 (coil)
     * @param c0 Coupling for state 0
     * @param c1 Coupling for state 1  
     * @param l Polymer length
     * @param placehold Placeholder parameter (unused)
     */
    Ising2(double w0, double w1, double c0, double c1, int l, std::string placehold) {
        EqSWstart =  Eigen::Vector4d(1, 1, 1, 1); // Start just sums up, does not correspond to first monomer 
        EqSWend = Eigen::Vector4d(w0, c0, w1, c1); // End corresponds to last monomer
    
        Qstart =  Eigen::Vector2d(0, 0);
        Qend = Eigen::Vector2d(0, 0);

        EqSWMatrix.resize(4, 4);
 
        std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

        IsingSlookup = {
            {'0', '0'},
            {'1', '0'},
            {'2', '1'},
            {'3', '1'}
        };

        IsingWlookup = {
            {'0', '0'},
            {'1', '1'},
            {'2', '0'},
            {'3', '1'}
        };

        EqSWMatrix << w0, w0, w0,  w0,
                      c0,  c0, c0,  c0,
                      w1,  w1, w1, w1,
                      c1,  c1, c1,  c1;
        
        L = l;

        CalcAllEigen();
        CalcAllPartition();
    }

    /**
     * @brief Constructor with diagonal helix-helix and coil-coil interactions
     * @param w00 Helix-helix interaction strength
     * @param w11 Coil-coil interaction strength
     * @param c0 Helix coupling parameter
     * @param c1 Coil coupling parameter
     * @param l Polymer length
     */
    Ising2(double w00, double w11, double c0, double c1, int l) {
        EqSWstart =  Eigen::Vector4d(1, 1, 1, 1); // Start just sums up, does not correspond to first monomer 
        EqSWend = Eigen::Vector4d(c0, c0, c1, c1); // End corresponds to last monomer
    
        Qstart =  Eigen::Vector2d(0, 0);
        Qend = Eigen::Vector2d(0, 0);

        EqSWMatrix.resize(4, 4);
 
        std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

        IsingSlookup = {
            {'0', '0'},
            {'1', '0'},
            {'2', '1'},
            {'3', '1'}
        };

        IsingWlookup = {
            {'0', '0'},
            {'1', '1'},
            {'2', '0'},
            {'3', '1'}
        };

        EqSWMatrix << w00, c0, c0,  c0,
                      c0,  c0, c0,  c0,
                      c1,  c1, w11, c1,
                      c1,  c1, c1,  c1;
        
        L = l;

        CalcAllEigen();
        CalcAllPartition();
    }

    /**
     * @brief Constructor with random parameters for testing
     * @param seed Random seed for parameter generation
     * @param l Polymer length
     * 
     * Creates a model with randomly generated parameters drawn from
     * uniform distribution [1,10]. Useful for testing and validation.
     */
     Ising2(unsigned int seed, int l) {
        
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> dist(1.0, 10.0);
        
        // End corresponds to last monomer
        EqSWstart =  Eigen::Vector4d(0, 0, 0, 0); // Start just sums up, does not correspond to first monomer 
        EqSWend = Eigen::Vector4d(0, 0, 0, 0);

        Qstart =  Eigen::Vector2d(0, 0);
        Qend = Eigen::Vector2d(0, 0);

        EqSWMatrix.resize(4, 4);
 
        std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

        IsingSlookup = {
            {'0', '0'},
            {'1', '0'},
            {'2', '1'},
            {'3', '1'}
        };

        IsingWlookup = {
            {'0', '0'},
            {'1', '1'},
            {'2', '0'},
            {'3', '1'}
        };

        // Fill the matrix with random integers
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                EqSWMatrix(i, j) = dist(gen);
            }
        }
        
        for (int i = 0; i < 4; i++) {
            EqSWstart(i) = dist(gen);
            EqSWend(i) = dist(gen);
        }
       

        std::cout << EqSWMatrix << std::endl;
        L = l;

        CalcAllEigen();
        CalcAllPartition();
    }

    /**
     * @brief Set up matrix fractions for Bernoulli sequence model
     * @param bernoulli Probability of state 0 in Bernoulli model
     * @param epsilon Numerical precision parameter
     * 
     * Initializes matrix fraction calculations for analyzing systems
     * where sequences are drawn from a Bernoulli distribution.
     */
    void getBernoulliMatrixFractions(double bernoulli, double epsilon = DEFAULT_EPSILON){
        int SWalphabetsize = IsingSlookup.size();

            // Initialize the sequence probabilities
        std::vector<double> probs = {bernoulli,1-bernoulli};
        
            // Since we're doing this for Ising2, these can be defined directly
        int wsize = 2;
        int ssize = 2;

        getIndependentMatrixFractions(SWalphabetsize,ssize, wsize, probs, epsilon);
    }

    /**
     * @brief Sample sequence and fold from quenched Bernoulli model
     * @param p Bernoulli parameter for sequence generation
     * @param gen Random number generator
     * @return Tuple of (sequence, fold, total_probability)
     * 
     * Generates a random sequence from Bernoulli(p) distribution, then
     * samples the corresponding fold configuration from the conditional
     * distribution P(fold|sequence). Returns the sequence, fold, and
     * total sampling probability.
     */
    std::tuple<std::string,std::string, double> SampleQuenchedBernoulli(double p, std::mt19937_64& gen) {
        std::bernoulli_distribution d(p);
        
        std::string sequence;
        sequence.reserve(L);
        double ps = 0.0;
        for(int i = 0; i < L; i++) {
            bool samp = d(gen);
            sequence += samp ? '0' : '1';
            ps += samp? -log(p): -log(1-p);
        }

        double pw = 0.0;
        std::string fold = "";

        std::vector<Eigen::MatrixXd> prob_list = getQuenchedProbList(sequence);

        Eigen::RowVectorXd prob_vec = prob_list[0];

        int chosen = drawFromProbVector(prob_vec,gen);
        fold += INT_TO_CHAR[chosen];
        pw -= log(prob_vec(chosen));
        
        int chosen_new;
        for (int i = 1; i <L ;i++) {
            chosen_new = drawFromProbVector(prob_list[i].row(chosen),gen);
            pw -= log(prob_list[i](chosen,chosen_new));
            chosen = chosen_new;
            fold += INT_TO_CHAR[chosen];
        }

        return(std::make_tuple(sequence,fold,ps+pw));
    }

    /**
     * @brief Generate multiple Bernoulli sequences
     * @param p Bernoulli parameter  
     * @param N Number of sequences to generate
     * @param L Length of each sequence
     * @param seed Random seed
     * @return Vector of generated sequences
     * 
     * Static method for generating multiple independent sequences
     * from Bernoulli distribution. Useful for batch processing.
     */
    static std::vector<std::string> SampleNBernoulliSequence(double p, int N, int L, unsigned int seed) {
        std::mt19937_64 gen(seed);
        std::bernoulli_distribution d(p);
        
        std::vector<std::string> sampled_sequences;
       
        for (int n = 0; n < N ; n++) {
            std::string sequence;
            sequence.reserve(L);
            double ps = 0.0;
            for(int i = 0; i < L; i++) {
                bool samp = d(gen);
                sequence += samp ? '0' : '1';
            }
            sampled_sequences.push_back(sequence);
        }
        
        return(sampled_sequences);
    }

    /**
     * @brief Generate multiple biased Bernoulli sequences
     * @param p Bernoulli parameter
     * @param N Number of sequences to generate  
     * @param L Length of each sequence
     * @param seed Random seed (also determines bias direction)
     * @return Vector of generated sequences
     * 
     * Generates sequences with bias toward '0' or '1' based on seed parity.
     * Used for testing systematic biases in sequence generation.
     */
    static std::vector<std::string> SampleNBiasedBernoulliSequence(double p, int N, int L, unsigned int seed) {
        std::mt19937_64 gen(seed);
        std::bernoulli_distribution d(p);
        
        std::vector<std::string> sampled_sequences;
        int samp_last = seed%2 ? 0 : 1;
        
        for (int n = 0; n < N ; n++) {
            std::string sequence;
            sequence.reserve(L);
            double ps = 0.0;
            for(int i = 0; i < L; i++) {
                bool samp = d(gen);
                sequence += samp ? INT_TO_CHAR[samp_last] : INT_TO_CHAR[1-samp_last];
            }
            sampled_sequences.push_back(sequence);
            std::cout << sequence <<std::endl;
        }
        
        return(sampled_sequences);
    }

    /**
     * @brief Verify matrix calculations for quenched disorder model
     * @param SCopyMap Input sequence distribution
     * @param p Bernoulli parameter for validation sampling
     * @param trials Number of Monte Carlo trials for validation
     * 
     * Cross-validates matrix-based calculations against Monte Carlo sampling
     * for quenched disorder systems. Compares theoretical predictions with
     * empirical estimates from random sampling.
     */
    void VerifyMatrixApproachQuenched(std::unordered_map<std::string,double> SCopyMap, double p, int trials) {
        std::cout << "Setting up matrices..." << std::endl;
        
        getPSWMatrices();

        std::cout << "Calculating quenched copy map..." << std::endl;
        
        std::unordered_map<std::string,double> WCopyMap; 
            WCopyMap.reserve(WEquilibriumMap.size());

        std::unordered_map<std::string,std::vector<Eigen::MatrixXd>> SProbLists;


        for (const auto& sdata : SCopyMap) {
            const auto& s_state = std::get<0>(sdata);
            SProbLists[s_state] = getQuenchedProbList(s_state);
        }

        double KL_SW = 0.0;
        double JointcopyEntropy = 0.0;
        double KL_W = 0.0;
        double FoldcopyEntropy = 0.0;

        for (const auto& swdata : EquilibriumTable) {
            const double peqSW = std::get<3>(swdata);
            const auto& s_state = std::get<1>(swdata);
            const auto& w_state = std::get<2>(swdata);
            const double foldEn = std::get<4>(swdata);
            const double s_copy_prob = SCopyMap.at(s_state);
            const double s_eq_prob = SEquilibriumMap.at(s_state);
            const double w_eq_prob = WEquilibriumMap[w_state];
            
            const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
            WCopyMap[w_state] += pcopySW;

            KL_SW += pcopySW * log(pcopySW/peqSW);
            JointcopyEntropy += -pcopySW * log(pcopySW);

            const double calculated = SCopyMap[s_state]*evalQuenchedProbList(w_state, SProbLists[s_state]);
            if (abs(calculated-pcopySW) > pow(10.0,-10)) {
                throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                    + std::to_string(abs(calculated-pcopySW)) + " OBTAINED!");
            }

            //std::cout << s_state <<"|" << w_state << "/" << pcopySW 
            //    << "/" << calculated << std::endl;
        }

        std::cout << "SW Quenched Verified, now moving on to W Quenched.." << std::endl;

        for (const auto& wdata : WCopyMap) {
            const double pcopyW = std::get<1>(wdata);
            const auto& w_state = std::get<0>(wdata);
            
            KL_W += pcopyW*log(pcopyW/WEquilibriumMap[w_state]);
            FoldcopyEntropy += -pcopyW * log(pcopyW);

            if (abs(CalcWCopyfromMatrixFrac(w_state)-pcopyW) > pow(10.0,-10)) {
                throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                    + std::to_string(abs(CalcWCopyfromMatrixFrac(w_state)-pcopyW)) + " OBTAINED!");
            }


            //std::cout << w_state << "/" << pcopyW << "/" << CalcWCopyfromMatrixFrac(w_state) << std::endl; 
            
        }

        double joint_entropy_est = 0.0;
        double fold_entropy_est = 0.0;

        std::random_device rd;
        std::mt19937_64 gen(rd());

        for (int i = 0; i < trials; i++) {
            auto sample = SampleQuenchedBernoulli(p, gen);
            joint_entropy_est += 1.0/double(trials)*std::get<2>(sample);
            // std::cout << std::get<0>(sample) << "/" << std::get<1>(sample) << "/" << std::get<2>(sample) << std::endl;
            fold_entropy_est += -1.0/double(trials)*log(CalcWCopyfromMatrixFrac(std::get<1>(sample)));
        }
        std::cout << "Estimated Joint Entropy of " << joint_entropy_est << " Versus True " << JointcopyEntropy <<std::endl;
        std::cout << "Estimated Fold Entropy of " << fold_entropy_est << " Versus True " << FoldcopyEntropy <<std::endl;
    }

    /**
     * @brief Sample entropy estimates from Bernoulli model
     * @param p Bernoulli parameter
     * @param trials Number of sampling trials
     * @param seed Random seed  
     * @return Tuple of (joint_mean, joint_stddev, fold_mean, fold_stddev)
     * 
     * Estimates joint and fold entropies through Monte Carlo sampling,
     * returning both means and standard deviations for statistical analysis.
     */
    std::tuple<double,double,double,double> SampleBernoulliEntropies(double p, int trials, int seed) {
        getBernoulliMatrixFractions(0.1);

        Eigen::VectorXd joint_entropy_vec(trials);  // your data vector
        Eigen::VectorXd fold_entropy_vec(trials);

        std::mt19937_64 gen(seed);

        for (int i = 0; i < trials; i++) {
            auto sample = SampleQuenchedBernoulli(p, gen);
            double joint_sample = std::get<2>(sample)/L;
            double fold_sample = -log(CalcWCopyfromMatrixFrac(std::get<1>(sample)))/L;
            
            joint_entropy_vec(i) = joint_sample;
            fold_entropy_vec(i) = fold_sample;
        }

        double joint_mean = joint_entropy_vec.mean();
        double joint_stddev = sqrt((joint_entropy_vec.array() - joint_mean).square().sum() / (joint_entropy_vec.size() - 1));
        double fold_mean = fold_entropy_vec.mean();
        double fold_stddev = sqrt((fold_entropy_vec.array() - fold_mean).square().sum() / (fold_entropy_vec.size() - 1));
        return(std::make_tuple(joint_mean,joint_stddev,fold_mean,fold_stddev));
    }
};

/**
 * @class Ising2S3F
 * @brief Extended Ising model with 2 sequence states and 3 fold states
 * 
 * Implements a more complex model where each residue can be in one of
 * two sequence states but three different fold conformations (helix,
 * antihelix, coil). Allows for more detailed analysis of secondary
 * structure preferences.
 */
class Ising2S3F : public IsingVar {
    public:
    /**
     * @brief Constructor for 2-sequence, 3-fold state model
     * @param w00 Helix-helix interaction energy for sequence state 0
     * @param w11 Helix-helix interaction energy for sequence state 1  
     * @param a01 Antihelix interaction energy (state 0 to 1)
     * @param a10 Antihelix interaction energy (state 1 to 0)
     * @param v End effects parameter
     * @param l Polymer length
     * 
     * Creates a model with 6 combined states:
     * - States 0,1,2: Sequence state 0 with fold states helix, antihelix, coil
     * - States 3,4,5: Sequence state 1 with fold states helix, antihelix, coil
     */
    Ising2S3F(double w00, double w11, double a01, double a10, double v, int l) { 
        IsingSlookup = {
            {'0', '0'},
            {'1', '0'},
            {'2', '0'},
            {'3', '1'},
            {'4', '1'},
            {'5', '1'},
        };

        IsingWlookup = {
            {'0', '0'},
            {'1', '1'},
            {'2', '2'},
            {'3', '0'},
            {'4', '1'},
            {'5', '2'}
        };
        
        EqSWstart =  Eigen::VectorXd(6);
            EqSWstart << 1, 1, 1, 1, 1, 1; // Start corresponds to first monomer
        EqSWend = Eigen::VectorXd(6); // End just sums up, does NOT correspond to last monomer
             EqSWend << v, v, 1, v, v, 1;
        EqSWMatrix.resize(6, 6);

        //Fold idx: helix 0, antihelix 1, coil 2; 
        //Order: 0 helix, 0 antihelix, 0 coil, 1 helix, 1 antihelix, 1 coil;
        EqSWMatrix << w00,  v, v,   v,   v, v,
                      v,    v, v,   v, a01, v, 
                      1,    1, 1,   1,   1, 1,
                      v,    v, v, w11,   v, v,
                      v,  a10, v,   v,   v, v,
                      1,    1, 1,   1,   1, 1;

        L = l;

        CalcAllEigen();
        CalcAllPartition();
    }
};
