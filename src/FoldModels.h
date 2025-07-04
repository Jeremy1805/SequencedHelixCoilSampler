#ifndef FOLDMODELS_H
#define FOLDMODELS_H

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
#include <array>
#include "CustomMatrix.h"

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
     */
    double GetEigen(Eigen::MatrixXd weightMatrix);

    /**
     * @brief Calculate numerical partition function using matrix multiplication
     * @param start Starting probability vector
     * @param end Ending probability vector  
     * @param weightMatrix Transfer matrix
     * @return Partition function value
     */
    double getNumPartition(Eigen::RowVectorXd start, Eigen::VectorXd end, Eigen::MatrixXd weightMatrix);

    /**
     * @brief Calculate partition function using dominant eigenvalue approximation
     * @return Eigenvalue-based partition function
     */
    double eigenPartition() const;

    /**
     * @brief Calculate free energy per monomer from eigenvalue
     * @return Free energy G = ln(λ_max)
     */
    double eigenG() const;

    /**
     * @brief Calculate free energy per monomer from numerical partition function
     * @return Free energy G = ln(Z)/L
     */
    double Gnum() const;

    /**
     * @brief Calculate all partition functions for the current model
     */
    void CalcAllPartition();

    /**
     * @brief Calculate all eigenvalues for the current model
     */
    void CalcAllEigen();

    /**
     * @brief Generate probability matrices for sequence-weight calculations
     */
    void getPSWMatrices();

    /**
     * @brief Calculate probability of a sequence-weight combination using precomputed matrices
     * @param seq Sequence string in internal alphabet
     * @return Probability P(sequence, weight)
     */
    double CalcPSWfromMatrices(std::string seq);

    /**
     * @brief Calculate probability of a sequence using sequence mapping matrices
     * @param seq Sequence string
     * @return Marginal probability P(sequence)
     */
    double CalcSfromMatrices(std::string seq);

    /**
     * @brief Calculate probability of a weight (fold) configuration using weight mapping matrices
     * @param seq Sequence string (used to map to weight states)
     * @return Marginal probability P(weight)
     */
    double CalcWfromMatrices(std::string seq);
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
    // SSlices and Wslices are more convenient when calculating probabilities with Matrix Fractions 
    std::unordered_map<char, std::vector<size_t>> IsingSSlices; ///< Sequence state index slices
    std::unordered_map<char, std::vector<size_t>> IsingWSlices; ///< Weight state index slices

    /**
     * @brief Extract sequence from combined sequence-weight string
     * @param SW Combined sequence-weight string
     * @return Pure sequence string
     */
    std::string GetS(std::string SW);

    /**
     * @brief Extract weight (fold) configuration from combined sequence-weight string
     * @param SW Combined sequence-weight string  
     * @return Pure weight string
     */
    std::string GetW(std::string SW);

    /**
     * @brief Generate complete equilibrium distribution table
     */
    void GetEquilibrumTable();

    /**
     * @brief Print the complete equilibrium table to console
     */
    void printEquilibriumTable();

    /**
     * @brief Print sequence marginal distribution to console
     */
    void printSEquilibriumMap();

    /**
     * @brief Print weight (fold) marginal distribution to console
     */
    void printWEquilibriumMap();

    /**
     * @brief Print all equilibrium distributions and validation
     */
    void printAll();

    /**
     * @brief Count number of helix residues in a fold configuration
     * @param fold Fold string where '0' represents helix
     * @return Number of helix residues
     */
    static int countHelix(std::string fold);

    /**
     * @brief Calculate Bernoulli probability for a binary sequence
     * @param Prob0 Probability of observing '0' 
     * @param sequence Binary sequence string
     * @return Probability under Bernoulli model
     */
    static double GetBernoulliProb(double Prob0, std::string sequence);

    /**
     * @brief Generate probability distribution over all binary sequences using Bernoulli model
     * @param Prob0 Probability of observing '0' at each position
     * @param len Length of sequences
     * @return Map from sequences to their Bernoulli probabilities
     */
    static std::unordered_map<std::string,double> GenerateBernoulliMap(double Prob0, int len);

    /**
     * @brief Generate random probability distribution over binary sequences
     * @param seed Random seed for reproducibility
     * @param len Length of sequences
     * @return Map from sequences to random probabilities (properly normalized)
     */
    static std::unordered_map<std::string,double> SampleRandomProb(unsigned int seed, int len);

    /**
     * @brief Generate probability distribution from template sequences with error model
     * @param template_ls List of template sequences
     * @param err_rate Error rate for deviations from templates
     * @return Map from sequences to template-based probabilities
     */
    static std::unordered_map<std::string,double> GenerateFromUniformTemplate(std::vector<std::string> template_ls, double err_rate);

    /**
     * @brief Calculate folding energy for a sequence-weight configuration
     * @param foldseq Combined sequence-weight string
     * @return Total folding energy
     */
    double CalcFoldEnergy(std::string foldseq);

    /**
     * @brief Comprehensive analysis of sequence distribution vs equilibrium
     * @param SCopyMapIn Input sequence probability distribution
     * @return Tuple containing 13 thermodynamic and information-theoretic quantities
     */
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> 
        Results(std::unordered_map<std::string,double> SCopyMapIn);

    /**
     * @brief Same as Results() but also saves fold distributions to files
     * @param SCopyMapIn Input sequence probability distribution
     * @param filename Base filename for output files
     * @return Same tuple as Results()
     */
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> 
        ResultsSaveFoldMap(std::unordered_map<std::string,double> SCopyMapIn, std::string filename);

    /**
     * @brief Verify matrix calculation methods against direct enumeration
     */
    void VerifyMatrixApproach();

    /**
     * @brief Set up sequence probability mapping matrices
     */
    void getSEqMatrices();

    /**
     * @brief Set up weight probability mapping matrices
     */
    void getWEqMatrices();

    /**
     * @brief Set up matrix fraction structures for independent sequence model
     * @param SWalphabetsize Size of combined sequence-weight alphabet
     * @param ssize Number of sequence states
     * @param wsize Number of weight states  
     * @param probs Probability vector for sequence states
     * @param epsilon Numerical precision parameter
     */
    void getIndependentMatrixFractions(int SWalphabetsize, int ssize, int wsize, std::vector<double> probs, double epsilon = DEFAULT_EPSILON);

    /**
     * @brief Calculate weight probability using matrix fractions
     * @param seq Weight sequence string
     * @return Probability P(weight) under independent sequence model
     */
    double CalcWCopyfromMatrixFrac(std::string seq);

    /**
     * @brief Calculate weight probability and vector complexity using matrix fractions
     * @param seq Weight sequence string
     * @return Tuple of (vector_complexity, probability)
     */
    std::tuple<int,double> CalcWCopyandVectorComplexity(std::string seq);

    /**
     * @brief Generate quenched probability matrices for a specific sequence
     * @param sequence Input sequence string
     * @return Vector of conditional probability matrices for each position
     */
    std::vector<Eigen::MatrixXd> getQuenchedProbList(std::string sequence);

    /**
     * @brief Evaluate fold probability using quenched probability matrices
     * @param fold Fold configuration string
     * @param prob_list Precomputed probability matrices from getQuenchedProbList
     * @return Conditional probability P(fold|sequence)
     */
    double evalQuenchedProbList(std::string fold, std::vector<Eigen::MatrixXd> prob_list);

    /**
     * @brief Sample sequence distribution close to equilibrium
     * @param tresh Relative threshold for perturbation (e.g., 0.1 for ±10%)
     * @param gen Random number generator
     * @return Perturbed sequence distribution map
     */
    std::unordered_map<std::string,double> SampleCloseSequenceDist(double tresh, std::mt19937_64& gen);

    /**
     * @brief Generate random tangent vector on probability simplex
     * @param seed Random seed for reproducibility
     * @return Random direction vector orthogonal to equilibrium distribution
     */
    std::unordered_map<std::string,double> EquilibriumTangentSample(unsigned int seed);

    /**
     * @brief Perform arc walk on probability simplex
     * @param tangentVec Tangent direction vector (from EquilibriumTangentSample)
     * @param angle Arc angle in radians
     * @return New probability distribution after arc walk
     */
    std::unordered_map<std::string,double> ArcWalk(std::unordered_map<std::string,double> tangentVec, double angle);

    /**
     * @brief Find angular limits to stay in positive orthant
     * @param tangentVec Tangent direction vector
     * @return Pair of (lower_limit, upper_limit) angles in radians
     */
    std::pair<double,double> PositiveOrthantLimits(std::unordered_map<std::string,double> tangentVec);

    /**
     * @brief Decompose probability distribution into equilibrium component and tangent vector
     * @param SCopyMap Input probability distribution
     * @return Pair of (tangent_vector, angle) for arc representation
     */
    std::pair<std::unordered_map<std::string,double>,double> GetVectorAndArc(std::unordered_map<std::string,double> SCopyMap);
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
     * @brief Direct matrix constructor for Ising2 model
     * 
     * Creates an Ising2 model instance with explicitly provided transfer matrix
     * and boundary vectors, bypassing the parameter-based constructors.
     * 
     * @param transferMatrix 4x4 transfer matrix for sequence-weight state transitions
     *                       Row i, Column j = probability of transitioning from state i to state j
     *                       States ordered as: [seq0-fold0, seq0-fold1, seq1-fold0, seq1-fold1]
     * @param startVector 4-element starting probability vector
     *                    Element i = probability of starting in state i
     *                    Should sum to a positive value
     * @param endVector 4-element ending weight vector  
     *                  Element i = terminal weight for ending in state i
     *                  Represents boundary conditions at the chain terminus
     * @param polymerLength Number of monomers in the polymer chain
     */
    Ising2(const Eigen::MatrixXd& transferMatrix, const Eigen::RowVectorXd& startVector, const Eigen::VectorXd& endVector, int l);
    
    /**
     * @brief Constructor with explicit helix-helix and coil parameters
     * @param w00 Helix-helix interaction energy for 00
     * @param w11 Coil-coil interaction energy  
     * @param w01 Helix-coil interaction energy
     * @param w10 Coil-helix interaction energy
     * @param v End effects parameter
     * @param l Polymer length
     */
    Ising2(double w00, double w11, double w01, double w10, double v, int l);

    /**
     * @brief Constructor with state-specific weights and couplings
     * @param w0 Weight for state 0 (helix)
     * @param w1 Weight for state 1 (coil)
     * @param c0 Coupling for state 0
     * @param c1 Coupling for state 1  
     * @param l Polymer length
     * @param placehold Placeholder parameter (unused)
     */
    Ising2(double w0, double w1, double c0, double c1, int l, std::string placehold);

    /**
     * @brief Constructor with diagonal helix-helix and coil-coil interactions
     * @param w00 Helix-helix interaction strength
     * @param w11 Coil-coil interaction strength
     * @param c0 Helix coupling parameter
     * @param c1 Coil coupling parameter
     * @param l Polymer length
     */
    Ising2(double w00, double w11, double c0, double c1, int l);

    /**
     * @brief Constructor with random parameters for testing
     * @param seed Random seed for parameter generation
     * @param l Polymer length
     */
    Ising2(unsigned int seed, int l);

    /**
     * @brief Set up matrix fractions for Bernoulli sequence model
     * @param bernoulli Probability of state 0 in Bernoulli model
     * @param epsilon Numerical precision parameter
     */
    void getBernoulliMatrixFractions(double bernoulli, double epsilon = DEFAULT_EPSILON);

    /**
     * @brief Sample sequence and fold from quenched Bernoulli model
     * @param p Bernoulli parameter for sequence generation
     * @param gen Random number generator
     * @return Tuple of (sequence, fold, total_probability)
     */
    std::tuple<std::string,std::string, double> SampleQuenchedBernoulli(double p, std::mt19937_64& gen);

    /**
     * @brief Generate multiple Bernoulli sequences
     * @param p Bernoulli parameter  
     * @param N Number of sequences to generate
     * @param L Length of each sequence
     * @param seed Random seed
     * @return Vector of generated sequences
     */
    static std::vector<std::string> SampleNBernoulliSequence(double p, int N, int L, unsigned int seed);

    /**
     * @brief Generate multiple biased Bernoulli sequences
     * @param p Bernoulli parameter
     * @param N Number of sequences to generate  
     * @param L Length of each sequence
     * @param seed Random seed (also determines bias direction)
     * @return Vector of generated sequences
     */
    static std::vector<std::string> SampleNBiasedBernoulliSequence(double p, int N, int L, unsigned int seed);

    /**
     * @brief Verify matrix calculations for quenched disorder model
     * @param SCopyMap Input sequence distribution
     * @param p Bernoulli parameter for validation sampling
     * @param trials Number of Monte Carlo trials for validation
     */
    void VerifyMatrixApproachQuenched(std::unordered_map<std::string,double> SCopyMap, double p, int trials);

    /**
     * @brief Sample entropy estimates from Bernoulli model
     * @param p Bernoulli parameter
     * @param trials Number of sampling trials
     * @param seed Random seed  
     * @return Tuple of (joint_mean, joint_stddev, fold_mean, fold_stddev)
     */
    std::tuple<double,double,double,double> SampleBernoulliEntropies(double p, int trials, int seed);
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
     */
    Ising2S3F(double w00, double w11, double a01, double a10, double v, int l);
};

#endif // FOLDMODELS_H
