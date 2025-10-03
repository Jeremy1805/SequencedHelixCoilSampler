#ifndef EQUILIBRIUMPARTITIONMAPGENERATOR_H
#define EQUILIBRIUMPARTITIONMAPGENERATOR_H

#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <iomanip>
#include <utility>
#include <unordered_map>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <array>

#ifdef _MSC_VER
    #define __builtin_expect(expr, expected) (expr)
#endif

/**
 * @class EquilibriumPartitionMapGenerator
 * @brief High-performance generator for complete equilibrium partition tables
 * 
 * This class provides optimized methods for exhaustively enumerating all possible
 * sequence-weight combinations in protein folding models and calculating their
 * equilibrium probabilities. Uses aggressive optimization techniques including
 * cache-friendly memory access, branch prediction hints, and loop unrolling.
 */
class EquilibriumPartitionMapGenerator {
    /// Precomputed character conversion table for performance
    static constexpr std::array<char, 16> INT_TO_CHAR = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f'
    };
    
    /// Precomputed integer lookup table initialized at compile time
    static constexpr std::array<int, 128> CHAR_TO_INT = []() {
        std::array<int, 128> arr{};
        for (int i = 0; i < 10; ++i) arr['0' + i] = i;
        for (int i = 0; i < 6; ++i) arr['a' + i] = 10 + i;
        return arr;
    }();

public:
    /**
     * @struct PartitionState
     * @brief Memory-aligned state container for partition calculations
     * 
     * Stores the current statistical weight value and last character state
     * with 16-byte alignment for optimal memory access patterns.
     */
    struct alignas(16) PartitionState {
        double value;     ///< Current statistical weight
        char lastChar;    ///< Last character in the sequence
        
        /// Constructor with default values
        constexpr PartitionState(double v = 0.0, char c = '0') noexcept 
            : value(v), lastChar(c) {}
    };

    /**
     * @brief Fast integer-to-character conversion
     * @param i Integer index (0-15)
     * @return Corresponding character
     * 
     * Uses precomputed lookup table for O(1) conversion without branches.
     */
    static inline char intToChar(int i) noexcept {
        return INT_TO_CHAR[i];
    }
    
    /**
     * @brief Fast character-to-integer conversion  
     * @param c Character to convert
     * @return Corresponding integer index
     * 
     * Uses precomputed lookup table for O(1) conversion without branches.
     */
    static inline int charToInt(char c) noexcept {
        return CHAR_TO_INT[c];
    }

    /**
     * @brief Optimized recursive partition function enumeration
     * @param results Output vector for (sw_string, s_string, w_string, probability, energy) tuples
     * @param current Current sequence-weight string being built
     * @param sequence Current sequence string being built  
     * @param fold Current fold string being built
     * @param sequence_lookup Array mapping SW states to sequence states
     * @param fold_lookup Array mapping SW states to fold states
     * @param sequence_hash Output map for sequence marginal probabilities
     * @param fold_hash Output map for fold marginal probabilities
     * @param validation Running sum for probability validation
     * @param position Current position in sequence (counting backwards)
     * @param parentState Current partition state from parent call
     * @param weightMatrix Transfer matrix for state transitions
     * @param start Starting probability vector
     * @param end Ending probability vector
     * @param normalization Partition function for probability normalization
     * @param alphabetSize Size of state alphabet
     * @param length Total sequence length
     * 
     * This is the core recursive function that generates all possible sequences
     * by building them backwards from the end. Uses aggressive optimizations:
     * - Branch prediction hints (__builtin_expect)
     * - Loop unrolling (#pragma GCC unroll)
     * - Array lookups instead of map lookups
     * - Memory-efficient string operations
     */
    static inline void generateRecursiveTail(
        std::vector<std::tuple<std::string, std::string, std::string, double, double>>& results,
        std::string& current,
        std::string& sequence,
        std::string& fold,
        const int* sequence_lookup,
        const int* fold_lookup,
        std::unordered_map<std::string, double>& sequence_hash,
        std::unordered_map<std::string, double>& fold_hash,
        double& validation,
        const int position,
        const PartitionState& parentState,
        const Eigen::MatrixXd& weightMatrix,
        const Eigen::RowVectorXd& start,
        const Eigen::VectorXd& end,
        const double normalization,
        const int alphabetSize,
        const int length);

    /**
     * @brief Generate complete partition table with probability calculations
     * @param results Output vector for complete enumeration results
     * @param sequence_hash Output map for sequence marginal probabilities  
     * @param fold_hash Output map for fold marginal probabilities
     * @param validation Output validation sum (should equal 1.0)
     * @param length Sequence length
     * @param weightMatrix Transfer matrix for statistical weights
     * @param start Starting probability vector
     * @param end Ending probability vector  
     * @param normalization Partition function value
     * @param alphabetSize Size of combined state alphabet
     * @param sequence_lookup Map from combined states to sequence states
     * @param fold_lookup Map from combined states to fold states
     * 
     * Main entry point for partition table generation. Performs setup and
     * optimization before calling the recursive enumeration function:
     * - Converts maps to arrays for faster lookup
     * - Pre-allocates memory with size estimates
     * - Initializes strings with exact required length
     * - Calls optimized recursive function
     * 
     * The results vector contains tuples of:
     * - Combined sequence-weight string
     * - Pure sequence string  
     * - Pure fold string
     * - Equilibrium probability P(s,w)
     * - Statistical mechanical energy -ln(weight)
     */
    static void generateWithPartition(
            std::vector<std::tuple<std::string, std::string, std::string, double, double>>& results,
            std::unordered_map<std::string,double>& sequence_hash,
            std::unordered_map<std::string,double>& fold_hash,
            double& validation,
            const int length,
            const Eigen::MatrixXd& weightMatrix,
            const Eigen::RowVectorXd& start,
            const Eigen::VectorXd& end,
            const double normalization,
            const int alphabetSize,
            const std::unordered_map<char,char>& sequence_lookup,
            const std::unordered_map<char,char>& fold_lookup);
};

#endif // EQUILIBRIUMPARTITIONMAPGENERATOR_H
