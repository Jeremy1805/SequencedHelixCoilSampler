#include "EquilibriumPartitionMapGenerator.h"

#ifdef _MSC_VER
    #define __builtin_expect(expr, expected) (expr)
#endif

// Static member definitions
constexpr std::array<char, 16> EquilibriumPartitionMapGenerator::INT_TO_CHAR;
constexpr std::array<int, 128> EquilibriumPartitionMapGenerator::CHAR_TO_INT;

void EquilibriumPartitionMapGenerator::generateRecursiveTail(
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
    const int length) {
    
    // Base case: reached beginning of sequence
    if (__builtin_expect(position < 0, 0)) {
        const int lastIndex = charToInt(parentState.lastChar);
        const double prob = parentState.value/normalization * start(lastIndex);
        const double energy = -log(parentState.value);
        validation += prob;
        
        results.emplace_back(current, sequence, fold, prob, energy);
        sequence_hash[sequence] += prob;
        fold_hash[fold] += prob;
        return;
    }
    
    // Avoid branches in the main loop
    const bool isLastPosition = position == length-1;
    const double baseValue = isLastPosition ? 1.0 : parentState.value;
    const int parentIndex = charToInt(parentState.lastChar);
    
    // Unroll small loops for better performance
    #pragma GCC unroll 8  
    for (int i = 0; i < alphabetSize; ++i) {
        const char currentChar = intToChar(i);
        current[position] = currentChar;
        sequence[position] = intToChar(sequence_lookup[i]);
        fold[position] = intToChar(fold_lookup[i]);
        
        double newValue;
        if (!isLastPosition) {
            newValue = baseValue * weightMatrix(i, parentIndex);
        } else {
            newValue = baseValue * end(i);
        }
        
        // Recursive call for next position
        generateRecursiveTail(
            results, current, sequence, fold, sequence_lookup, fold_lookup,
            sequence_hash, fold_hash, validation, position - 1,
            PartitionState(newValue, currentChar),
            weightMatrix, start, end, normalization, alphabetSize, length
        );
    }
}

void EquilibriumPartitionMapGenerator::generateWithPartition(
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
        const std::unordered_map<char,char>& fold_lookup) {
    
    // Convert lookup maps to arrays for faster access during recursion
    int sequence_arr[16] = {0};
    int fold_arr[16] = {0};
    for (int i = 0; i < alphabetSize; ++i) {
        char c = intToChar(i);
        sequence_arr[i] = charToInt(sequence_lookup.at(c));
        fold_arr[i] = charToInt(fold_lookup.at(c));
    }
    
    // Pre-calculate expected size for memory efficiency
    const size_t expectedSize = std::pow(alphabetSize, length);  
    
    // Reserve memory to avoid repeated allocations
    results.reserve(expectedSize);
    
    // Pre-allocate strings with exact size needed (no reallocation)
    std::string current(length, '0');
    std::string sequence(length, '0');
    std::string fold(length, '0');
    
    // Reserve hash maps with reasonable size estimates
    sequence_hash.reserve(std::pow(alphabetSize/2, length));
    fold_hash.reserve(std::pow(alphabetSize/2, length));

    // Initialize validation sum
    validation = 0.0;
    
    // Start recursive generation from the last position
    generateRecursiveTail(
        results, current, sequence, fold,
        sequence_arr, fold_arr, sequence_hash, fold_hash,
        validation, length - 1, PartitionState(1.0, intToChar(alphabetSize-1)),
        weightMatrix, start, end, normalization, alphabetSize, length
    );
}
