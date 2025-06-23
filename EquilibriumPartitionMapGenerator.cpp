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

class EquilibriumPartitionMapGenerator {
    // Cache for character conversion to avoid repeated calculations
    static constexpr std::array<char, 16> INT_TO_CHAR = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f'
    };
    
    static constexpr std::array<int, 128> CHAR_TO_INT = []() {
        std::array<int, 128> arr{};
        for (int i = 0; i < 10; ++i) arr['0' + i] = i;
        for (int i = 0; i < 6; ++i) arr['a' + i] = 10 + i;
        return arr;
    }();

public:
    struct alignas(16) PartitionState {  // Align for better memory access
        double value;
        char lastChar;
        
        constexpr PartitionState(double v = 0.0, char c = '0') noexcept 
            : value(v), lastChar(c) {}
    };

    static inline char intToChar(int i) noexcept {
        return INT_TO_CHAR[i];
    }
    
    static inline int charToInt(char c) noexcept {
        return CHAR_TO_INT[c];
    }

    static inline void generateRecursiveTail(
        std::vector<std::tuple<std::string, std::string, std::string, double, double>>& results,
        std::string& current,
        std::string& sequence,
        std::string& fold,
        const int* sequence_lookup,  // Array lookup instead of map
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
        
        if (__builtin_expect(position < 0, 0)) {  // Branch prediction hint
            const int lastIndex = charToInt(parentState.lastChar);
            const double prob = parentState.value/normalization * start(lastIndex);
            const double energy = -log(parentState.value);
            validation += prob;
            
            results.emplace_back(current, sequence, fold, prob, energy);
            sequence_hash[sequence] += prob;
            fold_hash[fold] += prob;
            return;
        }
        
        // Avoid branches in the loop
        const bool isLastPosition = position == length-1;
        const double baseValue = isLastPosition ? 1.0 : parentState.value;
        const int parentIndex = charToInt(parentState.lastChar);
        
        #pragma GCC unroll 8  // Unroll small loops
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
            
            generateRecursiveTail(
                results, current, sequence, fold, sequence_lookup, fold_lookup,
                sequence_hash, fold_hash, validation, position - 1,
                PartitionState(newValue, currentChar),
                weightMatrix, start, end, normalization, alphabetSize, length
            );
        }
    }

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
            const std::unordered_map<char,char>& fold_lookup) {
        // results is sw,s,w,prob(s,w)
        // Convert lookup maps to arrays for faster access
        int sequence_arr[16] = {0};
        int fold_arr[16] = {0};
        for (int i = 0; i < alphabetSize; ++i) {
            char c = intToChar(i);
            sequence_arr[i] = charToInt(sequence_lookup.at(c));
            fold_arr[i] = charToInt(fold_lookup.at(c));
        }
        
        const size_t expectedSize = std::pow(alphabetSize, length);  
        
        results.reserve(expectedSize);
        
        // Pre-allocate strings with exact size needed
        std::string current(length, '0');
        std::string sequence(length, '0');
        std::string fold(length, '0');
        
        sequence_hash.reserve(std::pow(alphabetSize/2, length));
        fold_hash.reserve(std::pow(alphabetSize/2, length));

        validation = 0.0;
        generateRecursiveTail(
            results, current, sequence, fold,
            sequence_arr, fold_arr, sequence_hash, fold_hash,
            validation, length - 1, PartitionState(1.0, intToChar(alphabetSize-1)),
            weightMatrix, start, end, normalization, alphabetSize, length
        );
    }
};