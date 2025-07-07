#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <random>    
#include <Eigen/Dense>
#include <tuple>
#include <sstream>

/**
 * @class Timer
 * @brief High-resolution timing utility for performance measurement
 * 
 * RAII-style timer that automatically prints elapsed time when destroyed.
 * Useful for profiling and benchmarking different code sections.
 */
class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    std::string name;        ///< Descriptive name for this timer
    TimePoint start_time;    ///< Time when timer was created/reset

public:
    /**
     * @brief Constructor starts timing immediately
     * @param name Descriptive name for timer output
     */
    Timer(const std::string& name = "Timer");
    
    /**
     * @brief Get current elapsed time without stopping timer
     * @return Elapsed seconds as double
     */
    double elapsed() const;
    
    /**
     * @brief Reset timer to current time
     */
    void reset();
    
    /**
     * @brief Destructor automatically prints elapsed time
     * 
     * This enables RAII-style timing where simply creating a Timer
     * object at the beginning of a scope will print timing when
     * the scope exits.
     */
    ~Timer();
};

// Function declarations

/**
 * @brief Escape tab characters in string fields
 * @param str Input string that may contain tabs
 * @return String with tabs replaced by spaces
 * 
 * Helper function to prevent field separation issues in TSV files
 * when string data contains tab characters.
 */
std::string escapeTab(const std::string& str);

/**
 * @brief Element-wise multiply matrix rows by vector
 * @param matrix Input matrix 
 * @param vector Input vector (must have same size as matrix columns)
 * @return Matrix with each row multiplied element-wise by vector
 * 
 * Performs the operation: result[i,j] = matrix[i,j] * vector[j] for all i,j.
 * Useful for applying position-dependent weights to transfer matrices.
 */
Eigen::MatrixXd multiplytoColVec(const Eigen::MatrixXd& matrix, const Eigen::VectorXd& vector);

/**
 * @brief Normalize matrix rows to sum to 1
 * @param matrix Input matrix
 * @return Matrix with each row normalized to unit sum
 * 
 * Converts each row to a probability distribution by dividing by row sum.
 * Includes safeguard against division by zero for numerical stability.
 */
Eigen::MatrixXd RowNormalize(const Eigen::MatrixXd& matrix);

/**
 * @brief Element-wise multiply matrix columns by row vector  
 * @param rowvec Input row vector
 * @param matrix Input matrix (rows must match rowvec columns)
 * @return Matrix with each column multiplied element-wise by rowvec
 * 
 * Performs the operation: result[i,j] = rowvec[i] * matrix[i,j] for all i,j.
 * Useful for applying starting state weights to transfer matrices.
 */
Eigen::MatrixXd multiplyfromRowVec(const Eigen::RowVectorXd& rowvec, const Eigen::MatrixXd& matrix);

/**
 * @brief Extract submatrix using row and column index vectors
 * @param M Input matrix
 * @param rows Vector of row indices to extract
 * @param cols Vector of column indices to extract  
 * @return Submatrix with specified rows and columns
 * 
 * Utility function for slicing matrices with arbitrary (possibly non-contiguous)
 * index sets. Essential for quenched disorder calculations where only certain
 * states are accessible for a given sequence.
 */
Eigen::MatrixXd sliceMatrix(const Eigen::MatrixXd& M, 
                           const std::vector<size_t>& rows,
                           const std::vector<size_t>& cols);

/**
 * @brief Calculate Bernoulli probability for binary string
 * @param s Binary string ('0' and '1' characters)
 * @param p Probability of observing '0'
 * @return Probability P(s) under independent Bernoulli model
 * 
 * Computes ∏ᵢ p^{s_i} (1-p)^{1-s_i} where s_i ∈ {0,1}.
 * Used for generating reference distributions in sequence analysis.
 */
double bernoulliStringProb(const std::string& s, double p);

/**
 * @brief Sample from discrete probability distribution
 * @param probs Row vector of probabilities (must sum to 1)
 * @param gen Random number generator 
 * @return Index of sampled element
 * 
 * Uses inverse transform sampling to draw from discrete distribution.
 * Essential for Monte Carlo sampling of states according to their
 * equilibrium or conditional probabilities.
 */
int drawFromProbVector(const Eigen::RowVectorXd& probs, std::mt19937_64& gen);

// Template functions - keep implementations in header

/**
 * @brief Save vector of tuples to tab-separated value file
 * @tparam T Variadic template for tuple types
 * @param data Vector of tuples containing data to save
 * @param filename Output filename
 * @param headers Optional vector of column headers
 * 
 * Template function that can save any vector of tuples to a TSV file.
 * Automatically handles different tuple types and ensures high precision
 * for scientific data. Uses scientific notation for floating-point values.
 */
template<typename... T>
void saveTuplesToCSV(const std::vector<std::tuple<T...>>& data, 
                     const std::string& filename,
                     const std::vector<std::string>& headers = {}) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    file.precision(15);
    file << std::scientific;

    if (!headers.empty()) {
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << "\t";
        }
        file << "\n";
    }

    for (const auto& tuple : data) {
        std::apply([&file](const auto&... args) {
            size_t idx = 0;
            ((file << args << (++idx != sizeof...(T) ? "\t" : "")), ...);
        }, tuple);
        file << "\n";
    }

    file.close();
}

/**
 * @brief Save unordered_map to tab-separated value file
 * @tparam K Key type
 * @tparam V Value type  
 * @param map Input map to save
 * @param filename Output filename
 * @return True if successful, false otherwise
 * 
 * Template function for saving any unordered_map to a TSV file.
 * Handles arbitrary key and value types by converting them to strings.
 * Includes proper tab escaping to prevent format corruption.
 */
template<typename K, typename V>
bool saveMapToTSV(const std::unordered_map<K, V>& map, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    file << "Key\tValue\n";
    
    for (const auto& pair : map) {
        std::ostringstream keyStr, valueStr;
        keyStr << pair.first;
        valueStr << pair.second;
        
        file << escapeTab(keyStr.str()) << "\t" 
             << escapeTab(valueStr.str()) << "\n";
    }
    
    file.close();
    return true;
}

#endif // UTILITIES_H
