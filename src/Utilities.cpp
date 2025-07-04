#include "Utilities.h"

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
    Timer(const std::string& name = "Timer") : name(name), start_time(Clock::now()) {}

    /**
     * @brief Get current elapsed time without stopping timer
     * @return Elapsed seconds as double
     */
    double elapsed() const {
        Duration elapsed = Clock::now() - start_time;
        return elapsed.count();
    }

    /**
     * @brief Reset timer to current time
     */
    void reset() {
        start_time = Clock::now();
    }

    /**
     * @brief Destructor automatically prints elapsed time
     * 
     * This enables RAII-style timing where simply creating a Timer
     * object at the beginning of a scope will print timing when
     * the scope exits.
     */
    ~Timer() {
        Duration elapsed = Clock::now() - start_time;
        std::cout << name << ": " << std::fixed << std::setprecision(6) 
                  << elapsed.count() << " seconds\n";
    }
};

/**
 * @brief Escape tab characters in string fields
 * @param str Input string that may contain tabs
 * @return String with tabs replaced by spaces
 * 
 * Helper function to prevent field separation issues in TSV files
 * when string data contains tab characters.
 */
std::string escapeTab(const std::string& str) {
    bool needsQuotes = str.find('\t') != std::string::npos || 
                      str.find('\n') != std::string::npos;
    
    if (!needsQuotes) {
        return str;
    }
    
    // Replace tabs with spaces to prevent field separation issues
    std::string escaped = str;
    size_t pos = 0;
    while ((pos = escaped.find('\t', pos)) != std::string::npos) {
        escaped.replace(pos, 1, " ");
        pos++;
    }
    return escaped;
}

/**
 * @brief Element-wise multiply matrix rows by vector
 * @param matrix Input matrix 
 * @param vector Input vector (must have same size as matrix columns)
 * @return Matrix with each row multiplied element-wise by vector
 * 
 * Performs the operation: result[i,j] = matrix[i,j] * vector[j] for all i,j.
 * Useful for applying position-dependent weights to transfer matrices.
 */
Eigen::MatrixXd multiplytoColVec(const Eigen::MatrixXd& matrix, const Eigen::VectorXd& vector) {
    // Check dimensions
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }

    // Create result matrix with same dimensions as input matrix
    Eigen::MatrixXd result = matrix;

    // Perform elementwise multiplication for each row using array operations
    for (int i = 0; i < matrix.rows(); ++i) {
        result.row(i) = matrix.row(i).array() * vector.transpose().array();
    }

    return result;
}

/**
 * @brief Normalize matrix rows to sum to 1
 * @param matrix Input matrix
 * @return Matrix with each row normalized to unit sum
 * 
 * Converts each row to a probability distribution by dividing by row sum.
 * Includes safeguard against division by zero for numerical stability.
 */
Eigen::MatrixXd RowNormalize(const Eigen::MatrixXd& matrix) {
    // Create result matrix with same dimensions as input matrix
    Eigen::MatrixXd result = matrix;

    // Normalize each row by its sum
    for (int i = 0; i < matrix.rows(); ++i) {
        double norm = result.row(i).sum();
        if (norm > 1e-10) {  // Avoid division by zero
            result.row(i) /= norm;
        }
    }

    return result;
}

/**
 * @brief Element-wise multiply matrix columns by row vector  
 * @param rowvec Input row vector
 * @param matrix Input matrix (rows must match rowvec columns)
 * @return Matrix with each column multiplied element-wise by rowvec
 * 
 * Performs the operation: result[i,j] = rowvec[i] * matrix[i,j] for all i,j.
 * Useful for applying starting state weights to transfer matrices.
 */
Eigen::MatrixXd multiplyfromRowVec(const Eigen::RowVectorXd& rowvec, const Eigen::MatrixXd& matrix) {
    // Check dimensions
    if (rowvec.cols() != matrix.rows()) {
        throw std::invalid_argument("Dimensions incompatible: row vector cols must match matrix rows");
    }
    
    // Create result matrix
    Eigen::MatrixXd result = matrix;
    
    // Multiply each column by corresponding element of row vector
    for (int i = 0; i < matrix.cols(); ++i) {
        result.col(i) = rowvec.transpose().array() * matrix.col(i).array();
    }
    
    return result;
}

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
                           const std::vector<size_t>& cols) {
    Eigen::MatrixXd result(rows.size(), cols.size());
    
    for(size_t i = 0; i < rows.size(); i++) {
        for(size_t j = 0; j < cols.size(); j++) {
            result(i,j) = M(rows[i], cols[j]);
        }
    }
    
    return result;
}

/**
 * @brief Calculate Bernoulli probability for binary string
 * @param s Binary string ('0' and '1' characters)
 * @param p Probability of observing '0'
 * @return Probability P(s) under independent Bernoulli model
 * 
 * Computes ∏ᵢ p^{s_i} (1-p)^{1-s_i} where s_i ∈ {0,1}.
 * Used for generating reference distributions in sequence analysis.
 */
double bernoulliStringProb(const std::string& s, double p) {
    // p = probability of '0'
    // 1-p = probability of '1'
    double prob = 1.0;
    
    for(char c : s) {
        if(c == '0') {
            prob *= p;
        } else {
            prob *= (1.0 - p);
        }
    }
    
    return prob;
}

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
int drawFromProbVector(const Eigen::RowVectorXd& probs, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    
    double r = d(gen);
    double sum = 0.0;
    
    for(int i = 0; i < probs.size(); i++) {
        sum += probs(i);
        if(r <= sum) {
            return i;
        }
    }
    
    // In case of numerical issues, return last index
    return probs.size() - 1;
}
