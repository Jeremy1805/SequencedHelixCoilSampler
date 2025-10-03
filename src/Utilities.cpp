#include "Utilities.h"

// Timer class implementations
Timer::Timer(const std::string& name) : name(name), start_time(Clock::now()) {}

double Timer::elapsed() const {
    Duration elapsed = Clock::now() - start_time;
    return elapsed.count();
}

void Timer::reset() {
    start_time = Clock::now();
}

Timer::~Timer() {
    Duration elapsed = Clock::now() - start_time;
    std::cout << name << ": " << std::fixed << std::setprecision(6) 
              << elapsed.count() << " seconds\n";
}

// Utility function implementations
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
