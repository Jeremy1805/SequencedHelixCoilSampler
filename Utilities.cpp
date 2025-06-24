#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <fstream>

class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    std::string name;
    TimePoint start_time;

public:
    Timer(const std::string& name = "Timer") : name(name), start_time(Clock::now()) {}

    // Get current duration without stopping timer
    double elapsed() const {
        Duration elapsed = Clock::now() - start_time;
        return elapsed.count();
    }

    // Reset the timer
    void reset() {
        start_time = Clock::now();
    }

    // Destructor prints time automatically
    ~Timer() {
        Duration elapsed = Clock::now() - start_time;
        std::cout << name << ": " << std::fixed << std::setprecision(6) 
                  << elapsed.count() << " seconds\n";
    }
};

template<typename... T>
void saveTuplesToCSV(const std::vector<std::tuple<T...>>& data, 
                     const std::string& filename,
                     const std::vector<std::string>& headers = {}) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    // Set precision for floating-point numbers
    file.precision(15);
    file << std::scientific;

    // Write headers if provided
    if (!headers.empty()) {
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << "\t";
        }
        file << "\n";
    }

    // Write data
    for (const auto& tuple : data) {
        std::apply([&file](const auto&... args) {
            size_t idx = 0;
            ((file << args << (++idx != sizeof...(T) ? "\t" : "")), ...);
        }, tuple);
        file << "\n";
    }

    file.close();
}

// Helper function to escape tab characters in fields
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

template<typename K, typename V>
bool saveMapToTSV(const std::unordered_map<K, V>& map, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "Key\tValue\n";
    
    // Write data
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

Eigen::MatrixXd multiplytoColVec(const Eigen::MatrixXd& matrix, const Eigen::VectorXd& vector) {
    // Check dimensions
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }

    // Create result matrix with same dimensions as input matrix
    Eigen::MatrixXd result = matrix;

    // Perform elementwise multiplication for each row
    // We can use array() to convert to coefficient-wise operations
    for (int i = 0; i < matrix.rows(); ++i) {
        result.row(i) = matrix.row(i).array() * vector.transpose().array();
    }

    return result;
}

Eigen::MatrixXd RowNormalize(const Eigen::MatrixXd& matrix) {
    // Create result matrix with same dimensions as input matrix
    Eigen::MatrixXd result = matrix;

    // Perform elementwise multiplication for each row
    // We can use array() to convert to coefficient-wise operations
    for (int i = 0; i < matrix.rows(); ++i) {
        double norm = result.row(i).sum();
        if (norm > 1e-10) {  // Avoid division by zero
            result.row(i) /= norm;
        }
    }

    return result;
}
// Normalize the row 
        
Eigen::MatrixXd multiplyfromRowVec(const Eigen::RowVectorXd& rowvec, const Eigen::MatrixXd& matrix) {
    // Check dimensions
    if (rowvec.cols() != matrix.rows()) {
        throw std::invalid_argument("Dimensions incompatible: row vector cols must match matrix rows");
    }
    
    // Multiply row vector with matrix
    Eigen::MatrixXd result = matrix;
    
    // Normalize by sum
    for (int i = 0; i < matrix.cols(); ++i) {
        result.col(i) = rowvec.transpose().array()*matrix.col(i).array();
    }
    
    return result;
}

//Utility function for slicing matrices
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