#include <vector>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <boost/config/assert_cxx17.hpp>
#include <set>

#ifndef DEFAULT_EPSILON_DEFINED
#define DEFAULT_EPSILON_DEFINED

inline constexpr double DEFAULT_EPSILON = 1e-300;

#endif // DEFAULT_EPSILON_DEFINED


template<typename T>
class Matrix {
public:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

    // Constructors
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols));
    }

    Matrix(const std::vector<std::vector<T>>& input) {
        if (input.empty()) {
            rows = 0;
            cols = 0;
            return;
        }
        
        rows = input.size();
        cols = input[0].size();
        
        // Validate that all rows have the same length
        for (const auto& row : input) {
            if (row.size() != cols) {
                throw std::invalid_argument("All rows must have the same length");
            }
        }
        
        data = input;
    }

    Matrix() {
        rows = 0;
        cols = 0;
    }

    // Access operators
    T& operator()(size_t i, size_t j) {
        validateIndices(i, j);
        return data[i][j];
    }

    const T& operator()(size_t i, size_t j) const {
        validateIndices(i, j);
        return data[i][j];
    }

    // Matrix multiplication with different types
    template<typename U>
    auto operator*(const Matrix<U>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }

        // Deduce result type from multiplication of elements
        using R = decltype(std::declval<T>() * std::declval<U>());
        Matrix<R> result(rows, other.cols);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                R sum = data[i][0]* other(0, j);
                for (size_t k = 1; k < cols; ++k) {
                    sum = sum + (data[i][k] * other(k, j));
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }

    // Contiguous slice operations
    Matrix<T> slice(size_t start_row, size_t end_row, size_t start_col, size_t end_col) const {
        validateSliceIndices(start_row, end_row, start_col, end_col);
        
        size_t new_rows = end_row - start_row + 1;
        size_t new_cols = end_col - start_col + 1;
        
        Matrix<T> result(new_rows, new_cols);
        
        for (size_t i = 0; i < new_rows; ++i) {
            for (size_t j = 0; j < new_cols; ++j) {
                result(i, j) = data[start_row + i][start_col + j];
            }
        }
        
        return result;
    }

    // Noncontiguous slice operations
    Matrix<T> slice(const std::vector<size_t>& row_indices, 
                   const std::vector<size_t>& col_indices) const {
        validateIndices(row_indices, col_indices);
        
        Matrix<T> result(row_indices.size(), col_indices.size());
        
        for (size_t i = 0; i < row_indices.size(); ++i) {
            for (size_t j = 0; j < col_indices.size(); ++j) {
                result(i, j) = data[row_indices[i]][col_indices[j]];
            }
        }
        
        return result;
    }

    // Single row/column slices
    Matrix<T> sliceRows(const std::vector<size_t>& row_indices) const {
        std::vector<size_t> all_cols(cols);
        std::iota(all_cols.begin(), all_cols.end(), 0);
        return slice(row_indices, all_cols);
    }

    Matrix<T> sliceCols(const std::vector<size_t>& col_indices) const {
        std::vector<size_t> all_rows(rows);
        std::iota(all_rows.begin(), all_rows.end(), 0);
        return slice(all_rows, col_indices);
    }

    // Getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // String representation
    std::string toString() const {
        std::ostringstream oss;
        oss << "Matrix " << rows << "x" << cols << ":\n";
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                oss << data[i][j] << "\t";
            }
            oss << "\n";
        }
        return oss.str();
    }

private:
    void validateIndices(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
    }

    void validateSliceIndices(size_t start_row, size_t end_row, 
                            size_t start_col, size_t end_col) const {
        if (start_row > end_row || start_col > end_col ||
            end_row >= rows || end_col >= cols) {
            throw std::invalid_argument("Invalid slice indices");
        }
    }

    void validateIndices(const std::vector<size_t>& row_indices, 
                        const std::vector<size_t>& col_indices) const {
        // Check if any indices are out of bounds
        for (size_t idx : row_indices) {
            if (idx >= rows) {
                throw std::out_of_range("Row index out of range");
            }
        }
        for (size_t idx : col_indices) {
            if (idx >= cols) {
                throw std::out_of_range("Column index out of range");
            }
        }
    }
};

// Example usage with a custom class
class MatrixFraction {
public:
    double num;
    Eigen::MatrixXd den;

    MatrixFraction(double num_in, const Eigen::MatrixXd& den_in) 
        : num(num_in), den(den_in) {}

    // Virtual destructor for proper inheritance
    virtual ~MatrixFraction() = default;

    // Copy constructor
    MatrixFraction(const MatrixFraction& other)
        : num(other.num), den(other.den) {}

    // Assignment operator
    MatrixFraction& operator=(const MatrixFraction& other) {
        if (this != &other) {
            num = other.num;
            den = other.den;
        }
        return *this;
    }

    MatrixFraction() : MatrixFraction(0, Eigen::RowVectorXd::Zero(1)) {}

    // Stream operator for VectorFraction
    friend std::ostream& operator<<(std::ostream& os, const MatrixFraction& vf) {
        Eigen::IOFormat fmt(4, Eigen::DontAlignCols, " ", "];[", "", "", "[", "]");
        os << vf.num << "/[" << vf.den.format(fmt) << "]";
        return os;
    }
};

class VectorFraction : public MatrixFraction {
public:
    // Constructor that takes a RowVectorXd
    VectorFraction(double num_in, const Eigen::RowVectorXd& den_in)
        : MatrixFraction(num_in, den_in) {
        // Verify that den is actually a row vector
        if (den.rows() != 1) {
            throw std::invalid_argument("Denominator must be a row vector");
        } else if (den(0) != 0) {
            // Normalize on creation
            double old_den0 = den(0);
            den = den/old_den0;
            num = num/old_den0;
        }
    }

    // Constructors: TO BE DELETED
    VectorFraction(int n) {num = 0; den = Eigen::RowVectorXd::Zero(n);}
    VectorFraction() : VectorFraction(1) {}
    
    // Get the denominator specifically as a RowVectorXd
    Eigen::RowVectorXd getDenominator() const {
        return den.row(0);
    }

    // Copy constructor
    VectorFraction(const VectorFraction& other)
        : MatrixFraction(other) {}

    // Assignment operator
    VectorFraction& operator=(const VectorFraction& other) {
        MatrixFraction::operator=(other);
        return *this;
    }

    VectorFraction operator*(const MatrixFraction matrix) const{
        return VectorFraction(num*matrix.num,den*matrix.den);
    }

    // TO BE DELETED! Summation should not be defined
    VectorFraction operator+(const VectorFraction v2) const{
        return VectorFraction(num+v2.num,den+v2.den);
    }

    // Stream operator for VectorFraction
    friend std::ostream& operator<<(std::ostream& os, const VectorFraction& vf) {
        Eigen::IOFormat fmt(4, Eigen::DontAlignCols, " ", " ", "", "", "[", "]");
        os << vf.num << "/[" << vf.den.format(fmt) << "]";
        return os;
    }
};

// Custom comparator for VectorFractions based on their denominators
struct VectorFractionCompare {
    double epsilon;

    // Constructor with default value
    explicit VectorFractionCompare(double eps = DEFAULT_EPSILON) : epsilon(eps) {}

    bool operator()(const VectorFraction& a, const VectorFraction& b) const {
        const auto& a_den = a.den;
        const auto& b_den = b.den;
        
        if (std::abs(a_den.cols() - b_den.cols()) > 0) 
            return a_den.cols() < b_den.cols();
        
        for (int i = 0; i < a_den.cols(); i++) {
            double diff = a_den(i) - b_den(i);
            if (std::abs(diff) > epsilon) {
                return diff < 0;
            }
        }
        return false;  // Equal or nearly equal vectors
    }
};

class VectorFractionList {
public:
    // Store the epsilon value at the class level
    double epsilon = DEFAULT_EPSILON;
    
    // Modified to use custom comparator with our epsilon
    std::set<VectorFraction, VectorFractionCompare> ls;
    
    // Default constructor with optional epsilon parameter
    VectorFractionList(double eps = DEFAULT_EPSILON) 
        : epsilon(eps), ls(VectorFractionCompare(eps)) {}

    // Constructor from vector of VectorFractions with optional epsilon
    VectorFractionList(const std::vector<VectorFraction>& fractions, double eps = DEFAULT_EPSILON) 
        : epsilon(eps), ls(VectorFractionCompare(eps)) {
        for (const auto& vf : fractions) {
            auto it = ls.find(vf);  // Will use VectorFractionCompare to find matching denominator
            if (it != ls.end()) {
                // If matching denominator exists, remove old and insert combined
                VectorFraction combined(it->num + vf.num, vf.den);
                ls.erase(it);
                ls.insert(combined);
            } else {
                ls.insert(vf);
            }
        }
    }

    // Copy constructor - preserve epsilon
    VectorFractionList(const VectorFractionList& other) 
        : epsilon(other.epsilon), ls(other.ls) {}

    // Modified constructor that also accepts epsilon
    VectorFractionList(int n, int m, double eps = DEFAULT_EPSILON) 
        : epsilon(eps), ls(VectorFractionCompare(eps)) {
        std::vector<VectorFraction> fractions(n, VectorFraction(1, Eigen::RowVectorXd::Ones(m)));
        for (const auto& vf : fractions) {
            ls.insert(vf);
        }
    }

    // Move constructor - preserve epsilon
    VectorFractionList(VectorFractionList&& other) noexcept 
        : epsilon(other.epsilon), ls(std::move(other.ls)) {}

    double Finisher(MatrixFraction finisher) {
        double result = 0.0;
        for (const auto& vf : ls) {
            result += vf.num*finisher.num/(vf.den*finisher.den)(0,0);
        }
        return(result);
    }

    // Copy assignment operator - preserve epsilon
    VectorFractionList& operator=(const VectorFractionList& other) {
        if (this != &other) {
            epsilon = other.epsilon;
            ls = other.ls;
        }
        return *this;
    }

    // Move assignment operator - preserve epsilon
    VectorFractionList& operator=(VectorFractionList&& other) noexcept {
        if (this != &other) {
            epsilon = other.epsilon;
            ls = std::move(other.ls);
        }
        return *this;
    }

    // Addition operator - ensure consistent epsilon handling
    VectorFractionList operator+(const VectorFractionList& other) const {
        // Use the epsilon from this object
        VectorFractionList result(epsilon);
        auto it1 = ls.begin();
        auto it2 = other.ls.begin();
        
        // Create a comparator with our epsilon
        VectorFractionCompare comp(epsilon);
        
        while (it1 != ls.end() && it2 != other.ls.end()) {
            if (!comp(*it1, *it2) && !comp(*it2, *it1)) {
                // Denominators are equal (within epsilon)
                result.ls.insert(VectorFraction(it1->num + it2->num, it1->den));
                ++it1;
                ++it2;
            } else if (comp(*it1, *it2)) {
                // it1's denominator is smaller
                result.ls.insert(*it1);
                ++it1;
            } else {
                // it2's denominator is smaller
                result.ls.insert(*it2);
                ++it2;
            }
        }
        
        // Add remaining elements
        while (it1 != ls.end()) {
            result.ls.insert(*it1);
            ++it1;
        }
        
        while (it2 != other.ls.end()) {
            result.ls.insert(*it2);
            ++it2;
        }
        
        return result;
    }

    // Matrix multiplication operator
    VectorFractionList operator*(const MatrixFraction& matrix) const {
        VectorFractionList result(epsilon);
        for (const auto& vf : ls) {
            VectorFraction multiplied = vf * matrix;
            auto it = result.ls.find(multiplied);
            if (it != result.ls.end()) {
                const_cast<VectorFraction&>(*it).num += multiplied.num;
            } else {
                result.ls.insert(multiplied);
            }
        }
        return result;
    }

    // Iterators
    auto begin() { return ls.begin(); }
    auto end() { return ls.end(); }
    auto begin() const { return ls.cbegin(); }
    auto end() const { return ls.cend(); }
    auto cbegin() const { return ls.cbegin(); }
    auto cend() const { return ls.cend(); }

    // Size operations
    size_t size() const { return ls.size(); }
    bool empty() const { return ls.empty(); }
    void clear() { ls.clear(); }

    friend std::ostream& operator<<(std::ostream& os, const VectorFractionList& vfl) {
        bool first = true;
        for (const auto& vf : vfl.ls) {
            if (!first) {
                os << "|";
            }
            os << vf;
            first = false;
        }
        return os;
    }
};