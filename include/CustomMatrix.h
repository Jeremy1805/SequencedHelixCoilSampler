#ifndef CUSTOMMATRIX_H
#define CUSTOMMATRIX_H

#include <vector>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <set>

#ifndef DEFAULT_EPSILON_DEFINED
#define DEFAULT_EPSILON_DEFINED

inline constexpr double DEFAULT_EPSILON = 1e-300;

#endif // DEFAULT_EPSILON_DEFINED

/**
 * @class Matrix
 * @brief Template matrix class with advanced slicing operations
 * @tparam T Element type (can be scalars, custom classes, etc.)
 * 
 * Generic matrix container that supports arbitrary element types and provides
 * efficient slicing operations for both contiguous and non-contiguous submatrices.
 * Designed for high-performance scientific computing with custom data types.
 */
template<typename T>
class Matrix {
public:
    std::vector<std::vector<T>> data;  ///< 2D storage for matrix elements
    size_t rows;  ///< Number of rows
    size_t cols;  ///< Number of columns

    /**
     * @brief Constructor for matrix with specified dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     * 
     * Creates matrix filled with default-constructed elements of type T.
     */
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols));
    }

    /**
     * @brief Constructor from 2D vector
     * @param input 2D vector containing initial data
     * 
     * Validates that all rows have consistent length and copies data.
     */
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

    /**
     * @brief Default constructor creates empty matrix
     */
    Matrix() {
        rows = 0;
        cols = 0;
    }

    /**
     * @brief Element access operator (non-const)
     * @param i Row index
     * @param j Column index
     * @return Reference to element at (i,j)
     */
    T& operator()(size_t i, size_t j) {
        validateIndices(i, j);
        return data[i][j];
    }

    /**
     * @brief Element access operator (const)
     * @param i Row index
     * @param j Column index
     * @return Const reference to element at (i,j)
     */
    const T& operator()(size_t i, size_t j) const {
        validateIndices(i, j);
        return data[i][j];
    }

    /**
     * @brief Matrix multiplication with automatic type deduction
     * @tparam U Type of other matrix elements
     * @param other Matrix to multiply with
     * @return Product matrix with deduced result type
     * 
     * Performs standard matrix multiplication with automatic type deduction
     * for the result. Supports mixed-type multiplication (e.g., double * MatrixFraction).
     */
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
                R sum = data[i][0] * other(0, j);
                for (size_t k = 1; k < cols; ++k) {
                    sum = sum + (data[i][k] * other(k, j));
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }

    /**
     * @brief Extract contiguous submatrix
     * @param start_row Starting row index (inclusive)
     * @param end_row Ending row index (inclusive)
     * @param start_col Starting column index (inclusive)
     * @param end_col Ending column index (inclusive)
     * @return Submatrix containing specified range
     */
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

    /**
     * @brief Extract non-contiguous submatrix using index vectors
     * @param row_indices Vector of row indices to extract
     * @param col_indices Vector of column indices to extract
     * @return Submatrix with specified rows and columns
     * 
     * Enables extraction of arbitrary submatrices where rows/columns
     * need not be contiguous. Essential for quenched disorder calculations.
     */
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

    /**
     * @brief Extract specified rows with all columns
     * @param row_indices Vector of row indices to extract
     * @return Submatrix containing specified rows
     */
    Matrix<T> sliceRows(const std::vector<size_t>& row_indices) const {
        std::vector<size_t> all_cols(cols);
        std::iota(all_cols.begin(), all_cols.end(), 0);
        return slice(row_indices, all_cols);
    }

    /**
     * @brief Extract specified columns with all rows
     * @param col_indices Vector of column indices to extract
     * @return Submatrix containing specified columns
     */
    Matrix<T> sliceCols(const std::vector<size_t>& col_indices) const {
        std::vector<size_t> all_rows(rows);
        std::iota(all_rows.begin(), all_rows.end(), 0);
        return slice(all_rows, col_indices);
    }

    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    size_t getRows() const { return rows; }

    /**
     * @brief Get number of columns  
     * @return Number of columns
     */
    size_t getCols() const { return cols; }

    /**
     * @brief Generate string representation of matrix
     * @return Multi-line string showing matrix contents
     * 
     * Creates formatted string representation suitable for debugging
     * and console output. Uses tab separation for readability.
     */
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
    /**
     * @brief Validate matrix indices
     * @param i Row index
     * @param j Column index
     * @throws std::out_of_range if indices are invalid
     */
    void validateIndices(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
    }

    /**
     * @brief Validate slice indices for contiguous slicing
     * @param start_row Starting row index
     * @param end_row Ending row index
     * @param start_col Starting column index  
     * @param end_col Ending column index
     * @throws std::invalid_argument if slice is invalid
     */
    void validateSliceIndices(size_t start_row, size_t end_row, 
                            size_t start_col, size_t end_col) const {
        if (start_row > end_row || start_col > end_col ||
            end_row >= rows || end_col >= cols) {
            throw std::invalid_argument("Invalid slice indices");
        }
    }

    /**
     * @brief Validate index vectors for non-contiguous slicing
     * @param row_indices Vector of row indices
     * @param col_indices Vector of column indices
     * @throws std::out_of_range if any index is invalid
     */
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

/**
 * @class MatrixFraction
 * @brief Novel exact arithmetic system with matrix denominators for statistical estimation of marginalized quantities
 * 
 * This class represents a novel approach to estimating the marginalized statistics of long quenched ising chains. 
 * Through the calculation of fold probability (up to some accuracy), enables the unbiased estimatation of sequence-marginalized 
 * fold entropy for our helix-coil system.  
 * 
 * Represents quantities as fractions num/det, where det is a matrix and where operations preserve
 * exact rational structure throughout complex calculations.
 * 
 */
class MatrixFraction {
public:
    double num;              ///< Scalar numerator
    Eigen::MatrixXd den;     ///< Matrix denominator

    /**
     * @brief Constructor with numerator and denominator
     * @param num_in Scalar numerator value
     * @param den_in Matrix denominator
     */
    MatrixFraction(double num_in, const Eigen::MatrixXd& den_in) 
        : num(num_in), den(den_in) {}

    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~MatrixFraction() = default;

    /**
     * @brief Copy constructor
     * @param other MatrixFraction to copy
     */
    MatrixFraction(const MatrixFraction& other)
        : num(other.num), den(other.den) {}

    /**
     * @brief Assignment operator
     * @param other MatrixFraction to assign from
     * @return Reference to this object
     */
    MatrixFraction& operator=(const MatrixFraction& other) {
        if (this != &other) {
            num = other.num;
            den = other.den;
        }
        return *this;
    }

    /**
     * @brief Default constructor (zero fraction)
     */
    MatrixFraction() : MatrixFraction(0, Eigen::RowVectorXd::Zero(1)) {}

    /**
     * @brief Stream output operator
     * @param os Output stream
     * @param vf MatrixFraction to output
     * @return Reference to output stream
     * 
     * Formats fraction as "numerator/[matrix_elements]" for debugging.
     */
    friend std::ostream& operator<<(std::ostream& os, const MatrixFraction& vf) {
        Eigen::IOFormat fmt(4, Eigen::DontAlignCols, " ", "];[", "", "", "[", "]");
        os << vf.num << "/[" << vf.den.format(fmt) << "]";
        return os;
    }
};

/**
 * @class VectorFraction  
 * @brief Complement to MatrixFraction for storing the results of transfer matrix calculations.
 * 
 * Represents quantities as fractions num/den, where den is a row vector and where operations
 * attempt to merge two vectorFractions with denominators pointing in the same direction.
 * This avoids blowing up the numebr of directions.
 *
 */
class VectorFraction : public MatrixFraction {
public:
    /**
     * @brief Constructor from scalar numerator and row vector denominator
     * @param num_in Scalar numerator
     * @param den_in Row vector denominator
     * 
     * Automatically normalizes by dividing by first element if non-zero.
     */
    VectorFraction(double num_in, const Eigen::RowVectorXd& den_in)
        : MatrixFraction(num_in, den_in) {
        // Verify that den is actually a row vector
        if (den.rows() != 1) {
            throw std::invalid_argument("Denominator must be a row vector");
        } else if (den(0) != 0) {
            // Normalize on creation
            double old_den0 = den(0);
            //double mag = std::sqrt(std::pow(den(0),2)+std::pow(den(1),2));
            //double mag = den(0)+den(1);
            den = den/old_den0;
            num = num/old_den0;
        }
    }

    /**
     * @brief Constructor creating zero vector of specified size
     * @param n Size of denominator vector
     */
    VectorFraction(int n) {num = 0; den = Eigen::RowVectorXd::Zero(n);}
    
    /**
     * @brief Default constructor (unit size)
     */
    VectorFraction() : VectorFraction(1) {}
    
    /**
     * @brief Get denominator as row vector
     * @return Row vector denominator
     */
    Eigen::RowVectorXd getDenominator() const {
        return den.row(0);
    }

    /**
     * @brief Copy constructor
     * @param other VectorFraction to copy
     */
    VectorFraction(const VectorFraction& other)
        : MatrixFraction(other) {}

    /**
     * @brief Assignment operator
     * @param other VectorFraction to assign from
     * @return Reference to this object
     */
    VectorFraction& operator=(const VectorFraction& other) {
        MatrixFraction::operator=(other);
        return *this;
    }

    /**
     * @brief Multiplication with MatrixFraction
     * @param matrix MatrixFraction to multiply with
     * @return Product as VectorFraction
     * 
     * Computes (num1 * num2) / (den1 * den2) where den2 is a matrix.
     */
    VectorFraction operator*(const MatrixFraction matrix) const{
        return VectorFraction(num*matrix.num,den*matrix.den);
    }

    /**
     * @brief Addition of VectorFractions
     * @param v2 VectorFraction to add
     * @return Sum as VectorFraction
     * 
     * Note: This operation is marked for deletion as summation
     * should not generally be defined for this type.
     */
    VectorFraction operator+(const VectorFraction v2) const{
        return VectorFraction(num+v2.num,den+v2.den);
    }

    /**
     * @brief Stream output operator for VectorFraction
     * @param os Output stream
     * @param vf VectorFraction to output
     * @return Reference to output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const VectorFraction& vf) {
        Eigen::IOFormat fmt(4, Eigen::DontAlignCols, " ", " ", "", "", "[", "]");
        os << vf.num << "/[" << vf.den.format(fmt) << "]";
        return os;
    }
};

/**
 * @struct VectorFractionCompare
 * @brief Custom comparator for VectorFraction based on denominators
 * 
 * Provides lexicographic comparison of VectorFractions by their denominators
 * with configurable numerical tolerance. Used for maintaining sorted
 * collections of VectorFractions.
 */
struct VectorFractionCompare {
    double epsilon;  ///< Numerical tolerance for comparisons

    /**
     * @brief Constructor with configurable epsilon
     * @param eps Numerical tolerance (default: DEFAULT_EPSILON)
     */
    explicit VectorFractionCompare(double eps = DEFAULT_EPSILON) : epsilon(eps) {}

    /**
     * @brief Comparison operator
     * @param a First VectorFraction
     * @param b Second VectorFraction  
     * @return True if a < b in lexicographic order
     * 
     * First compares by vector size, then element-wise with tolerance.
     */
    bool operator()(const VectorFraction& a, const VectorFraction& b) const {
        const auto& a_den = a.den;
        const auto& b_den = b.den;
        
        if (std::abs(a_den.cols() - b_den.cols()) > 0) 
            return a_den.cols() < b_den.cols();
        
        for (int i = 0; i < a_den.cols(); i++) {
            double diff = a_den(i) - b_den(i);
            if ( (std::abs(diff) > epsilon*a_den(i)) || (std::abs(diff) > epsilon*b_den(i))  ) {
                return diff < 0;
            }
        }
        
        return false;  // Equal or nearly equal vectors
    }
};

/**
 * @class VectorFractionList
 * @brief Marginalized probability calculation system for long quenched sequences
 * 
 * This class automatically manages collections of VectorFractions,
 * merging together VectorFractions with parallel denominators, avoiding blowup. 
 * 
 */
class VectorFractionList {
public:
    double epsilon = DEFAULT_EPSILON;  ///< Numerical tolerance for comparisons
    
    /// Sorted set with custom comparator
    std::set<VectorFraction, VectorFractionCompare> ls;
    
    /**
     * @brief Default constructor with configurable epsilon
     * @param eps Numerical tolerance
     */
    VectorFractionList(double eps = DEFAULT_EPSILON) 
        : epsilon(eps), ls(VectorFractionCompare(eps)) {}

    /**
     * @brief Constructor from vector with automatic consolidation
     * @param fractions Vector of VectorFractions to consolidate
     * @param eps Numerical tolerance
     * 
     * Automatically combines fractions with matching denominators
     * by adding their numerators.
     */
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

    /**
     * @brief Copy constructor preserving epsilon
     * @param other VectorFractionList to copy
     */
    VectorFractionList(const VectorFractionList& other) 
        : epsilon(other.epsilon), ls(other.ls) {}

    /**
     * @brief Constructor creating n identical fractions
     * @param n Number of fractions to create
     * @param m Size of each denominator vector
     * @param eps Numerical tolerance
     */
    VectorFractionList(int n, int m, double eps = DEFAULT_EPSILON) 
        : epsilon(eps), ls(VectorFractionCompare(eps)) {
        std::vector<VectorFraction> fractions(n, VectorFraction(1, Eigen::RowVectorXd::Ones(m)));
        for (const auto& vf : fractions) {
            ls.insert(vf);
        }
    }

    /**
     * @brief Move constructor preserving epsilon
     * @param other VectorFractionList to move from
     */
    VectorFractionList(VectorFractionList&& other) noexcept 
        : epsilon(other.epsilon), ls(std::move(other.ls)) {}

    /**
     * @brief Finalize calculation with MatrixFraction
     * @param finisher MatrixFraction to complete calculation
     * @return Final scalar result
     * 
     * Computes the final result by evaluating each VectorFraction
     * in the list against the finisher MatrixFraction and summing.
     */
    double Finisher(MatrixFraction finisher) {
        double result = 0.0;
        for (const auto& vf : ls) {
            result += vf.num*finisher.num/(vf.den*finisher.den)(0,0);
        }
        return(result);
    }

    /**
     * @brief Copy assignment operator preserving epsilon
     * @param other VectorFractionList to assign from
     * @return Reference to this object
     */
    VectorFractionList& operator=(const VectorFractionList& other) {
        if (this != &other) {
            epsilon = other.epsilon;
            ls = other.ls;
        }
        return *this;
    }

    /**
     * @brief Move assignment operator preserving epsilon
     * @param other VectorFractionList to move from
     * @return Reference to this object
     */
    VectorFractionList& operator=(VectorFractionList&& other) noexcept {
        if (this != &other) {
            epsilon = other.epsilon;
            ls = std::move(other.ls);
        }
        return *this;
    }

    /**
     * @brief Addition of VectorFractionLists with consolidation
     * @param other VectorFractionList to add
     * @return Sum with consolidated fractions
     * 
     * Merges two lists while automatically consolidating fractions
     * with matching denominators (within epsilon tolerance).
     */
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

    /**
     * @brief Matrix multiplication with automatic consolidation
     * @param matrix MatrixFraction to multiply with
     * @return Product VectorFractionList with consolidated terms
     * 
     * Multiplies each VectorFraction in the list by the MatrixFraction,
     * automatically consolidating results with matching denominators.
     */
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

    // Iterator support for range-based loops and STL algorithms
    auto begin() { return ls.begin(); }
    auto end() { return ls.end(); }
    auto begin() const { return ls.cbegin(); }
    auto end() const { return ls.cend(); }
    auto cbegin() const { return ls.cbegin(); }
    auto cend() const { return ls.cend(); }

    /**
     * @brief Get number of terms in the list
     * @return Number of distinct VectorFractions
     */
    size_t size() const { return ls.size(); }
    
    /**
     * @brief Check if list is empty
     * @return True if no terms present
     */
    bool empty() const { return ls.empty(); }
    
    /**
     * @brief Clear all terms from list
     */
    void clear() { ls.clear(); }

    /**
     * @brief Stream output operator for debugging
     * @param os Output stream
     * @param vfl VectorFractionList to output
     * @return Reference to output stream
     * 
     * Formats list as pipe-separated VectorFractions for debugging.
     */
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

#endif // CUSTOMMATRIX_H
