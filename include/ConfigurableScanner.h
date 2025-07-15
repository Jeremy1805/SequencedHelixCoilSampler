#ifndef CONFIGURABLESCANNER_H
#define CONFIGURABLESCANNER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <muParser.h>

/**
 * @file ConfigurableScanner.h
 * @brief Header for configurable parameter scanning system
 * 
 * This header provides classes and functions for reading JSON configuration files
 * and performing automated parameter scans over protein folding models.
 */

/**
 * @brief Configuration structure for parameter scans
 * 
 * Contains all parameters needed to define and execute a parameter scan,
 * including model specification, scan ranges, and output settings.
 */
struct Config {
    std::string scan_type;          ///< Type of scan ("bernoulli" or "error")
    std::string scan_subtype;       ///< Some scans will require a subtype
    std::string fold_model;         ///< Model type ("Ising2" or "Ising2S3F")
    std::vector<std::vector<std::string>> energy_matrix;  ///< Transfer matrix expressions
    std::vector<std::string> start_vector;               ///< Starting vector expressions
    std::vector<std::string> end_vector;                 ///< Ending vector expressions
    int length;                     ///< Polymer length
    bool x_var_is_log = false; /// If true, then the exponential 2^x is used as the true x variable. 
    double fixed_error= 0.001; // Needed for scans with fixed errors 
    /**
     * @brief Parameter range specification
     * 
     * Defines a range of parameter values to scan over with a given step size.
     */
    struct ParamRange {
        std::string name;           ///< Parameter name
        double start;               ///< Starting value
        double end;                 ///< Ending value (exclusive)
        double step;                ///< Step size
    } x_param_range, y_param_range; ///< X and Y parameter ranges
    
    std::vector<std::string> templates;  ///< Template sequences for error scan
};

/**
 * @brief Expression evaluator using muParser for mathematical expressions
 * 
 * Evaluates mathematical expressions containing parameter names using the
 * muParser library. Supports all standard mathematical functions and operations
 * including arithmetic, trigonometric, logarithmic, and conditional expressions.
 * 
 * Example expressions:
 * - "exp(2*x + 1)"
 * - "sin(x*pi)"  
 * - "sqrt(x^2 + y^2)"
 * - "x > 0 ? log(x) : 0"
 */
class ExpressionEvaluator {
private:
    std::map<std::string, double> parameters;  ///< Parameter name-value pairs
    
public:
    /**
     * @brief Set parameter value for expression evaluation
     * @param name Parameter name (e.g., "x", "bernoulli_prob")
     * @param value Parameter value
     */
    void setParameter(const std::string& name, double value);
    
    /**
     * @brief Evaluate mathematical expression with current parameters
     * @param expression Mathematical expression string
     * @return Evaluated numerical result
     * @throws std::runtime_error if expression is invalid or evaluation fails
     */
    double evaluate(const std::string& expression);
};

// Configuration parsing functions

/**
 * @brief Parse JSON configuration file using nlohmann/json
 * @param filename Path to JSON configuration file
 * @return Parsed configuration structure
 * @throws std::runtime_error if file cannot be opened or JSON is invalid
 */
Config parseConfig(const std::string& filename);

// Matrix building functions

/**
 * @brief Build Eigen matrix from configuration and parameters
 * @param matrixConfig 2D vector of expression strings
 * @param params Map of parameter names to current values
 * @return Evaluated numerical matrix
 * @throws std::runtime_error if matrix config is empty or expressions are invalid
 */
Eigen::MatrixXd buildMatrix(const std::vector<std::vector<std::string>>& matrixConfig,
                           const std::map<std::string, double>& params);

/**
 * @brief Build Eigen vector from configuration and parameters
 * @param vectorConfig Vector of expression strings
 * @param params Map of parameter names to current values
 * @return Evaluated numerical vector
 * @throws std::runtime_error if expressions are invalid
 */
Eigen::VectorXd buildVector(const std::vector<std::string>& vectorConfig,
                           const std::map<std::string, double>& params);

// Scan execution functions

/**
 * @brief Perform Bernoulli parameter scan
 * 
 * Scans over Bernoulli probability parameters and energy parameters,
 * analyzing the deviation from equilibrium for each combination.
 * 
 * The scan generates sequence distributions using Bernoulli models
 * and compares them against equilibrium distributions from the
 * specified folding model.
 * 
 * @param config Configuration containing scan parameters and model definition
 * @param outputFilename Output TSV filename for results
 * @throws std::runtime_error if scan type is unsupported or model creation fails
 */
void performBernoulliScan(const Config& config, const std::string& outputFilename);

/**
 * @brief Perform template-based error scan
 * 
 * Scans over template error rates and energy parameters, analyzing
 * sequence distributions derived from template sequences with errors.
 * 
 * The scan generates sequence distributions by introducing errors
 * into template sequences at various rates, then analyzes how these
 * distributions compare to folding model equilibria.
 * 
 * @param config Configuration containing scan parameters and template sequences
 * @param outputFilename Output TSV filename for results
 * @throws std::runtime_error if templates are missing or model creation fails
 */
void performErrorScan(const Config& config, const std::string& outputFilename);

/**
 * @brief Perform scan in Total Variation Distance
 * 
 * x_param is always total variation distance - a walk is performed starting from equilibrium
 * towards some other probability distribution and results are recorded at log/linear intervals.
 * y_param varies by scan subtypes (see above) 
 *
 * @param config Configuration containing scan parameters and template sequences
 * @param outputFilename Base name for output file
 */
void performTVScan(const Config& config, const std::string& outputFilename);
#endif
