#include "ConfigurableScanner.h"
#include "Utilities.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <sstream>
#include <map>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <muParser.h>
#include "FoldModels.h"

using json = nlohmann::json;

// ExpressionEvaluator implementation
void ExpressionEvaluator::setParameter(const std::string& name, double value) {
    parameters[name] = value;
}

double ExpressionEvaluator::evaluate(const std::string& expression) {
    try {
        mu::Parser parser;
        
        // Add all parameters as variables to the parser
        for (auto& [name, value] : parameters) {
            parser.DefineVar(name, &parameters[name]);
        }
        
        parser.SetExpr(expression);
        return parser.Eval();
        
    } catch (mu::Parser::exception_type& e) {
        throw std::runtime_error("Expression evaluation error in '" + expression + "': " + std::string(e.GetMsg()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Cannot evaluate expression '" + expression + "': " + std::string(e.what()));
    }
}

// Configuration parsing implementation
Config parseConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    file.close();
    
    Config config;
    
    // Parse basic fields
    config.scan_type = j.at("scan_type").get<std::string>();
    config.fold_model = j.at("fold_model").get<std::string>();
    config.length = j.at("length").get<int>();
    
    // Parse parameter ranges
    auto x_param = j.at("x_param_range");
    config.x_param_range.name = x_param.at("name").get<std::string>();
    config.x_param_range.start = x_param.at("start").get<double>();
    config.x_param_range.end = x_param.at("end").get<double>();
    config.x_param_range.step = x_param.at("step").get<double>();
    
    auto y_param = j.at("y_param_range");
    config.y_param_range.name = y_param.at("name").get<std::string>();
    config.y_param_range.start = y_param.at("start").get<double>();
    config.y_param_range.end = y_param.at("end").get<double>();
    config.y_param_range.step = y_param.at("step").get<double>();
    
    // Parse energy matrix (2D array of strings)
    auto energy_matrix_json = j.at("energy_matrix");
    for (const auto& row : energy_matrix_json) {
        std::vector<std::string> matrix_row;
        for (const auto& element : row) {
            matrix_row.push_back(element.get<std::string>());
        }
        config.energy_matrix.push_back(matrix_row);
    }
    
    // Parse start and end vectors (arrays of strings)
    auto start_vector_json = j.at("start_vector");
    for (const auto& element : start_vector_json) {
        config.start_vector.push_back(element.get<std::string>());
    }
    
    auto end_vector_json = j.at("end_vector");
    for (const auto& element : end_vector_json) {
        config.end_vector.push_back(element.get<std::string>());
    }
    
    // Parse templates (optional, for error scan)
    if (j.contains("templates")) {
        auto templates_json = j.at("templates");
        for (const auto& tmpl : templates_json) {
            config.templates.push_back(tmpl.get<std::string>());
        }
    }
    
    return config;
}

// Matrix building implementations
Eigen::MatrixXd buildMatrix(const std::vector<std::vector<std::string>>& matrixConfig,
                           const std::map<std::string, double>& params) {
    if (matrixConfig.empty()) {
        throw std::runtime_error("Empty matrix configuration");
    }
    
    int rows = matrixConfig.size();
    int cols = matrixConfig[0].size();
    Eigen::MatrixXd matrix(rows, cols);
    
    ExpressionEvaluator evaluator;
    for (const auto& [name, value] : params) {
        evaluator.setParameter(name, value);
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix(i, j) = evaluator.evaluate(matrixConfig[i][j]);
        }
    }
    
    return matrix;
}

/**
 * @brief Build Eigen vector from config and parameters
 */
Eigen::VectorXd buildVector(const std::vector<std::string>& vectorConfig,
                           const std::map<std::string, double>& params) {
    Eigen::VectorXd vector(vectorConfig.size());
    
    ExpressionEvaluator evaluator;
    for (const auto& [name, value] : params) {
        evaluator.setParameter(name, value);
    }
    
    for (size_t i = 0; i < vectorConfig.size(); i++) {
        vector(i) = evaluator.evaluate(vectorConfig[i]);
    }
    
    return vector;
}

/**
 * @brief Perform Bernoulli parameter scan
 * 
 * Scans over Bernoulli probability parameters and energy parameters,
 * analyzing the deviation from equilibrium for each combination.
 * 
 * @param config Configuration containing scan parameters and model definition
 * @param outputFilename Base name for output file
 */
void performBernoulliScan(const Config& config, const std::string& outputFilename) {
    std::cout << "Starting Bernoulli scan..." << std::endl;
    
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> results;
    
    // Nested loops over parameter ranges
    for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
        
        // Generate Bernoulli distribution for this x_param value
        std::unordered_map<std::string,double> SCopyMap;
        if (config.x_param_range.name == "bernoulli_prob") {
            SCopyMap = IsingVar::GenerateBernoulliMap(x_param, config.length);
        } else {
            throw std::runtime_error("Unsupported x_param for Bernoulli scan: " + config.x_param_range.name);
        }
        
        for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
            
            // Set up parameter values for matrix evaluation
            std::map<std::string, double> params;
            params[config.x_param_range.name] = x_param;
            params[config.y_param_range.name] = y_param;
            params["x"] = y_param;  // Default free variable
            
            // Build matrices from config
            auto matrix = buildMatrix(config.energy_matrix, params);
            auto start_vec = buildVector(config.start_vector, params);
            auto end_vec = buildVector(config.end_vector, params);
            
            // Create model instance using direct constructor
            std::unique_ptr<IsingVar> model;
            if (config.fold_model == "Ising2") {
                model = std::make_unique<Ising2>(matrix, start_vec.transpose(), end_vec, config.length);
            } else if (config.fold_model == "Ising2S3F") {
                model = std::make_unique<Ising2S3F>(matrix, start_vec.transpose(), end_vec, config.length);
            } else {
                throw std::runtime_error("Unsupported fold model: " + config.fold_model);
            }
            
            // Generate equilibrium table and analyze
            model->GetEquilibrumTable();
            auto entry = model->Results(SCopyMap);
            
            // Store results
            results.push_back(std::tuple_cat(std::make_tuple(x_param), std::make_tuple(y_param), entry));
            
            std::cout << "Completed: " << config.x_param_range.name << "=" << x_param 
                     << ", " << config.y_param_range.name << "=" << y_param << std::endl;
        }
    }
    
    // Save results
    std::vector<std::string> headers = {
        config.x_param_range.name, config.y_param_range.name,
        "D(pmap(s,w)||peq(s,w))", "D(pmap(s)||peq(s))", "D(pmap(w)||peq(w))",
        "NHelixeq", "NHelixMap", "<U>eq", "<U>map",
        "Heq(s|w)", "Hmap(s|w)", "Heq(w)", "Hmap(w)", "Heq(s)", "Hmap(s)"
    };
    
    saveTuplesToCSV(results, outputFilename, headers);
    std::cout << "Bernoulli scan completed. Results saved to: " << outputFilename << std::endl;
}

/**
 * @brief Perform template-based error scan
 * 
 * Scans over template error rates and energy parameters, analyzing
 * sequence distributions derived from template sequences with errors.
 * 
 * @param config Configuration containing scan parameters and template sequences
 * @param outputFilename Base name for output file
 */
void performErrorScan(const Config& config, const std::string& outputFilename) {
    std::cout << "Starting Error scan..." << std::endl;
    
    if (config.templates.empty()) {
        throw std::runtime_error("Error scan requires template sequences in config");
    }
    
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> results;
    
    // Nested loops over parameter ranges
    for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
        
        // Generate template-based distribution for this x_param value
        std::unordered_map<std::string,double> SCopyMap;
        if (config.x_param_range.name == "error_rate") {
            SCopyMap = IsingVar::GenerateFromUniformTemplate(config.templates, x_param);
        } else if (config.x_param_range.name == "log_error") {
            double error_rate = std::pow(2.0, x_param);
            SCopyMap = IsingVar::GenerateFromUniformTemplate(config.templates, error_rate);
        } else {
            throw std::runtime_error("Unsupported x_param for Error scan: " + config.x_param_range.name);
        }
        
        for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
            
            // Set up parameter values for matrix evaluation
            std::map<std::string, double> params;
            params[config.x_param_range.name] = x_param;
            params[config.y_param_range.name] = y_param;
            params["x"] = y_param;  // Default free variable
            
            // Build matrices from config
            auto matrix = buildMatrix(config.energy_matrix, params);
            auto start_vec = buildVector(config.start_vector, params);
            auto end_vec = buildVector(config.end_vector, params);
            
            // Create model instance using direct constructor
            std::unique_ptr<IsingVar> model;
            if (config.fold_model == "Ising2") {
                model = std::make_unique<Ising2>(matrix, start_vec.transpose(), end_vec, config.length);
            } else if (config.fold_model == "Ising2S3F") {
                model = std::make_unique<Ising2S3F>(matrix, start_vec.transpose(), end_vec, config.length);
            } else {
                throw std::runtime_error("Unsupported fold model: " + config.fold_model);
            }
            
            // Generate equilibrium table and analyze
            model->GetEquilibrumTable();
            auto entry = model->Results(SCopyMap);
            
            // Store results
            results.push_back(std::tuple_cat(std::make_tuple(x_param), std::make_tuple(y_param), entry));
            
            std::cout << "Completed: " << config.x_param_range.name << "=" << x_param 
                     << ", " << config.y_param_range.name << "=" << y_param << std::endl;
        }
    }
    
    // Save results
    std::vector<std::string> headers = {
        config.x_param_range.name, config.y_param_range.name,
        "D(pmap(s,w)||peq(s,w))", "D(pmap(s)||peq(s))", "D(pmap(w)||peq(w))",
        "NHelixeq", "NHelixMap", "<U>eq", "<U>map",
        "Heq(s|w)", "Hmap(s|w)", "Heq(w)", "Hmap(w)", "Heq(s)", "Hmap(s)"
    };
    
    saveTuplesToCSV(results, outputFilename, headers);
    std::cout << "Error scan completed. Results saved to: " << outputFilename << std::endl;
}
