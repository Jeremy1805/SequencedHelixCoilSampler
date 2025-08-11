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
    
    bool energy_params_required = true;

    // Parse basic fields
    config.scan_type = j.at("scan_type").get<std::string>();
    config.fold_model = j.at("fold_model").get<std::string>();
    config.length = j.at("length").get<int>();
    
    if (config.scan_type == "longentropyverify"){
        energy_params_required = false;
    }
    // Parse optionals
    if (j.contains("scan_subtype")) {
        config.scan_subtype = j.at("scan_subtype").get<std::string>();
        if (config.scan_subtype == "rand_param_rand_traj"){
            energy_params_required = false;
        }
        if (config.scan_subtype == "rand_all_low_rank"){
            energy_params_required = false;
        }
    }
    if (j.contains("fixed_error")) {
        config.fixed_error = j.at("fixed_error").get<double>();
    }
    if (j.contains("fixed_bernoulli")) {
        config.fixed_bernoulli = j.at("fixed_bernoulli").get<double>();
    }
     if (j.contains("num_templates")) {
        config.num_templates = j.at("num_templates").get<int>();
    }
    if (j.contains("x_var_is_log")) {
        config.x_var_is_log = j.at("x_var_is_log").get<bool>();
    }
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
    
    if (energy_params_required) {
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
    }
    // Parse templates (optional, for error scan)
    if (j.contains("templates")) {
        auto templates_json = j.at("templates");
        for (const auto& tmpl : templates_json) {
            std::string str_tmpl = tmpl.get<std::string>();
            if (str_tmpl.length() != size_t(config.length)) {
                throw std::runtime_error("Error scan: one or more templates inconsistent with polymer length");
            } else {
                config.templates.push_back(str_tmpl);
            }
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
        double true_x_param;
        if(config.x_var_is_log){
            true_x_param = pow(2,x_param);
        } else {
            true_x_param = x_param;
        }
        // Generate Bernoulli distribution for this x_param value
        std::unordered_map<std::string,double> SCopyMap;
        SCopyMap = IsingVar::GenerateBernoulliMap(true_x_param, config.length);
        
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
        double true_x_param;
        
        if(config.x_var_is_log){
            true_x_param = pow(2,x_param);
        } else {
            true_x_param = x_param;
        }
        
        // Generate template-based distribution for this x_param value
        std::unordered_map<std::string,double> SCopyMap = IsingVar::GenerateFromUniformTemplate(config.templates, true_x_param);
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

// Helper struct to encapsulate model creation logic
struct ModelCreator {
    static std::unique_ptr<IsingVar> createModel(const Config& config, 
                                                const std::map<std::string, double>& params) {
        auto matrix = buildMatrix(config.energy_matrix, params);
        auto start_vec = buildVector(config.start_vector, params);
        auto end_vec = buildVector(config.end_vector, params);
        
        if (config.fold_model == "Ising2") {
            return std::make_unique<Ising2>(matrix, start_vec.transpose(), end_vec, config.length);
        } else if (config.fold_model == "Ising2S3F") {
            return std::make_unique<Ising2S3F>(matrix, start_vec.transpose(), end_vec, config.length);
        } else {
            throw std::runtime_error("Unsupported fold model: " + config.fold_model);
        }
    }
    
    static std::unique_ptr<IsingVar> createRandomModel(const Config& config, int seed) {
        if (config.fold_model == "Ising2") {
            return std::make_unique<Ising2>(seed, config.length);
        } else {
            throw std::runtime_error("Unsupported fold model for random creation: " + config.fold_model);
        }
    }
};

// Helper function to add reference distribution result
void addReferenceResult(std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results,
                       const Config& config,
                       double y_param,
                       const std::unordered_map<std::string,double>& referenceMap,
                       const std::unique_ptr<IsingVar>& model) {
    auto entry = model->Results(referenceMap);
    double measured_tv = IsingVar::CalculateTotalVariationDistance(model->SEquilibriumMap, referenceMap);
    
    double x_value = config.x_var_is_log ? log2(measured_tv) : measured_tv;
    results.push_back(std::tuple_cat(std::make_tuple(x_value), std::make_tuple(y_param), entry, std::make_tuple("reference_dist")));
}

// Helper function to add TV walk result with boundary checking (now takes precomputed boundary)
void addTVWalkResultWithBoundary(std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results,
                               const Config& config,
                               double x_param,
                               double y_param,
                               const std::unordered_map<std::string,double>& referenceMap,
                               const std::unique_ptr<IsingVar>& model,
                               double min_tv_boundary,
                               double max_tv_boundary,
                               bool& boundary_reached) {
    double true_x_param = config.x_var_is_log ? pow(2, x_param) : x_param;
    
    if (true_x_param > max_tv_boundary) {
        // We've hit the positive boundary - use the maximum valid TV distance instead
        std::unordered_map<std::string,double> SCopyMap = model->TVWalk(referenceMap, max_tv_boundary);
        auto entry = model->Results(SCopyMap);
        
        // Store with actual achieved TV distance and boundary tag
        double actual_x_value = config.x_var_is_log ? log2(max_tv_boundary) : max_tv_boundary;
        results.push_back(std::tuple_cat(std::make_tuple(actual_x_value), std::make_tuple(y_param), entry, std::make_tuple("boundary")));
        
        std::cout << "Hit positive boundary at: " << config.x_param_range.name << "=" << actual_x_value 
                  << ", " << config.y_param_range.name << "=" << y_param << std::endl;
        
        boundary_reached = true;
        return;
    }
    
    // Note: min_tv_boundary available for future negative direction walks
    // Future check would be: if (true_x_param < min_tv_boundary) { ... }
    
    // Normal case - we can reach the requested TV distance
    std::unordered_map<std::string,double> SCopyMap = model->TVWalk(referenceMap, true_x_param);
    auto entry = model->Results(SCopyMap);
    
    results.push_back(std::tuple_cat(std::make_tuple(x_param), std::make_tuple(y_param), entry, std::make_tuple("none")));
    
    std::cout << "Completed: " << config.x_param_range.name << "=" << x_param 
              << ", " << config.y_param_range.name << "=" << y_param << std::endl;
}

// TV scan subtype (see function performTVscan for an overview) implementations

// Subtype of TV scan where energy matrix is varied
void performFixedErrorScan(const Config& config, 
                          std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results) {
    if (config.templates.empty()) {
        throw std::runtime_error("TV error scans require template sequences in config");
    }
    
    std::unordered_map<std::string,double> SReferenceMap = 
        IsingVar::GenerateFromUniformTemplate(config.templates, config.fixed_error);
    
    for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
        std::map<std::string, double> params;
        params["x"] = y_param;
        
        auto model = ModelCreator::createModel(config, params);
        model->GetEquilibrumTable();
        
        // Add reference result
        addReferenceResult(results, config, y_param, SReferenceMap, model);
        
        // Compute TV boundaries once for this trajectory
        auto [min_tv, max_tv] = model->FindTotalVariationDistanceRange(SReferenceMap);
        
        // Add TV walk results with boundary checking
        bool boundary_reached = false;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            if (boundary_reached) break; // Stop walking once boundary is hit
            addTVWalkResultWithBoundary(results, config, x_param, y_param, SReferenceMap, model, min_tv, max_tv, boundary_reached);
        }
    }
}

// Subtype of TV scan where energy matrix and error rate are fixed, while templates are sampled randomly
void performFixedParamRandomTemplateScan(const Config& config, 
                          std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results) {
    
    std::map<std::string, double> params;
    params["x"] = std::numeric_limits<double>::quiet_NaN();  // Should throw error if used, as fixed matrix expected

    std::unique_ptr<IsingVar> model = nullptr;
    try {
        model = ModelCreator::createModel(config, params);
    } catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "fixed_param_rand_template TV scans cannot run with variable energy matrices. Please use constant entries." << std::endl;
    }
    model->GetEquilibrumTable();
    for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
        
        std::vector<std::string> sampled_sequences = Ising2::SampleNBernoulliSequence(config.fixed_bernoulli,config.num_templates,config.length,y_param);

        std::unordered_map<std::string,double> SReferenceMap = 
            IsingVar::GenerateFromUniformTemplate(sampled_sequences, config.fixed_error);

        // Add reference result
        addReferenceResult(results, config, y_param, SReferenceMap, model);
        
        // Compute TV boundaries once for this trajectory
        auto [min_tv, max_tv] = model->FindTotalVariationDistanceRange(SReferenceMap);
        
        // Add TV walk results with boundary checking
        bool boundary_reached = false;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            if (boundary_reached) break; // Stop walking once boundary is hit
            addTVWalkResultWithBoundary(results, config, x_param, y_param, SReferenceMap, model, min_tv, max_tv, boundary_reached);
        }
    }

    std::vector<std::string> zero = {std::string(config.length, '0')};
    std::unordered_map<std::string,double> SReferenceMap = 
        IsingVar::GenerateFromUniformTemplate(zero, config.fixed_error);

    // Add reference result
    addReferenceResult(results, config, -1, SReferenceMap, model);
    
    // Compute TV boundaries once for this trajectory
    auto [min_tv, max_tv] = model->FindTotalVariationDistanceRange(SReferenceMap);
    
    // Add TV walk results with boundary checking
    bool boundary_reached = false;
    for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
        if (boundary_reached) break; // Stop walking once boundary is hit
        addTVWalkResultWithBoundary(results, config, x_param, -1, SReferenceMap, model, min_tv, max_tv, boundary_reached);
    }
}

// Subtype of TV scan where energy matrix is fixed and walk direction is randomized
void performFixedParamRandomTrajScan(const Config& config, 
                                   std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results) {
    if (config.y_param_range.name != "seed") {
        throw std::runtime_error("fixed_param_rand_traj TV scans always run with random seeds as the y parameter.");
    }
    
    std::map<std::string, double> params;
    params["x"] = std::numeric_limits<double>::quiet_NaN();  // Should throw error if used, as fixed matrix expected
    
    std::unique_ptr<IsingVar> model = nullptr;
    try {
        model = ModelCreator::createModel(config, params);
    } catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "fixed_param_rand_traj TV scans cannot run with variable energy matrices. Please use constant entries." << std::endl;
    }
    model->GetEquilibrumTable();
    
    for (int y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
        std::unordered_map<std::string,double> SReferenceMap = 
            IsingVar::SampleRandomProb(y_param, config.length);
        
        // Add reference result
        addReferenceResult(results, config, y_param, SReferenceMap, model);
        
        // Compute TV boundaries once for this trajectory
        auto [min_tv, max_tv] = model->FindTotalVariationDistanceRange(SReferenceMap);
        
        // Add TV walk results with boundary checking
        bool boundary_reached = false;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            if (boundary_reached) break; // Stop walking once boundary is hit
            
            addTVWalkResultWithBoundary(results, config, x_param, y_param, SReferenceMap, model, min_tv, max_tv, boundary_reached);
        }
    }
}

// Subtype of TV scan where energy matrix and walk direction are both randomized
void performRandomParamRandomTrajScan(const Config& config, 
                                    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>>& results) {
    if (config.y_param_range.name != "seed") {
        throw std::runtime_error("rand_param_rand_traj TV scans always run with random seeds as the y parameter.");
    }
    
    for (int y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
        
        auto model = ModelCreator::createRandomModel(config, y_param);
        model->GetEquilibrumTable();
        
        std::unordered_map<std::string,double> SReferenceMap = 
            IsingVar::SampleRandomProb(y_param+1234567, config.length); // Seed offset for independence
        
        // Add reference result
        addReferenceResult(results, config, y_param, SReferenceMap, model);
        
        // Compute TV boundaries once for this trajectory
        auto [min_tv, max_tv] = model->FindTotalVariationDistanceRange(SReferenceMap);
        
        // Add TV walk results with boundary checking
        bool boundary_reached = false;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            if (boundary_reached) break; // Stop walking once boundary is hit
            
            addTVWalkResultWithBoundary(results, config, x_param, y_param, SReferenceMap, model, min_tv, max_tv, boundary_reached);
        }
    }
}

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
void performTVScan(const Config& config, const std::string& outputFilename) {
    std::cout << "Starting TV scan..." << std::endl;
    
    // Validate x parameter name
    if (config.x_param_range.name != "tv_distance" && config.x_param_range.name != "log_tv_distance") {
        throw std::runtime_error("TV scans always run with total variation distance as the x parameter. "
                               "Please set the x param name as 'tv_distance' or 'log_tv_distance'.");
    }
    
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double,std::string>> results;
    
    // Dispatch to appropriate scan implementation
    if (config.scan_subtype == "fixed_error") {
        performFixedErrorScan(config, results);
    } else if (config.scan_subtype == "fixed_param_rand_traj") {
        performFixedParamRandomTrajScan(config, results);
    } else if (config.scan_subtype == "rand_param_rand_traj") {
        performRandomParamRandomTrajScan(config, results);
    } else if (config.scan_subtype == "fixed_param_rand_temp"){
        performFixedParamRandomTemplateScan(config,results);
    } else {
        throw std::runtime_error("TV Scan must be defined with one scan subtype from "
                               "'fixed_error', 'fixed_param_rand_traj','rand_param_rand_traj' or 'fixed_param_rand_temp'");
    }
    
    // Save results
    std::vector<std::string> headers = {
        config.x_param_range.name, config.y_param_range.name,
        "D(pmap(s,w)||peq(s,w))", "D(pmap(s)||peq(s))", "D(pmap(w)||peq(w))",
        "NHelixeq", "NHelixMap", "<U>eq", "<U>map",
        "Heq(s|w)", "Hmap(s|w)", "Heq(w)", "Hmap(w)", "Heq(s)", "Hmap(s)", "tags"
    };
    
    saveTuplesToCSV(results, outputFilename, headers);
    std::cout << "TV scan completed. Results saved to: " << outputFilename << std::endl;
}

/**
 * @brief Perform fixed weight, random bernoulli parameter reciprocal matrix verification scan
 * 
 * For a fixed energy matrix, test reciprocal matrix method for calculating p(omega)
 * by varying epsilon threshold and randomizing the bernoulli parameter and sampled fold. 
 * 
 * @param config Configuration containing scan parameters and model definition
 * @param outputFilename Base name for output file
 */
void performFWRBLongBernoulliVerify(const Config& config, const std::string& outputFilename) {
    std::vector<std::tuple<double,int,double,double>> tbl;
    
    for (int y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
        
        // First, define the model
        std::map<std::string, double> params;
        params["x"] = std::numeric_limits<double>::quiet_NaN();  // Should throw error if used, as fixed matrix expected

        std::unique_ptr<IsingVar> model = nullptr;
        try {
            model = ModelCreator::createModel(config, params);
        } catch (const std::invalid_argument& e) {
            std::cout << "Error: " << e.what() << std::endl;
            std::cout << "fixed_weight_rand_bernoulli Long Bernoulli scans cannot run with variable energy matrices. Please use constant entries." << std::endl;
        }

        // Second, sample a bernoulli parameter
        std::mt19937_64 gen(y_param);
        std::uniform_real_distribution<double> urand(0.0, 1.0);
        double ber = urand(gen);
        
        std::cout << "Bernoulli parameters sampled, now defining matrix reciprocals..." <<std::endl;
        // Third, define the matrix fractions
        model->getBernoulliMatrixFractions(ber,1e-300);
        
        std::cout << "Sampling fold..." <<std::endl;
        // Fourth, sample a fold. 
        std::mt19937_64 gen2(y_param+12345678); // separate seed for sampling the distribution, to ease reproducibility if needed
        std::tuple<std::string,std::string, double> tup = model->SampleQuenchedBernoulli(ber, gen2); 
        std::cout << "Fold sampled, now calculating fold probabilities..." <<std::endl;
        
        // Finally, find the probability of the fold
        bool first = true;
        double ref = 0.0;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            double true_x_param;
            if(config.x_var_is_log){
                true_x_param = std::pow(10,x_param); // base 10 used for this particular parameter set as it makes more sense
            } else {
                true_x_param = x_param;
            }
            model->UpdateEpsilon(true_x_param);
            std::tuple<int,double> res = model->CalcWCopyandVectorComplexity(std::get<1>(tup));
            std::cout << "Processing results..." <<std::endl;
            if (first) {
                ref = std::get<1>(res);
                first = false;
            }
            tbl.push_back(std::tuple_cat(std::make_tuple(x_param),res,std::make_tuple(ref)));
            std::cout << "results: " << x_param << "/"  << std::get<0>(res) << "/" << std::get<1>(res) << "/" << ref << "/" << std::get<1>(res) -  ref <<std::endl;
        }
    }

    saveTuplesToCSV(tbl, outputFilename,{"LogTolerance","VectorDirNum","pW","refpW"});
    
    std::cout << "Long Verification Completed. Results saved to: " << outputFilename << std::endl;
}

/**
 * @brief Perform variable weight, fixed bernoulli parameter reciprocal matrix verification scan
 * 
 * For varying energy matrices, test reciprocal matrix method for calculating p(omega)
 * and varying the bernoulli parameter and sampling a fold randomly. 
 * 
 * @param config Configuration containing scan parameters and model definition
 * @param outputFilename Base name for output file
 */
void performVWVBRFLongBernoulliVerify(const Config& config, const std::string& outputFilename){
    std::vector<std::tuple<double,int,double>> tbl;

    for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
        double true_x_param;
        if(config.x_var_is_log){
            true_x_param = std::pow(2,x_param);
        } else {
            true_x_param = x_param;
        }
        // First, define the model
        std::map<std::string, double> params;
        params["x"] = true_x_param;  // Should throw error if used, as fixed matrix expected
        double epsilon = config.fixed_error; // trick: using the fixed error parameter as epsilon

        for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
            std::cout << "beta: " << x_param << " bernoulli_param: " << y_param << std::endl; 
            //First, Define Model
            std::unique_ptr<IsingVar> model = nullptr;
            try {
                model = ModelCreator::createModel(config, params);
            } catch (const std::invalid_argument& e) {
                std::cout << "Error: " << e.what() << std::endl;
                std::cout << "fixed_weight_rand_bernoulli Long Bernoulli scans cannot run with variable energy matrices. Please use constant entries." << std::endl;
            }
        
            // Second, Get Matrix Fractions with y_param as ber
            model->getBernoulliMatrixFractions(y_param,epsilon);
            
            // Third, sample a fold. 
            std::mt19937_64 gen(0); // zero seeded for this, assume typical limit
            std::tuple<std::string,std::string, double> tup = model->SampleQuenchedBernoulli(y_param, gen); 
            std::cout << "Fold sampled, now calculating fold probabilities..." <<std::endl;
            // Finally,calculate. 
            std::tuple<int,double> res = model->CalcWCopyandVectorComplexity(std::get<1>(tup));
            tbl.push_back(std::tuple_cat(std::make_tuple(x_param),res));

            std::cout << "results: " << x_param << "/"  << std::get<0>(res) << "/" << std::get<1>(res) << std::endl;
        }
    }
    saveTuplesToCSV(tbl, outputFilename,{"LogTolerance","VectorDirNum","pW"});
    
    std::cout << "Long Verification Completed. Results saved to: " << outputFilename << std::endl;
}

/**
 * @brief Perform random reciprocal matrix verification scan
 * 
 * For random energy matrices with low rank off diagonal submatrices, 
 * test reciprocal matrix method for calculating p(omega) and random 
 * bernoulli parameter and sampling a fold randomly. 
 * 
 * @param config Configuration containing scan parameters and model definition
 * @param outputFilename Base name for output file
 */
void performRandLowRankLongBernoulliVerify(const Config& config, const std::string& outputFilename){
    std::vector<std::tuple<double,int,double,double>> tbl;

    for (double y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {    
        // First, define a model
        
        if  (config.fold_model!="Ising2") {
            throw std::runtime_error("low_rank_rand_all reciprocal matrix verification scan only takes Ising2 as model.");
        }
        
        Ising2 model = Ising2(y_param,config.length,"low_rank_off_diagonal");

        // Second, sample a bernoulli parameter
        std::mt19937_64 gen(y_param+123);
        std::uniform_real_distribution<double> urand(0.0, 1.0);
        double ber = urand(gen);

        // Third, Get Matrix Fractions
        model.getBernoulliMatrixFractions(ber,1e-300);

        // Fourth, sample a fold. 
        std::mt19937_64 gen2(y_param+1234567);
        std::tuple<std::string,std::string, double> tup = model.SampleQuenchedBernoulli(ber, gen2);

        bool first = true;
        double ref = 0.0;
        for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
            double true_x_param;
            if(config.x_var_is_log){
                true_x_param = std::pow(10,x_param);
            } else {
                true_x_param = x_param;
            }
            
            // Update epsilon
            model.UpdateEpsilon(true_x_param);
            
            // Finally,calculate. 
            std::tuple<int,double> res = model.CalcWCopyandVectorComplexity(std::get<1>(tup));
            std::cout << "Processing results..." <<std::endl;
            if (first) {
                ref = std::get<1>(res);
                first = false;
            }
            tbl.push_back(std::tuple_cat(std::make_tuple(x_param),res,std::make_tuple(ref)));
            std::cout << "results: " << x_param << "/"  << std::get<0>(res) << "/" << std::get<1>(res) << "/" << ref << "/" << std::get<1>(res) -  ref <<std::endl;
        }
    }
    
    saveTuplesToCSV(tbl, outputFilename,{"LogTolerance","VectorDirNum","pW","refpW"});
    
    std::cout << "Long Verification Completed. Results saved to: " << outputFilename << std::endl;
}

/**
 * @brief Perform reciprocal matrix verification scan
 *
 * @param config Configuration containing scan parameters and template sequences
 * @param outputFilename Base name for output file
 */
void performLongBernVerifyScan(const Config& config, const std::string& outputFilename) {
    std::cout << "Starting Long verify scan..." << std::endl;

    // Dispatch to appropriate scan implementation
    if (config.scan_subtype == "fixed_weight_rand_ber_fold") {
        performFWRBLongBernoulliVerify(config, outputFilename);
    } else if (config.scan_subtype == "var_weight_var_bern_fold_sample") {
        performVWVBRFLongBernoulliVerify(config, outputFilename);
    } else if (config.scan_subtype == "rand_all_low_rank")
        performRandLowRankLongBernoulliVerify(config, outputFilename);
    else {
        throw std::runtime_error("Long verify scan must run with one scan subtype from "
                               "'fixed_weight_rand_ber_fold', 'var_weight_fixed_bern_fold_sample','rand_all_low_rank'");
    }
}

/**
 * @brief Perform reciprocal matrix verification scan of fold and joint entropy
 *
 * For random energy matrices and short lengths, calculate H(S,W) and H(W)
 * by exact enumeration, by using reciprocal/quenched transfer matrices and
 * then averaging exactly over all pairs/folds, and by sampling pairs/folds. 
 * exact_enumeration == exact_average is expected, while |exact_enumeration-sampled_average| 
 * is expected to be small
 *
 * @param config Configuration containing scan parameters and template sequences
 * @param outputFilename Base name for output file
 */
void performLongEntropyVerify(const Config& config, const std::string& outputFilename) {
    std::cout << "Starting Long Bernoulli scan..." << std::endl;
    
    std::vector<std::tuple<double,int,double,double,double,double,double,double>> results;
    
    // Nested loops over parameter ranges
    for (double x_param = config.x_param_range.start; x_param < config.x_param_range.end; x_param += config.x_param_range.step) {
        double true_x_param;
        if(config.x_var_is_log){
            true_x_param = pow(2,x_param);
        } else {
            true_x_param = x_param;
        }
        // Generate Bernoulli distribution for this x_param value
        std::unordered_map<std::string,double> SCopyMap;
        SCopyMap = IsingVar::GenerateBernoulliMap(true_x_param, config.length);
        
        for (int y_param = config.y_param_range.start; y_param < config.y_param_range.end; y_param += config.y_param_range.step) {
            
            if (config.fold_model != "Ising2") {
                throw std::runtime_error("Unsupported fold model: " + config.fold_model);
            } 
            
            Ising2 model(y_param,config.length); // randomized fold weights

            // Generate equilibrium table and analyze
            model.GetEquilibrumTable();
            model.getBernoulliMatrixFractions(true_x_param,1e-300);
            auto entry = model.VerifyMatrixApproachQuenched(SCopyMap,true_x_param,100);
            
            // Store results
            results.push_back(std::tuple_cat(std::make_tuple(x_param), std::make_tuple(y_param), entry));
            
            std::cout << "Completed: " << config.x_param_range.name << "=" << x_param 
                     << ", " << config.y_param_range.name << "=" << y_param << std::endl;
        }
    }
    
    // Save results
    std::vector<std::string> headers = {
        config.x_param_range.name, config.y_param_range.name,
        "HJointEst", "HJointInf", "HJointTrue",
        "HWEst", "HWInf", "HWTrue"
    };
    
    saveTuplesToCSV(results, outputFilename, headers);
    std::cout << "Long Bernoulli scan completed. Results saved to: " << outputFilename << std::endl;
}