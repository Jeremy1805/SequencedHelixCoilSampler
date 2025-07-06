#include <iostream>
#include <string>
#include <stdexcept>
#include "Utilities.h"
#include "ConfigurableScanner.h"

/**
 * @brief Main entry point for configurable parameter scanning
 * 
 * Usage: ./configurable_scanner <config_file.json>
 * 
 * Reads a JSON configuration file and performs the specified scan type.
 * Output filename is automatically generated from the config filename.
 * 
 * Supported scan types:
 * - "bernoulli": Scans over Bernoulli sequence distributions
 * - "error": Scans over template-based sequence distributions with error rates
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " configs/bernoulli_scan.json" << std::endl;
        return 1;
    }
    
    std::string configFilename = argv[1];
    
    try {
        // Parse configuration file
        auto config = parseConfig(configFilename);
        
        // Generate output filename from config filename
        std::string outputFilename = configFilename;
        size_t lastDot = outputFilename.find_last_of(".");
        if (lastDot != std::string::npos) {
            outputFilename = outputFilename.substr(0, lastDot);
        }
        outputFilename += "_results.tsv";
        
        // Print configuration summary
        std::cout << "=== Configuration Summary ===" << std::endl;
        std::cout << "Config file: " << configFilename << std::endl;
        std::cout << "Scan type: " << config.scan_type << std::endl;
        std::cout << "Fold model: " << config.fold_model << std::endl;
        std::cout << "Length: " << config.length << std::endl;
        std::cout << "X parameter: " << config.x_param_range.name 
                  << " [" << config.x_param_range.start << " to " << config.x_param_range.end 
                  << " step " << config.x_param_range.step << "]" << std::endl;
        std::cout << "Y parameter: " << config.y_param_range.name 
                  << " [" << config.y_param_range.start << " to " << config.y_param_range.end 
                  << " step " << config.y_param_range.step << "]" << std::endl;
        std::cout << "Output file: " << outputFilename << std::endl;
        
        if (!config.templates.empty()) {
            std::cout << "Templates: ";
            for (const auto& tmpl : config.templates) {
                std::cout << tmpl << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=============================" << std::endl << std::endl;
        
        // Start timing
        Timer timer("Total scan time");
        
        // Route to appropriate scan function
        if (config.scan_type == "bernoulli") {
            performBernoulliScan(config, outputFilename);
            
        } else if (config.scan_type == "error") {
            performErrorScan(config, outputFilename);
            
        } else {
            throw std::runtime_error("Unknown scan type: " + config.scan_type + 
                                   ". Supported types: bernoulli, error");
        }
        
        std::cout << std::endl << "Scan completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
