#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
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
    if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <config_file.json> [output_directory]" << std::endl;
    std::cerr << "Example: " << argv[0] << " configs/bernoulli_scan.json" << std::endl;
    std::cerr << "Example: " << argv[0] << " configs/bernoulli_scan.json ./my_results" << std::endl;
    return 1;
    }
    
    std::string configFilename = argv[1];
    std::string outputDirectory;
    
    if (argc == 3) {
        outputDirectory = argv[2];
    } else {
        outputDirectory = "results";
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDirectory);
    
    try {
        // Parse configuration file
        auto config = parseConfig(configFilename);
        
        // Generate output filename from config filename
        std::filesystem::path filePath(configFilename);
        std::string baseName = filePath.stem().string();
        std::string outputFilename = (std::filesystem::path(outputDirectory) / (baseName + "_results.tsv") ).string();
        
        // Print configuration summary
        std::cout << "=== Configuration Summary ===" << std::endl;
        std::cout << "Config file: " << configFilename << std::endl;
        std::cout << "Scan type: " << config.scan_type << std::endl;
        std::cout << "Scan subtype: " << config.scan_subtype << std::endl;
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
        } else if (config.scan_type == "tvwalk") {
            performTVScan(config,outputFilename);
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
