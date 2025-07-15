#include "FoldModels.h"
#include "EquilibriumPartitionMapGenerator.h"
#include "Utilities.h"      
#include "CustomMatrix.h"               
#include <omp.h>
#include <limits>

#ifndef NORMALIZATION_TOLERANCE
#define NORMALIZATION_TOLERANCE 1e-8
#endif

/**
 * @brief Validate matrix properties for physical consistency
 * 
 * Performs basic sanity checks on transfer matrices and boundary vectors
 * to ensure they represent valid statistical mechanical models.
 * 
 * @param transferMatrix Transfer matrix to validate
 * @param startVector Starting vector to validate  
 * @param endVector Ending vector to validate
 * @param modelName Name of the model (for error messages)
 * 
 * @throws std::invalid_argument if matrices fail validation
 * 
 * Validation checks:
 * - All matrix elements are non-negative (required for probabilities)
 * - Matrix is square and properly sized
 * - Start vector has positive elements that can be normalized
 * - End vector has non-negative elements
 * - No NaN or infinite values
 */
 
void validateMatrices(const Eigen::MatrixXd& transferMatrix,
                     const Eigen::RowVectorXd& startVector,
                     const Eigen::VectorXd& endVector,
                     const std::string& modelName) {
    
    // Check for NaN or infinite values
    if (!transferMatrix.allFinite()) {
        throw std::invalid_argument(modelName + ": Transfer matrix contains NaN or infinite values");
    }
    if (!startVector.allFinite()) {
        throw std::invalid_argument(modelName + ": Start vector contains NaN or infinite values");
    }
    if (!endVector.allFinite()) {
        throw std::invalid_argument(modelName + ": End vector contains NaN or infinite values");
    }
    
    // Check non-negativity (required for statistical weights)
    if ((transferMatrix.array() <= 0.0).any()) {
        throw std::invalid_argument(modelName + ": Transfer matrix contains non-positive values");
    }
    if ((endVector.array() <= 0.0).any()) {
        throw std::invalid_argument(modelName + ": End vector contains non-positive values");
    }
    if ((startVector.array() <= 0.0).any()) {
        throw std::invalid_argument(modelName + ": Start vector contains non-positive values");
    }
    
    // Check matrix is square
    if (transferMatrix.rows() != transferMatrix.cols()) {
        throw std::invalid_argument(modelName + ": Transfer matrix is not square");
    }
    
    // Check vector dimensions match matrix
    if (startVector.size() != transferMatrix.cols()) {
        throw std::invalid_argument(modelName + ": Start vector size doesn't match matrix columns");
    }
    if (endVector.size() != transferMatrix.rows()) {
        throw std::invalid_argument(modelName + ": End vector size doesn't match matrix rows");
    }
    
    std::cout << "Matrix validation passed for " << modelName << std::endl;
}

// Static member definitions for GFold
constexpr std::array<char, 16> GFold::INT_TO_CHAR;
constexpr std::array<int, 128> GFold::CHAR_TO_INT;

// GFold method implementations
double GFold::GetEigen(Eigen::MatrixXd weightMatrix) {
    Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces(weightMatrix);
    Eigen::VectorXcd eigenvals = ces.eigenvalues();
    double maxReal = eigenvals.real().maxCoeff();
    return(maxReal);
}

double GFold::getNumPartition(Eigen::RowVectorXd start, Eigen::VectorXd end, Eigen::MatrixXd weightMatrix) {
    Eigen::RowVectorXd run = start;
    for (int i = 0; i < L-1; i++) {
        //L-1 because start already counts the first monomer. end does NOT correspond to a monomer
        run = run * weightMatrix;
    }
    return(run.dot(end));
}

double GFold::eigenPartition() const {
    return std::pow(SWeigenMax, L);
}

double GFold::eigenG() const {
    return std::log(SWeigenMax);
}

double GFold::Gnum() const {
    return std::log(SWZnum)/L;
}

void GFold::CalcAllPartition() {
    SWZnum = getNumPartition(EqSWstart,EqSWend, EqSWMatrix);
}

void GFold::CalcAllEigen() {
    SWeigenMax = GetEigen(EqSWMatrix); 
}

void GFold::getPSWMatrices() {
    Eigen::MatrixXd CondProb_L_gvn_Lm1 = RowNormalize(multiplytoColVec(EqSWMatrix,EqSWend));
    Eigen::VectorXd backward_iter_vec = EqSWMatrix*EqSWend;

    EqProbList.resize(L);
    
    EqProbList[L-1] = CondProb_L_gvn_Lm1;
    std::cout << EqProbList[L-1] << std::endl;
    std::cout <<std::endl;
    for (int i = L-2; i > 0; i--) {
        EqProbList[i] = RowNormalize(multiplytoColVec(EqSWMatrix,backward_iter_vec));
        backward_iter_vec = EqSWMatrix*backward_iter_vec;
        std::cout << EqProbList[i] << std::endl;
        std::cout <<std::endl;
    }

    EqProbList[0] = RowNormalize(multiplytoColVec(EqSWstart,backward_iter_vec));
    std::cout << EqProbList[0] << std::endl;
    std::cout <<std::endl;
}

double GFold::CalcPSWfromMatrices(std::string seq) {
    double prob = EqProbList[0](CHAR_TO_INT[seq[0]]);
    for (int i = 1; i < L; i++) {
        prob = prob*EqProbList[i](CHAR_TO_INT[seq[i-1]],CHAR_TO_INT[seq[i]]);
    }

    return(prob);
}

double GFold::CalcSfromMatrices(std::string seq) {
    Eigen::RowVectorXd prob = (EqProbList[0].array()*SeqProbMapsSS[CHAR_TO_INT[seq[0]]].array()).matrix();
    for (int i = 1; i < L; i++) {
        Eigen::MatrixXd mapped_matrix = (EqProbList[i].array()*SeqProbMapsCond[CHAR_TO_INT[seq[i-1]]][CHAR_TO_INT[seq[i]]].array()).matrix();
        prob = prob*mapped_matrix;
    }

    return(prob.sum());
}

double GFold::CalcWfromMatrices(std::string seq) {
    Eigen::RowVectorXd prob = (EqProbList[0].array()*WeqProbMapsSS[CHAR_TO_INT[seq[0]]].array()).matrix();
    for (int i = 1; i < L; i++) {
        Eigen::MatrixXd mapped_matrix = (EqProbList[i].array()*WeqProbMapsCond[CHAR_TO_INT[seq[i-1]]][CHAR_TO_INT[seq[i]]].array()).matrix();
        prob = prob*mapped_matrix;
    }

    return(prob.sum());
}

// IsingVar method implementations
std::string IsingVar::GetS(std::string SW){
    std::string ans = "";
    for (char c : SW) {
        ans = ans + IsingSlookup.at(c);
    }
    return(ans);
}

std::string IsingVar::GetW(std::string SW){
    std::string ans = "";
    for (char c : SW) {
        ans = ans + IsingWlookup.at(c);
    }
    return(ans);
}

void IsingVar::GetEquilibrumTable() {
    if (equilibrium_defined) {
        throw std::runtime_error("Attempt to generate equilibrium table for model with equilibrium table defined.");
    }
    EquilibriumPartitionMapGenerator::generateWithPartition(EquilibriumTable,SEquilibriumMap,WEquilibriumMap,EquilibriumValidation,
        L, EqSWMatrix, EqSWstart, EqSWend, SWZnum, EqSWMatrix.rows(), IsingSlookup, IsingWlookup);
    equilibrium_defined = true;
}

void IsingVar::printEquilibriumTable() {
    // Print results
    std::cout << std::left 
            << std::setw(15) << "s,w" << std::setw(15) << "s" << std::setw(15) << "w"
            << std::setw(20) << "P(s,w)" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    for (const auto& [str, str1, str2, value, energy] : EquilibriumTable) {
        std::cout << std::left 
                << std::setw(15) << str << std::setw(15) << str1 << std::setw(15) << str2
                << std::scientific << std::setprecision(6) 
                << value << std::setw(15) << energy << std::endl;
    }
}

void IsingVar::printSEquilibriumMap() {
    std::cout << std::left 
            << std::setw(15) << "s" 
            << std::setw(20) << "P(s)" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    double sum = 0.0;
    for (const auto& [str, value] :  SEquilibriumMap) {
        std::cout << std::left 
                << std::setw(15) << str 
                << std::scientific << std::setprecision(6) 
                << value << std::endl;
        sum += value;
    }
    std::cout << sum << std:: endl;
}

void IsingVar::printWEquilibriumMap() {
    std::cout << std::left 
          << std::setw(15) << "w" 
          << std::setw(20) << "P(w)" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    double sum = 0.0;
    for (const auto& [str, value] :  WEquilibriumMap) {
        std::cout << std::left 
                << std::setw(15) << str 
                << std::scientific << std::setprecision(6) 
                << value << std::endl;
        sum += value;
    }

    std::cout << "Sum: " << sum << std::endl;
}

void IsingVar::printAll() {
    printEquilibriumTable();
    std::cout << "Sum: " << EquilibriumValidation << std::endl;
    std::cout << std::endl;
    printSEquilibriumMap();
    std::cout << std::endl;
    printWEquilibriumMap();
    std::cout << std::endl;
}

int IsingVar::countHelix(std::string fold) {
    int helix_count = 0;
    for (char c : fold) {
        if (c == '0'){
            helix_count++;
        }
    }
    return(helix_count);
}

double IsingVar::GetBernoulliProb(double Prob0, std::string sequence){
    double prob = 1;
    for (char c : sequence) {
        if (c == '0'){
            prob = prob*Prob0;
        } else {
            prob = prob*(1-Prob0);
        }
    }
    return(prob);
}

std::unordered_map<std::string,double> IsingVar::GenerateBernoulliMap(double Prob0,int len) {
    std::unordered_map<std::string,double>  SCopyMap;
    // Total number of strings will be 2^L
    int n_strings = 1 << len;  // Same as 2^L
    SCopyMap.reserve(n_strings);
    // Generate each number from 0 to 2^L - 1
    for(int i = 0; i < n_strings; i++) {
        std::string binary;
        
        // Convert to binary by checking each bit
        for(int j = len-1; j >= 0; j--) {
            // Check if jth bit is set
            binary += (i & (1 << j)) ? '1' : '0';
        }
        
        SCopyMap[binary] = GetBernoulliProb(Prob0,binary);
    }
    return SCopyMap;
}

std::unordered_map<std::string,double> IsingVar::SampleRandomProb(unsigned int seed, int len) {
    std::unordered_map<std::string,double>  SCopyMap;
    
     // Total number of strings will be 2^L

    int n_strings = 1 << len;  // Same as 2^L
    SCopyMap.reserve(n_strings);
    
    std::mt19937_64 gen(seed);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double normalization = 0.0;
     // Generate each number from 0 to 2^L - 1
    for(int i = 0; i < n_strings; i++) {
        std::string binary;
        
        // Convert to binary by checking each bit
        for(int j = len-1; j >= 0; j--) {
            // Check if jth bit is set
            binary += (i & (1 << j)) ? '1' : '0';
        }
        
        SCopyMap[binary] = -log(dist(gen));
        normalization += SCopyMap[binary];
    }

    for (const auto& pair : SCopyMap) {
        SCopyMap[pair.first] = SCopyMap[pair.first]/normalization;
    }
   
    return SCopyMap;
}

std::unordered_map<std::string,double> IsingVar::GenerateFromUniformTemplate(std::vector<std::string> template_ls, double err_rate) {
    std::unordered_map<std::string,double>  SCopyMap;
    double template_prob = 1/double(template_ls.size());
    // Total number of strings will be 2^L
    int len = template_ls[0].size();
    int n_strings = 1 << len;  // Same as 2^L
    SCopyMap.reserve(n_strings);
    // Generate each number from 0 to 2^L - 1
    for(int i = 0; i < n_strings; i++) {
        std::string binary;
        
        // Convert to binary by checking each bit
        for(int j = len-1; j >= 0; j--) {
            // Check if jth bit is set
            binary += (i & (1 << j)) ? '1' : '0';
        }
        
        for (auto template_str: template_ls) {
            int differences = 0;
            for (int i = 0; i < template_str.length(); i++) {
                if (template_str[i] != binary[i]) {
                    differences++;
                }
            }
            //std::cout << template_prob << " " << pow(err_rate,differences) << " " << pow(1-err_rate,len-differences) << std::endl;
            SCopyMap[binary] += template_prob*pow(err_rate,differences)*pow(1-err_rate,len-differences);
        } 
        //std::cout << binary << " " << SCopyMap[binary] << std::endl;
    }
    return SCopyMap;
}

double IsingVar::CalcFoldEnergy(std::string foldseq) {
    double energy = 0.0;
    for (int i = 1; i < L; i++) { 
            //minus sign because we used positive energy
        energy = energy - log(EqSWMatrix(CHAR_TO_INT[foldseq[i-1]],CHAR_TO_INT[foldseq[i]]));
    }
    energy = energy-log(EqSWend[CHAR_TO_INT[foldseq[L-1]]]);

    return energy;
}

std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> IsingVar::Results(std::unordered_map<std::string,double> SCopyMapIn) {
    double KL_SW = 0.0;
    double KL_S = 0.0;
    double KL_W = 0.0;

    double validation = 0.0;

    double EWuni = 0.0;
    double EWeq = 0.0;

    double eqEnergy = 0.0;
    double uniEnergy = 0.0;

    double eqCondEntropy = 0.0;
    double uniCondEntropy = 0.0;

    double HWeq = 0.0;
    double HWcopy = 0.0;

    double HSeq = 0.0;
    double HScopy = 0.0;

    std::string fold_idx;

    std::unordered_map<std::string,double> WCopyMap; 
        WCopyMap.reserve(WEquilibriumMap.size());
    std::unordered_map<std::string,double> SCopyMap = SCopyMapIn;

    for (const auto& pair : SEquilibriumMap) {
        validation += SCopyMap[pair.first];
        HScopy += -SCopyMap[pair.first]*log(SCopyMap[pair.first]);
        HSeq += -pair.second*log(pair.second);
        if (SCopyMap[pair.first] > 0) { // This zero handling matters when we are crossing out of the positive orthant
            KL_S += SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second);
        }
        // std::cout << pair.first << " " << SCopyMap[pair.first] << " "<< SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second) << " " << KL_S << std::endl;
    }

    if (std::abs(validation - 1.0) > NORMALIZATION_TOLERANCE) {
        std::ostringstream oss;
        oss << "SCopy: Probability sum  " << std::fixed << std::setprecision(15) << validation <<
            " deviates from 1.0 by more than tolerance " << std::fixed << std::setprecision(15) << NORMALIZATION_TOLERANCE;
        throw std::runtime_error(oss.str());
    }
    
    #ifndef NDEBUG
        #ifdef _OPENMP
        std::cout << "OpenMP is enabled" << std::endl;
        #else
        std::cout << "OpenMP is not enabled" << std::endl;
        #endif
    #endif

    KL_SW = 0.0;
    // Parallelize stuff that can be parallelized
    #pragma omp parallel for reduction(+:KL_SW,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy)
    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& s_state = std::get<1>(swdata);
        const auto& w_state = std::get<2>(swdata);
        const double foldEn = std::get<4>(swdata);
        const double s_copy_prob = SCopyMap.at(s_state);
        const double s_eq_prob = SEquilibriumMap.at(s_state);
        const double w_eq_prob = WEquilibriumMap[w_state];
        
        const double pcopySW = peqSW * s_copy_prob / s_eq_prob;

        if (pcopySW > 0) { // This zero handling matters when we are crossing out of the positive orthant
            KL_SW += pcopySW * log(pcopySW/peqSW);
            uniCondEntropy += -pcopySW * log(pcopySW);
        }
        
        eqEnergy += peqSW * foldEn;
        uniEnergy += pcopySW * foldEn;
        
        eqCondEntropy += -peqSW * log(peqSW/w_eq_prob);
    }
    
    // Then do WCopyMap updates serially
    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& s_state = std::get<1>(swdata);
        const auto& w_state = std::get<2>(swdata);
        const double s_copy_prob = SCopyMap[s_state];
        const double s_eq_prob = SEquilibriumMap[s_state];
        
        const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
        WCopyMap[w_state] += pcopySW;
    }
    
    validation = 0;
    for (const auto& pair : WEquilibriumMap) {
        validation += WCopyMap[pair.first];
        KL_W += WCopyMap[pair.first]*log(WCopyMap[pair.first]/pair.second);
        double helix_count = countHelix(pair.first);

        HWcopy += -WCopyMap[pair.first]*log(WCopyMap[pair.first]);
        HWeq += -pair.second*log(pair.second);

        EWuni += helix_count*WCopyMap[pair.first];
        EWeq += helix_count*pair.second;
    }
    uniCondEntropy = uniCondEntropy - HWcopy;
    if (std::abs(validation - 1.0) > NORMALIZATION_TOLERANCE) {
        throw std::runtime_error("WCopy: Probability sum " + std::to_string(validation) + 
                                " deviates from 1.0 by more than tolerance " + std::to_string(NORMALIZATION_TOLERANCE));
    }
    return( std::make_tuple(KL_SW,KL_S,KL_W,EWeq,EWuni,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy,HWeq,HWcopy,HSeq,HScopy) );
}

std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> IsingVar::ResultsSaveFoldMap(std::unordered_map<std::string,double> SCopyMapIn, std::string filename) {
    double KL_SW = 0.0;
    double KL_S = 0.0;
    double KL_W = 0.0;

    double validation = 0.0;

    double EWuni = 0.0;
    double EWeq = 0.0;

    double eqEnergy = 0.0;
    double uniEnergy = 0.0;

    double eqCondEntropy = 0.0;
    double uniCondEntropy = 0.0;

    double HWeq = 0.0;
    double HWcopy = 0.0;

    double HSeq = 0.0;
    double HScopy = 0.0;

    std::string fold_idx;

    std::unordered_map<std::string,double> WCopyMap; 
        WCopyMap.reserve(WEquilibriumMap.size());
    std::unordered_map<std::string,double> SCopyMap = SCopyMapIn;

    for (const auto& pair : SEquilibriumMap) {
        validation += SCopyMap[pair.first];
        HScopy += -SCopyMap[pair.first]*log(SCopyMap[pair.first]);
        HSeq += -pair.second*log(pair.second);
        KL_S += SCopyMap[pair.first]*log(SCopyMap[pair.first]/pair.second);
    }

    if (std::abs(validation - 1.0) > NORMALIZATION_TOLERANCE) {
        std::ostringstream oss;
        oss << "SCopy: Probability sum  " << std::fixed << std::setprecision(15) << validation <<
            " deviates from 1.0 by more than tolerance " << std::fixed << std::setprecision(15) << NORMALIZATION_TOLERANCE;
        throw std::runtime_error(oss.str());
    }
    
    #ifdef _OPENMP
    std::cout << "OpenMP is enabled" << std::endl;
    #else
    std::cout << "OpenMP is not enabled" << std::endl;
    #endif

    KL_SW = 0.0;
    // Parallelize stuff that can be parallelized
    #pragma omp parallel for reduction(+:KL_SW,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy)
    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& s_state = std::get<1>(swdata);
        const auto& w_state = std::get<2>(swdata);
        const double foldEn = std::get<4>(swdata);
        const double s_copy_prob = SCopyMap.at(s_state);
        const double s_eq_prob = SEquilibriumMap.at(s_state);
        const double w_eq_prob = WEquilibriumMap[w_state];
        
        const double pcopySW = peqSW * s_copy_prob / s_eq_prob;

        KL_SW += pcopySW * log(pcopySW/peqSW);
        
        eqEnergy += peqSW * foldEn;
        uniEnergy += pcopySW * foldEn;
        
        eqCondEntropy += -peqSW * log(peqSW/w_eq_prob);
        uniCondEntropy += -pcopySW * log(pcopySW);
    }
    
    // Then do WCopyMap updates serially
    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& s_state = std::get<1>(swdata);
        const auto& w_state = std::get<2>(swdata);
        const double s_copy_prob = SCopyMap[s_state];
        const double s_eq_prob = SEquilibriumMap[s_state];
        
        const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
        WCopyMap[w_state] += pcopySW;
    }
    
    validation = 0;
    for (const auto& pair : WEquilibriumMap) {
        validation += WCopyMap[pair.first];
        KL_W += WCopyMap[pair.first]*log(WCopyMap[pair.first]/pair.second);
        double helix_count = countHelix(pair.first);

        HWcopy += -WCopyMap[pair.first]*log(WCopyMap[pair.first]);
        HWeq += -pair.second*log(pair.second);

        EWuni += helix_count*WCopyMap[pair.first];
        EWeq += helix_count*pair.second;
    }
    uniCondEntropy = uniCondEntropy - HWcopy;
    
    saveMapToTSV(WEquilibriumMap, "FOLDEq"+filename);
    saveMapToTSV(WCopyMap, "FOLDMap"+filename);
    
    if (std::abs(validation - 1.0) > NORMALIZATION_TOLERANCE) {
        throw std::runtime_error("WCopy: Probability sum " + std::to_string(validation) + 
                                " deviates from 1.0 by more than tolerance " + std::to_string(NORMALIZATION_TOLERANCE));
    }
    return( std::make_tuple(KL_SW,KL_S,KL_W,EWeq,EWuni,eqEnergy,uniEnergy,eqCondEntropy,uniCondEntropy,HWeq,HWcopy,HSeq,HScopy) );
}

void IsingVar::VerifyMatrixApproach() {
    std::cout << "Setting up matrices..." << std::endl;
    
    getPSWMatrices();
    getSEqMatrices();
    getWEqMatrices();

    std::cout << "Verifying..." << std:: endl;
    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& sw_state = std::get<0>(swdata);
        
        if (abs(CalcPSWfromMatrices(sw_state)-peqSW) > pow(10.0,-10)) {
            throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                + std::to_string(abs(CalcPSWfromMatrices(sw_state)-peqSW)) + " OBTAINED!");
        }
    }
    std::cout << "SW Equilibrium Verification succeeded"<< std::endl;
    for (const auto& sdata : SEquilibriumMap) {
        const double peqS = std::get<1>(sdata);
        const auto& s_state = std::get<0>(sdata);

        if (abs(CalcSfromMatrices(s_state)-peqS) > pow(10.0,-10)) {
            throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                + std::to_string(abs(CalcSfromMatrices(s_state)-peqS)) + " OBTAINED!");
        }
    }
    std::cout << "S Equilibrium Verification succeeded"<< std::endl;
    for (const auto& wdata : WEquilibriumMap) {
        const double peqW = std::get<1>(wdata);
        const auto& w_state = std::get<0>(wdata);
        
        if (abs(CalcWfromMatrices(w_state)-peqW) > pow(10.0,-10)) {
            throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                + std::to_string(abs(CalcWfromMatrices(w_state)-peqW)) + " OBTAINED!");
        }
        std::cout << w_state << "/" << peqW << "/" << CalcWfromMatrices(w_state) << std::endl; 
    }
    std::cout << "W Equilibrium Verification succeeded"<< std::endl;
}

void IsingVar::getSEqMatrices() {
    // Find maximum s index
    int SWalphabetsize = IsingSlookup.size();
    int maxs = 0;
    
    for (int i = 0; i < SWalphabetsize; i++) {
        int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
        maxs = std::max(maxs,s);
    }
    maxs = maxs+1;

    // Initialize our data structures
    Eigen::MatrixXd default_condmatrix(SWalphabetsize, SWalphabetsize);
    default_condmatrix.setZero();
    Eigen::RowVectorXd default_SSvector(SWalphabetsize);
    default_SSvector.setZero();

    SeqProbMapsCond.assign( maxs, std::vector<Eigen::MatrixXd>(maxs, default_condmatrix));
    SeqProbMapsSS.assign( maxs, default_SSvector );

    // Define our maps
    for (int i = 0; i < SWalphabetsize; i++) {
        int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
        SeqProbMapsSS[s](i) = 1;
        for (int j = 0; j < SWalphabetsize; j++) {
            int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
            SeqProbMapsCond[s][s2](i,j) = 1;
        }
    }
}

void IsingVar::getWEqMatrices() {
    // Find maximum s index
    int SWalphabetsize = IsingWlookup.size();
    int maxw = 0;
    
    for (int i = 0; i < SWalphabetsize; i++) {
        int w = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[i]]];
        maxw = std::max(maxw,w);
    }
    maxw = maxw+1;

    // Initialize our data structures
    Eigen::MatrixXd default_condmatrix(SWalphabetsize, SWalphabetsize);
    default_condmatrix.setZero();
    Eigen::RowVectorXd default_SSvector(SWalphabetsize);
    default_SSvector.setZero();

    WeqProbMapsCond.assign( maxw, std::vector<Eigen::MatrixXd>(maxw, default_condmatrix));
    WeqProbMapsSS.assign( maxw, default_SSvector );

    // Define our maps
    for (int i = 0; i < SWalphabetsize; i++) {
        int w = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[i]]];
        WeqProbMapsSS[w](i) = 1;
        for (int j = 0; j < SWalphabetsize; j++) {
            int w2 = CHAR_TO_INT[IsingWlookup[INT_TO_CHAR[j]]];
            WeqProbMapsCond[w][w2](i,j) = 1;
        }
    }
}

void IsingVar::getIndependentMatrixFractions(int SWalphabetsize,int ssize, int wsize, std::vector<double> probs, double epsilon) {
    // Initialize the VectorFractionList
    std::vector<std::vector<VectorFractionList>> start_init = std::vector(1,std::vector(SWalphabetsize,VectorFractionList(1,2, epsilon)));
    start_vec_frac = Matrix<VectorFractionList>(start_init);
        // Generate Matrices and end vectors for the calculation of sequence weights
    Eigen::MatrixXd default_matrix = Eigen::MatrixXd::Zero(ssize,ssize);
    Eigen::VectorXd default_vector = Eigen::VectorXd::Zero(ssize);
    
    std::vector<std::vector<Eigen::MatrixXd>> seq_matrices = std::vector(ssize,std::vector(ssize,default_matrix));
    std::vector<Eigen::VectorXd> end_vectors = std::vector(ssize,default_vector);
    
    for (int i = 0; i < SWalphabetsize; i++) {
        char skey = IsingSlookup[INT_TO_CHAR[i]];
        int s = CHAR_TO_INT[skey];
        char wkey = IsingWlookup[INT_TO_CHAR[i]];
        int w = CHAR_TO_INT[wkey];
        IsingWSlices[wkey].push_back(i);
        IsingSSlices[skey].push_back(i);
        end_vectors[s](w) = EqSWend(i);  
        for (int j = 0; j < SWalphabetsize; j++) {
            int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
            char wkey2 = IsingWlookup[INT_TO_CHAR[j]];
            int w2 = CHAR_TO_INT[wkey2];
            seq_matrices[s][s2](w,w2) = EqSWMatrix(i,j);
        }
    }
        // Generate Matrices of MatrixFractions, as well as end MatrixFractions
    std::vector<std::vector<MatrixFraction>> mat_frac_init = std::vector(SWalphabetsize,std::vector<MatrixFraction>(SWalphabetsize));
    std::vector<std::vector<MatrixFraction>> end_frac_init = std::vector(SWalphabetsize,std::vector<MatrixFraction>(1));

    for (int i = 0; i < SWalphabetsize; i++) {
        int s = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[i]]];
        end_frac_init[i][0] = MatrixFraction(probs[s]*EqSWend(i),end_vectors[s]);
        for (int j = 0; j < SWalphabetsize; j++) {
            int s2 = CHAR_TO_INT[IsingSlookup[INT_TO_CHAR[j]]];
            mat_frac_init[i][j] = MatrixFraction(probs[s]*EqSWMatrix(i,j),seq_matrices[s][s2]);
        }
    }

    indep_mat_frac = Matrix<MatrixFraction>(mat_frac_init);
    end_vec_frac = Matrix<MatrixFraction>(end_frac_init);
}

double IsingVar::CalcWCopyfromMatrixFrac(std::string seq) {
    Matrix<VectorFractionList> running = start_vec_frac.slice({0},IsingWSlices[seq[0]]);
    for (int i = 1; i < L; i++) { 
         running = running * indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]);
    }
    std::cout << running(0,0).ls.size() << std::endl;
    Matrix<MatrixFraction> sliced_end = end_vec_frac.slice(IsingWSlices[seq[L-1]],{0});
    double result = 0.0;
    for (int j = 0; j < sliced_end.rows; j++){
        result += running(0,j).Finisher(sliced_end(j,0));
    }
    return(result);
}

std::tuple<int,double> IsingVar::CalcWCopyandVectorComplexity(std::string seq) {
    Matrix<VectorFractionList> running = start_vec_frac.slice({0},IsingWSlices[seq[0]]);
    for (int i = 1; i < L; i++) { 
         running = running * indep_mat_frac.slice(IsingWSlices[seq[i-1]],IsingWSlices[seq[i]]);
    }
    Matrix<MatrixFraction> sliced_end = end_vec_frac.slice(IsingWSlices[seq[L-1]],{0});
    double result = 0.0;
    for (int j = 0; j < sliced_end.rows; j++){
        result += running(0,j).Finisher(sliced_end(j,0));
    }
    return(std::make_tuple(running(0,0).ls.size()+running(0,1).ls.size(),result));
}

std::vector<Eigen::MatrixXd> IsingVar::getQuenchedProbList(std::string sequence){
    std::vector<Eigen::MatrixXd> partition_list(L+1);
    std::vector<Eigen::MatrixXd> prob_list(L);

    partition_list[0] = sliceMatrix(EqSWstart,{0},IsingSSlices[sequence[0]]);
    
    for (int i = 1; i < L; i++) {
        partition_list[i]  = sliceMatrix(EqSWMatrix,IsingSSlices[sequence[i-1]],IsingSSlices[sequence[i]]);
    }
    partition_list[L] = sliceMatrix(EqSWend,IsingSSlices[sequence[L-1]],{0});

    Eigen::MatrixXd CondProb_end = RowNormalize(multiplytoColVec(partition_list[L-1],partition_list[L]));
    Eigen::VectorXd backward_iter_vec = partition_list[L-1]*partition_list[L];
    
    prob_list[L-1] = CondProb_end;

    for (int i = L-2; i > 0; i--) {
        prob_list[i] = RowNormalize(multiplytoColVec(partition_list[i],backward_iter_vec));
        backward_iter_vec = partition_list[i]*backward_iter_vec;
        backward_iter_vec = backward_iter_vec/backward_iter_vec.sum(); // Helps avoid blowing up 
    }
    prob_list[0] = RowNormalize(multiplytoColVec(partition_list[0],backward_iter_vec));

    return(prob_list);
}

double IsingVar::evalQuenchedProbList(std::string fold, std::vector<Eigen::MatrixXd> prob_list){
    double prob = prob_list[0](CHAR_TO_INT[fold[0]]);
    for (int i = 1; i < L; i++) {
        prob = prob*prob_list[i](CHAR_TO_INT[fold[i-1]],CHAR_TO_INT[fold[i]]);
    }

    return(prob);
}

std::unordered_map<std::string,double> IsingVar::SampleCloseSequenceDist(double tresh, std::mt19937_64& gen){
    std::unordered_map<std::string,double> SCopyMap; 
    double newTotal = 0.0;
    std::uniform_real_distribution<double> dist(1-tresh, 1+tresh);
    
    for (const auto& pair : SEquilibriumMap) {
        SCopyMap[pair.first] = SEquilibriumMap[pair.first]*dist(gen);
        newTotal += SCopyMap[pair.first];
    }

    for (const auto& pair : SCopyMap) {
        SCopyMap[pair.first] = SCopyMap[pair.first]/newTotal;
    }

    return(SCopyMap);
}

std::unordered_map<std::string,double> IsingVar::EquilibriumTangentSample(unsigned int seed){
    double dotProd = 0.0;

    std::unordered_map<std::string,double> initSamp;
    std::unordered_map<std::string,double> SqrtDirs; 
    std::normal_distribution<double> normal_dist(0,1);
    std::mt19937_64 gen(seed);

    for (const auto& pair : SEquilibriumMap) {
        initSamp[pair.first] = normal_dist(gen);
        dotProd += initSamp[pair.first]*sqrt(pair.second); 
    }

    double newMagSqr = 0.0;
    for (const auto& pair : SEquilibriumMap) {
        SqrtDirs[pair.first] = initSamp[pair.first] - dotProd*sqrt(pair.second);
        newMagSqr += pow(SqrtDirs[pair.first],2);
    }

    double magsqrValidate = 0.0;
    double dotValidate = 0.0;

    for (const auto& pair : SEquilibriumMap) {
        SqrtDirs[pair.first] = SqrtDirs[pair.first]/sqrt(newMagSqr); 
        dotValidate += SqrtDirs[pair.first]*sqrt(pair.second);
        magsqrValidate += pow(SqrtDirs[pair.first],2);
    }

    std::cout << "Orthogonality validation:" << dotValidate << std::endl;
    std::cout << "Sum validation:" << magsqrValidate << std::endl;

    return(SqrtDirs);
}

std::unordered_map<std::string,double> IsingVar::ArcWalk(std::unordered_map<std::string,double> tangentVec, double angle){
    std::unordered_map<std::string,double> NewProb;
    double Total = 0;
    for (const auto& pair : SEquilibriumMap) {
        NewProb[pair.first] = pow(sqrt(pair.second)*cos(angle) + tangentVec[pair.first]*sin(angle),2); 
        Total += NewProb[pair.first];
    }
    return(NewProb);
}

std::pair<double,double> IsingVar::PositiveOrthantLimits(std::unordered_map<std::string,double> tangentVec){
    std::unordered_map<std::string,double> NewProb;
    double lowlim = -3.142;
    double highlim = 3.142;

    double testlowlim;  
    double testhighlim;

    for (const auto& pair : SEquilibriumMap) {

        double test = atan(-sqrt(pair.second)/tangentVec[pair.first]);
        if (test<0) {
            testlowlim = test;
            testhighlim = test+3.142;
        } else {
            testhighlim = test;
            testlowlim = test-3.142;
        }
        if (testlowlim > lowlim){
            lowlim = testlowlim;
        }
        if (testhighlim < highlim){
            highlim = testhighlim;
        }
    }
    std::cout << lowlim << " " << highlim << std::endl;
    return(std::pair(lowlim,highlim));
}

std::pair<std::unordered_map<std::string,double>,double> IsingVar::GetVectorAndArc(std::unordered_map<std::string,double> SCopyMap){
    std::unordered_map<std::string,double> vector;
    double dotprod = 0.0;
    double eqMag = 0.0;
    double qMag = 0.0;
    for (const auto& pair : SEquilibriumMap) {
        dotprod += sqrt(pair.second)*sqrt(SCopyMap[pair.first]); 
        eqMag += pair.second;
        qMag += SCopyMap[pair.first];
    }
    eqMag = sqrt(eqMag);
    qMag = sqrt(qMag);
    double angle = acos(dotprod/(eqMag*qMag));
    
    for (const auto& pair : SEquilibriumMap) {
        vector[pair.first] = (sqrt(SCopyMap[pair.first])-sqrt(pair.second)*cos(angle))/sin(angle); 
    }

    return(std::pair(vector,angle));
}

// Implementations for explorations in Total Variation (TV) distance.
const double IsingVar::CalculateTotalVariationDistance(
    const std::unordered_map<std::string,double>& prob1,
    const std::unordered_map<std::string,double>& prob2) {
    
    double tv_distance = 0.0;
    
    // Use prob1 as reference and check that all keys exist in prob2
    for (const auto& pair : prob1) {
        const std::string& key = pair.first;
        double p1 = pair.second;
        
        auto it = prob2.find(key);
        if (it == prob2.end()) {
            throw std::runtime_error("Key '" + key + "' found in prob1 but not in prob2");
        }
        
        double p2 = it->second;

        tv_distance += std::abs(p1 - p2);

        //std::cout << pair.first << " " << pair.second << " " <<  it->second << " " <<  std::abs(p1 - p2) << std::endl;
    }
    
    return 0.5 * tv_distance;
}

std::unordered_map<std::string,double> IsingVar::TVWalk(
    const std::unordered_map<std::string,double>& targetProb,
    double targetTV) {
    
    // Calculate current TV distance from equilibrium to target
    double currentTV = CalculateTotalVariationDistance(SEquilibriumMap, targetProb);
    
    // Calculate interpolation parameter t
    // For linear interpolation: TV(P_eq, P(t)) = t * TV(P_eq, P_target)
    double t = (currentTV > 1e-15) ? targetTV / currentTV : 0.0;
    // std::cout << "t/" << currentTV << "/" << targetTV<< std::endl;
    // Delegate to parametric walk
    return TVWalkParametric(targetProb, t);
}

std::unordered_map<std::string,double> IsingVar::TVWalkParametric(
    const std::unordered_map<std::string,double>& targetProb,
    double t) {
    
    std::unordered_map<std::string,double> result;
    
    // Use equilibrium as reference for keys
    double total = 0.0;
    double eq_total = 0.0;
    double target_total = 0.0;
    for (const auto& pair : SEquilibriumMap) {
        const std::string& seq = pair.first;
        double p_eq = pair.second;
        
        auto it = targetProb.find(seq);
        if (it == targetProb.end()) {
            throw std::runtime_error("Sequence '" + seq + "' found in equilibrium but not in target distribution");
        }
        double p_target = it->second;
        
        double p_interpolated = (1.0 - t) * p_eq + t * p_target;
        
        result[seq] = p_interpolated;
        eq_total += p_eq;
        total += p_interpolated;
        target_total += p_target;
    }
    
    // Check if probabilities sum to approximately 1.0
    if (std::abs(total - 1.0) > NORMALIZATION_TOLERANCE) {
        std::ostringstream oss;
        oss << "TVWalkParametric: Probability sum " << std::fixed << std::setprecision(15) << total <<
            " deviates from 1.0 by more than tolerance " << std::fixed << std::setprecision(15) << NORMALIZATION_TOLERANCE
            << ". Equilibrium total: " << std::fixed << std::setprecision(15) << eq_total << " / " << "Target total: " 
            << std::fixed << std::setprecision(15) << target_total << "/ " << "t: " << t <<". Consider if the equilibrium and reference distributions are too close.";
        throw std::runtime_error(oss.str());
    }
    
    return result;
}

std::pair<double, double> IsingVar::FindTotalVariationDistanceRange(const std::unordered_map<std::string,double>& targetProb) {
    
    double max_t = std::numeric_limits<double>::infinity();   // Start with positive infinity
    double min_t = -std::numeric_limits<double>::infinity();  // Start with negative infinity
    
    // Use equilibrium as reference for keys
    for (const auto& pair : SEquilibriumMap) {
        const std::string& seq = pair.first;
        double p_eq = pair.second;
        
        auto it = targetProb.find(seq);
        if (it == targetProb.end()) {
            throw std::runtime_error("Sequence '" + seq + "' found in equilibrium but not in target distribution");
        }
        double p_target = it->second;
        
        // For interpolation: p_interpolated = (1-t) * p_eq + t * p_target
        // We need: 0 <= p_interpolated <= 1
        
        if (p_target != p_eq) {  // Avoid division by zero
            // Non-negativity constraint: (1-t) * p_eq + t * p_target >= 0
            // Rearranging: t * (p_target - p_eq) >= -p_eq
            if (p_target < p_eq) {
                // p_target - p_eq < 0, so dividing flips inequality: t <= bound
                double t_bound = -p_eq / (p_target - p_eq);
                max_t = std::min(max_t, t_bound);
            } else {
                // p_target - p_eq > 0, so inequality stays: t >= bound  
                double t_bound = -p_eq / (p_target - p_eq);
                min_t = std::max(min_t, t_bound);
            }
            
            // Upper bound constraint: (1-t) * p_eq + t * p_target <= 1
            // Rearranging: t * (p_target - p_eq) <= 1 - p_eq
            if (p_target > p_eq) {
                // p_target - p_eq > 0, so inequality stays: t <= bound
                double t_bound = (1.0 - p_eq) / (p_target - p_eq);
                max_t = std::min(max_t, t_bound);
            } else {
                // p_target - p_eq < 0, so dividing flips inequality: t >= bound
                double t_bound = (1.0 - p_eq) / (p_target - p_eq);
                min_t = std::max(min_t, t_bound);
            }
        }
    }
    
    // Validate that we have a valid range
    if (min_t > max_t) {
        throw std::runtime_error("FindTotalVariationDistanceRange: No valid t range exists (min_t=" + 
                               std::to_string(min_t) + " > max_t=" + std::to_string(max_t) + 
                               "). Distributions are incompatible.");
    }
    
    // Calculate TV distances at both extremes using linearity
    double tv_distance_full = CalculateTotalVariationDistance(SEquilibriumMap, targetProb);
    double max_tv = tv_distance_full * max_t;
    double min_tv = tv_distance_full * min_t;  // This will be negative since min_t < 0
    
    return std::make_pair(min_tv, max_tv);
}

// Ising2 constructor implementations
Ising2::Ising2(const Eigen::MatrixXd& transferMatrix,
       const Eigen::RowVectorXd& startVector,
       const Eigen::VectorXd& endVector,
       int l) {
  
    validateMatrices(transferMatrix, startVector, endVector, "Ising2");
    
    // Validate input dimensions
    if (transferMatrix.rows() != 4 || transferMatrix.cols() != 4) {
        throw std::invalid_argument("Ising2 requires 4x4 transfer matrix");
    }
    if (startVector.size() != 4) {
        throw std::invalid_argument("Ising2 requires 4-element start vector");
    }
    if (endVector.size() != 4) {
        throw std::invalid_argument("Ising2 requires 4-element end vector");
    }
    if (l <= 0) {
        throw std::invalid_argument("Polymer length must be positive");
    }
    
    // Set up state mappings (same as other Ising2 constructors)
    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '1'},
        {'3', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '0'},
        {'3', '1'}
    };
    
    // Set matrix values directly
    EqSWMatrix = transferMatrix;
    EqSWstart = startVector;
    EqSWend = endVector;
    L = l;
    
    // Initialize other members
    Qstart = Eigen::Vector2d::Zero();
    Qend = Eigen::Vector2d::Zero();
    
    //std::cout << EqSWMatrix << std::endl;
    // Calculate derived quantities
    CalcAllEigen();
    CalcAllPartition();
}

Ising2::Ising2(double w00, double w11, double w01, double w10, double v, int l) {
    EqSWstart =  Eigen::Vector4d(1, 1, 1, 1);
    EqSWend = Eigen::Vector4d(v, 1, v, 1);

    Qstart =  Eigen::Vector2d(0, 0);
    Qend = Eigen::Vector2d(0, 0);

    EqSWMatrix.resize(4, 4);

    std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '1'},
        {'3', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '0'},
        {'3', '1'}
    };

    EqSWMatrix << w00, v, w01, v,
                  1,   1, 1,   1,
                  w10, v, w11, v,
                  1,   1, 1,   1;
    
    L = l;

    CalcAllEigen();
    CalcAllPartition();
}
// Same parameter set, different matrices. placehold is a trick.
Ising2::Ising2(double w0, double w1, double c0, double c1, int l, std::string placehold) {
    EqSWstart =  Eigen::Vector4d(1, 1, 1, 1);
    EqSWend = Eigen::Vector4d(w0, c0, w1, c1);

    Qstart =  Eigen::Vector2d(0, 0);
    Qend = Eigen::Vector2d(0, 0);

    EqSWMatrix.resize(4, 4);

    std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '1'},
        {'3', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '0'},
        {'3', '1'}
    };

    EqSWMatrix << w0, w0, w0,  w0,
                  c0,  c0, c0,  c0,
                  w1,  w1, w1, w1,
                  c1,  c1, c1,  c1;
    
    L = l;

    CalcAllEigen();
    CalcAllPartition();
}

Ising2::Ising2(double w00, double w11, double c0, double c1, int l) {
    EqSWstart =  Eigen::Vector4d(1, 1, 1, 1);
    EqSWend = Eigen::Vector4d(c0, c0, c1, c1);

    Qstart =  Eigen::Vector2d(0, 0);
    Qend = Eigen::Vector2d(0, 0);

    EqSWMatrix.resize(4, 4);

    std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '1'},
        {'3', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '0'},
        {'3', '1'}
    };

    EqSWMatrix << w00, c0, c0,  c0,
                  c0,  c0, c0,  c0,
                  c1,  c1, w11, c1,
                  c1,  c1, c1,  c1;
    
    L = l;

    CalcAllEigen();
    CalcAllPartition();
}

Ising2::Ising2(unsigned int seed, int l) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(1.0, 10.0);
    
    EqSWstart =  Eigen::Vector4d(0, 0, 0, 0);
    EqSWend = Eigen::Vector4d(0, 0, 0, 0);

    Qstart =  Eigen::Vector2d(0, 0);
    Qend = Eigen::Vector2d(0, 0);

    EqSWMatrix.resize(4, 4);

    std::vector<std::vector<Eigen::MatrixXd>> newQuenchedWeightMatrix(2,std::vector<Eigen::MatrixXd>(2,Eigen::MatrixXd(2,2)));

    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '1'},
        {'3', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '0'},
        {'3', '1'}
    };

    // Fill the matrix with random integers
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EqSWMatrix(i, j) = dist(gen);
        }
    }
    
    for (int i = 0; i < 4; i++) {
        EqSWstart(i) = dist(gen);
        EqSWend(i) = dist(gen);
    }

    // std::cout << EqSWMatrix << std::endl;
    L = l;

    CalcAllEigen();
    CalcAllPartition();
}

// Ising2 method implementations
void Ising2::getBernoulliMatrixFractions(double bernoulli, double epsilon){
    int SWalphabetsize = IsingSlookup.size();

    // Initialize the sequence probabilities
    std::vector<double> probs = {bernoulli,1-bernoulli};
    
    // Since we're doing this for Ising2, these can be defined directly
    int wsize = 2;
    int ssize = 2;

    getIndependentMatrixFractions(SWalphabetsize,ssize, wsize, probs, epsilon);
}

std::tuple<std::string,std::string, double> Ising2::SampleQuenchedBernoulli(double p, std::mt19937_64& gen) {
    std::bernoulli_distribution d(p);
    
    std::string sequence;
    sequence.reserve(L);
    double ps = 0.0;
    for(int i = 0; i < L; i++) {
        bool samp = d(gen);
        sequence += samp ? '0' : '1';
        ps += samp? -log(p): -log(1-p);
    }

    double pw = 0.0;
    std::string fold = "";

    std::vector<Eigen::MatrixXd> prob_list = getQuenchedProbList(sequence);

    Eigen::RowVectorXd prob_vec = prob_list[0];

    int chosen = drawFromProbVector(prob_vec,gen);
    fold += INT_TO_CHAR[chosen];
    pw -= log(prob_vec(chosen));
    
    int chosen_new;
    for (int i = 1; i <L ;i++) {
        chosen_new = drawFromProbVector(prob_list[i].row(chosen),gen);
        pw -= log(prob_list[i](chosen,chosen_new));
        chosen = chosen_new;
        fold += INT_TO_CHAR[chosen];
    }

    return(std::make_tuple(sequence,fold,ps+pw));
}

std::vector<std::string> Ising2::SampleNBernoulliSequence(double p, int N, int L, unsigned int seed) {
    std::mt19937_64 gen(seed);
    std::bernoulli_distribution d(p);
    
    std::vector<std::string> sampled_sequences;
   
    for (int n = 0; n < N ; n++) {
        std::string sequence;
        sequence.reserve(L);
        for(int i = 0; i < L; i++) {
            bool samp = d(gen);
            sequence += samp ? '0' : '1';
        }
        sampled_sequences.push_back(sequence);
    }
    
    return(sampled_sequences);
}

std::vector<std::string> Ising2::SampleNBiasedBernoulliSequence(double p, int N, int L, unsigned int seed) {
    std::mt19937_64 gen(seed);
    std::bernoulli_distribution d(p);
    
    std::vector<std::string> sampled_sequences;
    int samp_last = seed%2 ? 0 : 1;
    
    for (int n = 0; n < N ; n++) {
        std::string sequence;
        sequence.reserve(L);
        for(int i = 0; i < L; i++) {
            bool samp = d(gen);
            sequence += samp ? INT_TO_CHAR[samp_last] : INT_TO_CHAR[1-samp_last];
        }
        sampled_sequences.push_back(sequence);
        std::cout << sequence <<std::endl;
    }
    
    return(sampled_sequences);
}

void Ising2::VerifyMatrixApproachQuenched(std::unordered_map<std::string,double> SCopyMap, double p, int trials) {
    std::cout << "Setting up matrices..." << std::endl;
    
    getPSWMatrices();

    std::cout << "Calculating quenched copy map..." << std::endl;
    
    std::unordered_map<std::string,double> WCopyMap; 
        WCopyMap.reserve(WEquilibriumMap.size());

    std::unordered_map<std::string,std::vector<Eigen::MatrixXd>> SProbLists;

    for (const auto& sdata : SCopyMap) {
        const auto& s_state = std::get<0>(sdata);
        SProbLists[s_state] = getQuenchedProbList(s_state);
    }

    double KL_SW = 0.0;
    double JointcopyEntropy = 0.0;
    double KL_W = 0.0;
    double FoldcopyEntropy = 0.0;

    for (const auto& swdata : EquilibriumTable) {
        const double peqSW = std::get<3>(swdata);
        const auto& s_state = std::get<1>(swdata);
        const auto& w_state = std::get<2>(swdata);
        const double foldEn = std::get<4>(swdata);
        const double s_copy_prob = SCopyMap.at(s_state);
        const double s_eq_prob = SEquilibriumMap.at(s_state);
        const double w_eq_prob = WEquilibriumMap[w_state];
        
        const double pcopySW = peqSW * s_copy_prob / s_eq_prob;
        WCopyMap[w_state] += pcopySW;

        KL_SW += pcopySW * log(pcopySW/peqSW);
        JointcopyEntropy += -pcopySW * log(pcopySW);

        const double calculated = SCopyMap[s_state]*evalQuenchedProbList(w_state, SProbLists[s_state]);
        if (abs(calculated-pcopySW) > pow(10.0,-10)) {
            throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                + std::to_string(abs(calculated-pcopySW)) + " OBTAINED!");
        }
    }

    std::cout << "SW Quenched Verified, now moving on to W Quenched.." << std::endl;

    for (const auto& wdata : WCopyMap) {
        const double pcopyW = std::get<1>(wdata);
        const auto& w_state = std::get<0>(wdata);
        
        KL_W += pcopyW*log(pcopyW/WEquilibriumMap[w_state]);
        FoldcopyEntropy += -pcopyW * log(pcopyW);

        if (abs(CalcWCopyfromMatrixFrac(w_state)-pcopyW) > pow(10.0,-10)) {
            throw std::runtime_error("VERIFICATION FAILED! ERROR OF " 
                + std::to_string(abs(CalcWCopyfromMatrixFrac(w_state)-pcopyW)) + " OBTAINED!");
        }
    }

    double joint_entropy_est = 0.0;
    double fold_entropy_est = 0.0;

    std::random_device rd;
    std::mt19937_64 gen(rd());

    for (int i = 0; i < trials; i++) {
        auto sample = SampleQuenchedBernoulli(p, gen);
        joint_entropy_est += 1.0/double(trials)*std::get<2>(sample);
        fold_entropy_est += -1.0/double(trials)*log(CalcWCopyfromMatrixFrac(std::get<1>(sample)));
    }
    std::cout << "Estimated Joint Entropy of " << joint_entropy_est << " Versus True " << JointcopyEntropy <<std::endl;
    std::cout << "Estimated Fold Entropy of " << fold_entropy_est << " Versus True " << FoldcopyEntropy <<std::endl;
}

std::tuple<double,double,double,double> Ising2::SampleBernoulliEntropies(double p, int trials, int seed) {
    getBernoulliMatrixFractions(0.1);

    Eigen::VectorXd joint_entropy_vec(trials);
    Eigen::VectorXd fold_entropy_vec(trials);

    std::mt19937_64 gen(seed);

    for (int i = 0; i < trials; i++) {
        auto sample = SampleQuenchedBernoulli(p, gen);
        double joint_sample = std::get<2>(sample)/L;
        double fold_sample = -log(CalcWCopyfromMatrixFrac(std::get<1>(sample)))/L;
        
        joint_entropy_vec(i) = joint_sample;
        fold_entropy_vec(i) = fold_sample;
    }

    double joint_mean = joint_entropy_vec.mean();
    double joint_stddev = sqrt((joint_entropy_vec.array() - joint_mean).square().sum() / (joint_entropy_vec.size() - 1));
    double fold_mean = fold_entropy_vec.mean();
    double fold_stddev = sqrt((fold_entropy_vec.array() - fold_mean).square().sum() / (fold_entropy_vec.size() - 1));
    return(std::make_tuple(joint_mean,joint_stddev,fold_mean,fold_stddev));
}

// Ising2S3F constructor implementations
Ising2S3F::Ising2S3F(const Eigen::MatrixXd& transferMatrix,
          const Eigen::RowVectorXd& startVector,
          const Eigen::VectorXd& endVector,
          int l) {
    validateMatrices(transferMatrix, startVector, endVector, "Ising2S3F");
  
    // Validate input dimensions  
    if (transferMatrix.rows() != 6 || transferMatrix.cols() != 6) {
        throw std::invalid_argument("Ising2S3F requires 6x6 transfer matrix");
    }
    if (startVector.size() != 6) {
        throw std::invalid_argument("Ising2S3F requires 6-element start vector");
    }
    if (endVector.size() != 6) {
        throw std::invalid_argument("Ising2S3F requires 6-element end vector");
    }
    if (l <= 0) {
        throw std::invalid_argument("Polymer length must be positive");
    }
    
    // Set up state mappings (same as other Ising2S3F constructors)
    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '0'},
        {'3', '1'},
        {'4', '1'},
        {'5', '1'}
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '2'},
        {'3', '0'},
        {'4', '1'},
        {'5', '2'}
    };
    
    // Set matrix values directly
    EqSWMatrix = transferMatrix;
    EqSWstart = startVector;
    EqSWend = endVector;
    L = l;
    
    // Initialize other members
    Qstart = Eigen::VectorXd::Zero(2);
    Qend = Eigen::VectorXd::Zero(2);
    
    // Calculate derived quantities
    CalcAllEigen();
    CalcAllPartition();
}

Ising2S3F::Ising2S3F(double w00, double w11, double a01, double a10, double v, int l) { 
    IsingSlookup = {
        {'0', '0'},
        {'1', '0'},
        {'2', '0'},
        {'3', '1'},
        {'4', '1'},
        {'5', '1'},
    };

    IsingWlookup = {
        {'0', '0'},
        {'1', '1'},
        {'2', '2'},
        {'3', '0'},
        {'4', '1'},
        {'5', '2'}
    };
    
    EqSWstart =  Eigen::VectorXd(6);
        EqSWstart << 1, 1, 1, 1, 1, 1;
    EqSWend = Eigen::VectorXd(6);
         EqSWend << v, v, 1, v, v, 1;
    EqSWMatrix.resize(6, 6);

    //Fold idx: helix 0, antihelix 1, coil 2; 
    //Order: 0 helix, 0 antihelix, 0 coil, 1 helix, 1 antihelix, 1 coil;
    EqSWMatrix << w00,  v, v,   v,   v, v,
                  v,    v, v,   v, a01, v, 
                  1,    1, 1,   1,   1, 1,
                  v,    v, v, w11,   v, v,
                  v,  a10, v,   v,   v, v,
                  1,    1, 1,   1,   1, 1;

    L = l;

    CalcAllEigen();
    CalcAllPartition();
}
