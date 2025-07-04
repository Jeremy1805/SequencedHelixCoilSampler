#ifndef FOLDMODELS_H
#define FOLDMODELS_H

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <string>
#include <unordered_map>
#include <tuple>
#include <array>
#include "CustomMatrix.h"

#ifndef DEFAULT_EPSILON_DEFINED
#define DEFAULT_EPSILON_DEFINED
inline constexpr double DEFAULT_EPSILON = 1e-300;
#endif // DEFAULT_EPSILON_DEFINED

/**
 * @class GFold
 * @brief Base class for general protein folding models using statistical mechanics
 */
class GFold {
public:
    static constexpr std::array<char, 16> INT_TO_CHAR = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f'
    };
    
    static constexpr std::array<int, 128> CHAR_TO_INT = []() {
        std::array<int, 128> arr{};
        for (int i = 0; i < 10; ++i) arr['0' + i] = i;
        for (int i = 0; i < 6; ++i) arr['a' + i] = 10 + i;
        return arr;
    }();

    // Weight Matrices for Equilibrium and Quenched Disorder
    Eigen::MatrixXd EqSWMatrix;
    std::vector<std::vector<Eigen::MatrixXd>> QuenchedWeightMatrix;
    
    // Start and end vectors, eigenvalues, partition func
    Eigen::RowVectorXd EqSWstart;
    Eigen::VectorXd EqSWend;
    double SWeigenMax;
    double SWZnum;

    // Equilibrium probability matrices
    std::vector<Eigen::MatrixXd> EqProbList;
    std::vector<std::vector<Eigen::MatrixXd>> SeqProbMapsCond;
    std::vector<Eigen::RowVectorXd> SeqProbMapsSS;
    std::vector<Eigen::MatrixXd> WeqProbList;
    std::vector<std::vector<Eigen::MatrixXd>> WeqProbMapsCond;
    std::vector<Eigen::RowVectorXd> WeqProbMapsSS;

    // For Bernoulli p(w) evaluations using matrix fractions
    Matrix<VectorFractionList> start_vec_frac;
    Matrix<MatrixFraction> indep_mat_frac;
    Matrix<MatrixFraction> end_vec_frac;

    // Start and End for Quenched Disorder
    Eigen::RowVectorXd Qstart;
    Eigen::VectorXd Qend;

    int L; // Real polymer length
    
    // Complete equilibrium table
    std::vector<std::tuple<std::string, std::string, std::string, double, double>> EquilibriumTable;
    std::unordered_map<std::string, double> SEquilibriumMap;
    std::unordered_map<std::string, double> WEquilibriumMap;
    double EquilibriumValidation;

    // Method declarations
    double GetEigen(Eigen::MatrixXd weightMatrix);
    double getNumPartition(Eigen::RowVectorXd start, Eigen::VectorXd end, Eigen::MatrixXd weightMatrix);
    double eigenPartition() const;
    double eigenG() const;
    double Gnum() const;
    void CalcAllPartition();
    void CalcAllEigen();
    void getPSWMatrices();
    double CalcPSWfromMatrices(std::string seq);
    double CalcSfromMatrices(std::string seq);
    double CalcWfromMatrices(std::string seq);
};

/**
 * @class IsingVar
 * @brief Ising model variant with sequence-to-fold mapping functionality
 */
class IsingVar : public GFold {
public:
    std::unordered_map<char, char> IsingSlookup;
    std::unordered_map<char, char> IsingWlookup;
    std::unordered_map<char, std::vector<size_t>> IsingSSlices;
    std::unordered_map<char, std::vector<size_t>> IsingWSlices;

    // Method declarations
    std::string GetS(std::string SW);
    std::string GetW(std::string SW);
    void GetEquilibrumTable();
    void printEquilibriumTable();
    void printSEquilibriumMap();
    void printWEquilibriumMap();
    void printAll();
    
    static int countHelix(std::string fold);
    static double GetBernoulliProb(double Prob0, std::string sequence);
    static std::unordered_map<std::string,double> GenerateBernoulliMap(double Prob0, int len);
    static std::unordered_map<std::string,double> SampleRandomProb(unsigned int seed, int len);
    static std::unordered_map<std::string,double> GenerateFromUniformTemplate(std::vector<std::string> template_ls, double err_rate);
    
    double CalcFoldEnergy(std::string foldseq);
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> 
        Results(std::unordered_map<std::string,double> SCopyMapIn);
    std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double> 
        ResultsSaveFoldMap(std::unordered_map<std::string,double> SCopyMapIn, std::string filename);
    
    void VerifyMatrixApproach();
    void getSEqMatrices();
    void getWEqMatrices();
    void getIndependentMatrixFractions(int SWalphabetsize, int ssize, int wsize, std::vector<double> probs, double epsilon = DEFAULT_
