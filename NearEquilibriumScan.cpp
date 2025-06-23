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
#include "FoldModels.cpp"
#include <omp.h>

int main() {
    Timer t1("Full operation");
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> tbl;
    int l = 11;

    //std::unordered_map<std::string,double>  
    unsigned int seed = 0;
    std::mt19937_64 gen(seed);
    double pwr = 2.0;
    for (double tresh = 0.0; tresh < 0.051; tresh += 0.01){
        for (int trials = 0; trials < 10; trials++) {
                double whom1 = exp(pwr);
                double whom2 = exp(pwr);
                double ahet1 = exp(pwr);
                double ahet2 = exp(pwr);
                Ising2 ISInst(whom1, whom2, 1, 1, 1, l);
                //Ising2S3F ISInst(whom1, whom2, ahet1, ahet2, 1, l);
                ISInst.GetEquilibrumTable();
                // ISInst.printAll();
                std::cout << "SW Equilibrium Validation: " << ISInst.EquilibriumValidation << std::endl;
                //std::cout << SCopyMap["11111111111"] << std::endl;
                std::unordered_map<std::string,double> SCopyMap = ISInst.SampleCloseSequenceDist(0.05, gen);
                auto entry = ISInst.Results(SCopyMap);
                tbl.push_back(std::tuple_cat(std::make_tuple(tresh),std::make_tuple(trials),entry));
        }
    }
    
    saveTuplesToCSV(tbl, "near_equilibrium_symmetric_2_l_11.tsv",
        {"threshold","trial","D(pmap(s,w)||peq(s,w))",
        "D(pmap(s)||peq(s))",
        "D(pmap(w)||peq(w))",
        "NHelixeq",
        "NHelixMap",
        "<U>eq",
        "<U>map",
        "Heq(s|w)",
        "Hmap(s|w)",
        "Heq(w)",
        "Hmap(w)",
        "Heq(s)",
        "Hmap(s)"});
    
    return 0;
}
/*int main() {
     // Initialize Ising model
    std::random_device rd;
    std::mt19937 gen(11125689);

    Ising2 ISInst(2.0, 2.0, 0.1, 0.1, 0.1, 100);
    
    ISInst.CalcAllPartition();
    // Calculate and print results
    double eigenG = ISInst.eigenG();
    double gNum = ISInst.Gnum();
    
    std::cout << "EigenG SW: " << eigenG << std::endl;
    std::cout << "Gnum SW: " << gNum << std::endl;
    
    // Calculate transfer probabilities
    //ISInst.eqTransferProb();
    //for_each(ISInst.eqCondProb.begin(), ISInst.eqCondProb.end(), [](Eigen::MatrixXd i) {
    //      std::cout << i << std::endl;
    //});

    ISInst.SampleSequence(10,0.3,gen);
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 7; j++){
            std::cout<< ISInst.SequenceList[i][j] <<" ";
        }
        std::cout<<std::endl;
    }
}*/

// Functions which are not yet useful

/*std::vector<Eigen::MatrixXd> eqTransferProb(Eigen::RowVectorXd start, Eigen::VectorXd end, Eigen::MatrixXd weightMatrix) {
        std::vector<Eigen::VectorXd> backwardrun(L);
        std::vector<Eigen::MatrixXd> eqCondProb;
        eqCondProb.resize(L);

        backwardrun[L-1] = end;

        for (int i = 1; i < L; i++) {
            backwardrun[L-i-1] = weightMatrix * backwardrun[L-i];
        }
        
        for (int i = 0; i < L; i++) {
            Eigen::MatrixXd VMprod = weightMatrix;
            for (int j = 0; j < VMprod.rows(); j++) {
                std::cout << weightMatrix.row(j) << std::endl;
                VMprod.row(j) = weightMatrix.row(j).cwiseProduct(backwardrun[i].transpose());
            }
            
            Eigen::VectorXd rowSums = VMprod.rowwise().sum();
            for (int j = 0; j < VMprod.rows(); j++) {
                VMprod.row(j) /= rowSums(j);
            }
            
            eqCondProb[i] = VMprod;
        }
        
        eqCondProb[0] = start * eqCondProb[0];
        
        return eqCondProb;
    }*/

/*
void SampleSequence(int num, double prob,std::mt19937 gen){
        std::bernoulli_distribution bernoulli_dist(prob);
        std::vector<std::vector<int>> vecs(num,std::vector<int>(L));
        for (int n = 0; n < num; n++)
            for (int l = 0; l < L; l++)
                vecs[n][l] = int(bernoulli_dist(gen)) + 1;
        SequenceList = vecs;
    }
*/