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
#include <omp.h>
#include "FoldModels.cpp"

int main() {
    Timer t1("Full operation");
    std::vector<std::tuple<double,double,double,double,double,double,double,double,double,double,double,double,double,double,double>> tbl;
    int l = 11;

    //std::unordered_map<std::string,double>  SCopyMap = IsingVar::GenerateBernoulliMap(0.1,l);
    std::vector<std::string> template_11 = {"01101110101"};
    /*template_11 = {"10010110101",
        "00111001011",
        "11001010001",
        "01011100110",
        "10100010111"};*/
    for (double lerr = -8; lerr < -0.51; lerr = lerr+0.5) {
        std::unordered_map<std::string,double>  SCopyMap = IsingVar::GenerateFromUniformTemplate(template_11, pow(2,lerr));
        for (double pwr = 0.0; pwr < 4.05; pwr = pwr+0.1) {
            double whom0 = exp(3*pwr);
            double c0 = exp(pwr);
            double whom1 = exp(-pwr);
            double c1 = exp(2*pwr);
            Ising2 ISInst(whom0, whom1, c0, c1, l);
            ISInst.GetEquilibrumTable();
            // ISInst.printAll();
            std::cout << "SW Equilibrium Validation: " << ISInst.EquilibriumValidation << std::endl;
            auto entry = ISInst.Results(SCopyMap);
            tbl.push_back(std::tuple_cat(std::make_tuple(lerr),std::make_tuple(pwr),entry));
        }
    }

    saveTuplesToCSV(tbl, "3m112_asymmetric_vary_log_err_eps_01101110101.tsv",
        {"log_error","eps","D(pmap(s,w)||peq(s,w))",
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
