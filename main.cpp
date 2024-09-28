#include <iostream>
#include <vector>
#include <random>
#include "stock_hmm.cpp"  // or "stock_hmm.h" if you've separated the declaration and implementation

int main() {
    // Create a 4-state HMM
    StockHMM hmm(4);

    // Generate some random return data for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.001, 0.02);
    std::vector<double> returns(1000);
    for (auto& ret : returns) {
        ret = dis(gen);
    }

    // Train the model
    hmm.baum_welch(returns);

    // Make a prediction
    double next_return = hmm.predict_next_return();
    std::cout << "Predicted next return: " << next_return << std::endl;

    // Get trading signal
    std::string signal = hmm.get_trading_signal();
    std::cout << "Trading signal: " << signal << std::endl;

    // Calculate model selection criteria
    std::vector<int> discretized_returns = hmm.discretize_returns(returns);
    double log_likelihood = 0;
    for (const auto& alpha : hmm.forward(discretized_returns)) {
        log_likelihood += alpha;
    }

    double aic = hmm.calculate_aic(log_likelihood);
    double bic = hmm.calculate_bic(log_likelihood, returns.size());
    double hqc = hmm.calculate_hqc(log_likelihood, returns.size());
    double caic = hmm.calculate_caic(log_likelihood, returns.size());

    std::cout << "AIC: " << aic << ", BIC: " << bic << ", HQC: " << hqc << ", CAIC: " << caic << std::endl;

    return 0;
}