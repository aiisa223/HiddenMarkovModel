#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>

namespace py = pybind11;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


class StockHMM {
private:
    int num_states;
    std::vector<double> initial_probs;
    std::vector<std::vector<double>> transition_probs;
    std::vector<double> mean_returns;
    std::vector<double> std_returns;

    // Helper function for log-sum-exp trick
    double logsumexp(const std::vector<double>& vec) const {
        double max_val = *std::max_element(vec.begin(), vec.end());
        double sum = 0.0;
        for (double x : vec) {
            sum += std::exp(x - max_val);
        }
        return max_val + std::log(sum);
    }

public:
    StockHMM(int states = 4) : num_states(states) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Initialize probabilities and parameters
        initial_probs.resize(num_states);
        transition_probs.resize(num_states, std::vector<double>(num_states));
        mean_returns.resize(num_states);
        std_returns.resize(num_states);

        // Random initialization
        for (int i = 0; i < num_states; ++i) {
            initial_probs[i] = dis(gen);
            mean_returns[i] = dis(gen) * 0.02 - 0.01;  // Random mean between -1% and 1%
            std_returns[i] = dis(gen) * 0.02;  // Random std between 0% and 2%
            for (int j = 0; j < num_states; ++j) {
                transition_probs[i][j] = dis(gen);
            }
        }

        // Normalize probabilities
        double sum_initial = std::accumulate(initial_probs.begin(), initial_probs.end(), 0.0);
        for (int i = 0; i < num_states; ++i) {
            initial_probs[i] /= sum_initial;
            double sum_trans = std::accumulate(transition_probs[i].begin(), transition_probs[i].end(), 0.0);
            for (int j = 0; j < num_states; ++j) {
                transition_probs[i][j] /= sum_trans;
            }
        }
    }

    void baum_welch(const std::vector<double>& returns, int max_iterations = 100, double tolerance = 1e-6) {
        int T = returns.size();
        double prev_log_likelihood = -std::numeric_limits<double>::infinity();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            // Forward-Backward algorithm
            auto [alpha, beta, log_likelihood] = forward_backward(returns);

            // Compute gamma and xi
            std::vector<std::vector<double>> gamma(T, std::vector<double>(num_states));
            std::vector<std::vector<std::vector<double>>> xi(T-1, std::vector<std::vector<double>>(num_states, std::vector<double>(num_states)));

            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < num_states; ++i) {
                    gamma[t][i] = std::exp(alpha[t][i] + beta[t][i] - log_likelihood);
                    if (t < T - 1) {
                        for (int j = 0; j < num_states; ++j) {
                            xi[t][i][j] = std::exp(alpha[t][i] + std::log(transition_probs[i][j]) +
                                                   std::log(std::exp(-0.5 * std::pow((returns[t+1] - mean_returns[j]) / std_returns[j], 2)) /
                                                            (std_returns[j] * std::sqrt(2 * M_PI))) +
                                                   beta[t+1][j] - log_likelihood);
                        }
                    }
                }
            }

            // Update parameters
            for (int i = 0; i < num_states; ++i) {
                initial_probs[i] = gamma[0][i];

                double sum_gamma = 0;
                double sum_returns = 0;
                double sum_squared_returns = 0;
                for (int t = 0; t < T - 1; ++t) {
                    sum_gamma += gamma[t][i];
                    sum_returns += gamma[t][i] * returns[t];
                    sum_squared_returns += gamma[t][i] * returns[t] * returns[t];
                }

                for (int j = 0; j < num_states; ++j) {
                    double sum_xi = 0;
                    for (int t = 0; t < T - 1; ++t) {
                        sum_xi += xi[t][i][j];
                    }
                    transition_probs[i][j] = sum_xi / sum_gamma;
                }

                mean_returns[i] = sum_returns / sum_gamma;
                std_returns[i] = std::sqrt(sum_squared_returns / sum_gamma - mean_returns[i] * mean_returns[i]);
            }

            // Check for convergence
            if (std::abs(log_likelihood - prev_log_likelihood) < tolerance) {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, double>
    forward_backward(const std::vector<double>& observations) const {
        int T = observations.size();
        std::vector<std::vector<double>> alpha(T, std::vector<double>(num_states));
        std::vector<std::vector<double>> beta(T, std::vector<double>(num_states, 0.0));

        // Forward pass
        for (int i = 0; i < num_states; ++i) {
            alpha[0][i] = std::log(initial_probs[i]) +
                          std::log(std::exp(-0.5 * std::pow((observations[0] - mean_returns[i]) / std_returns[i], 2)) /
                                   (std_returns[i] * std::sqrt(2 * M_PI)));
        }
        for (int t = 1; t < T; ++t) {
            for (int j = 0; j < num_states; ++j) {
                std::vector<double> temp(num_states);
                for (int i = 0; i < num_states; ++i) {
                    temp[i] = alpha[t-1][i] + std::log(transition_probs[i][j]);
                }
                alpha[t][j] = logsumexp(temp) +
                              std::log(std::exp(-0.5 * std::pow((observations[t] - mean_returns[j]) / std_returns[j], 2)) /
                                       (std_returns[j] * std::sqrt(2 * M_PI)));
            }
        }

        // Backward pass
        for (int t = T - 2; t >= 0; --t) {
            for (int i = 0; i < num_states; ++i) {
                std::vector<double> temp(num_states);
                for (int j = 0; j < num_states; ++j) {
                    temp[j] = std::log(transition_probs[i][j]) +
                              std::log(std::exp(-0.5 * std::pow((observations[t+1] - mean_returns[j]) / std_returns[j], 2)) /
                                       (std_returns[j] * std::sqrt(2 * M_PI))) +
                              beta[t+1][j];
                }
                beta[t][i] = logsumexp(temp);
            }
        }

        double log_likelihood = logsumexp(alpha.back());
        return {alpha, beta, log_likelihood};
    }

    double predict_next_return() const {
        double predicted_return = 0.0;
        for (int i = 0; i < num_states; ++i) {
            predicted_return += initial_probs[i] * mean_returns[i];
        }
        return predicted_return;
    }

    double calculate_aic(double log_likelihood) const {
        int num_params = num_states * (num_states - 1) + num_states * 2;
        return 2 * num_params - 2 * log_likelihood;
    }

    double calculate_bic(double log_likelihood, int num_observations) const {
        int num_params = num_states * (num_states - 1) + num_states * 2;
        return num_params * std::log(num_observations) - 2 * log_likelihood;
    }

    double calculate_hqc(double log_likelihood, int num_observations) const {
        int num_params = num_states * (num_states - 1) + num_states * 2;
        return -2 * log_likelihood + 2 * num_params * std::log(std::log(num_observations));
    }

    double calculate_caic(double log_likelihood, int num_observations) const {
        int num_params = num_states * (num_states - 1) + num_states * 2;
        return -2 * log_likelihood + num_params * (std::log(num_observations) + 1);
    }

    double calculate_out_of_sample_r_squared(const std::vector<double>& true_returns, const std::vector<double>& predicted_returns) const {
        double mean_return = std::accumulate(true_returns.begin(), true_returns.end(), 0.0) / true_returns.size();
        double tss = 0.0, rss = 0.0;
        for (size_t i = 0; i < true_returns.size(); ++i) {
            tss += std::pow(true_returns[i] - mean_return, 2);
            rss += std::pow(true_returns[i] - predicted_returns[i], 2);
        }
        return 1.0 - (rss / tss);
    }

    std::string get_trading_signal() const {
        double predicted_return = predict_next_return();
        if (predicted_return > 0.005) {  // 0.5% threshold for buying
            return "BUY";
        } else if (predicted_return < -0.005) {  // -0.5% threshold for selling
            return "SELL";
        } else {
            return "HOLD";
        }
    }
};

PYBIND11_MODULE(stock_hmm, m) {
    py::class_<StockHMM>(m, "StockHMM")
            .def(py::init<int>())
            .def("baum_welch", &StockHMM::baum_welch)
            .def("predict_next_return", &StockHMM::predict_next_return)
            .def("calculate_aic", &StockHMM::calculate_aic)
            .def("calculate_bic", &StockHMM::calculate_bic)
            .def("calculate_hqc", &StockHMM::calculate_hqc)
            .def("calculate_caic", &StockHMM::calculate_caic)
            .def("calculate_out_of_sample_r_squared", &StockHMM::calculate_out_of_sample_r_squared)
            .def("get_trading_signal", &StockHMM::get_trading_signal);
}

