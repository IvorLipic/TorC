// examples/linear_regression.cpp
#include "torc/nn.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"
#include "torc/data.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using torc::Tensor;
using torc::Variable;
using torc::nn::Linear;
using torc::nn::MSELoss;
using torc::optim::SGD;
using torc::data::SyntheticRegression;
using torc::data::DataLoader;

int main() {
    const int num_samples = 100;
    const int num_features = 1;
    const float true_weight = 3.0f;
    const float true_bias = 1.0f;
    const float noise_std = 0.1f;
    const int epochs = 100;
    const float lr = 0.01f;
    const int batch_size = 10;

    std::cout << "Step 5.11 — End-to-end linear regression\n";
    std::cout << "Generating synthetic data: y = " << true_weight << " * x + " << true_bias
              << " + noise\n";

    SyntheticRegression dataset(num_samples, num_features, true_weight, true_bias, noise_std, 42);
    DataLoader loader(dataset, batch_size, true);

    Linear model(num_features, 1);
    auto params = model.parameters();
    SGD optimizer(params, lr);
    MSELoss loss_fn;

    std::ofstream loss_file("examples/linear_regression/loss_history.csv");
    loss_file << "epoch,loss\n";

    std::vector<float> all_x;
    std::vector<float> all_y_true;
    std::vector<float> all_y_pred;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        loader.reset();
        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        while (loader.has_next()) {
            auto [x_batch, y_batch] = loader.next_batch();

            Variable x(x_batch, true);
            Variable y(y_batch, false);

            Variable out = model(x);
            Variable loss = loss_fn(out, y);
            float batch_loss = loss.data().data()[0];
            epoch_loss += batch_loss;
            ++num_batches;

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        epoch_loss /= static_cast<float>(num_batches);
        loss_file << epoch << "," << epoch_loss << "\n";

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "epoch " << (epoch + 1) << "/" << epochs
                      << " — loss: " << epoch_loss << "\n";
        }
    }

    loss_file.close();

    std::cout << "\nLearned parameters:\n";
    const Tensor& W = model.named_parameters().at("weight").data();
    const Tensor& b = model.named_parameters().at("bias").data();
    std::cout << "  weight: " << W.data()[0] << " (true: " << true_weight << ")\n";
    std::cout << "  bias:   " << b.data()[0] << " (true: " << true_bias << ")\n";

    std::ofstream pred_file("examples/linear_regression/predictions.csv");
    pred_file << "x,y_true,y_pred\n";

    loader.reset();
    while (loader.has_next()) {
        auto [x_batch, y_batch] = loader.next_batch();
        Variable x(x_batch, false);
        Variable out = model(x);

        const float* x_ptr = x_batch.data();
        const float* y_true_ptr = y_batch.data();
        const float* y_pred_ptr = out.data().data();

        for (size_t i = 0; i < x_batch.shape().front(); ++i) {
            pred_file << x_ptr[i] << "," << y_true_ptr[i] << "," << y_pred_ptr[i] << "\n";
        }
    }
    pred_file.close();

    std::cout << "\nSaved examples/linear_regression/loss_history.csv and examples/linear_regression/predictions.csv\n";
    std::cout << "Run 'python examples/linear_regression/plot_results.py' to visualize.\n";

    return 0;
}
