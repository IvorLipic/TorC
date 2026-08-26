// examples/mnist_mlp/mnist_mlp.cpp
#include "torc/nn.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"
#include "torc/data.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using torc::Tensor;
using torc::Variable;
using torc::nn::Linear;
using torc::nn::ReLU;
using torc::nn::Sequential;
using torc::nn::CrossEntropyLoss;
using torc::optim::AdamW;
using torc::data::MNISTDataset;
using torc::data::DataLoader;

static std::pair<std::vector<int>, std::vector<int>> evaluate(const Sequential& model, const MNISTDataset& dataset, int batch_size) {
    DataLoader loader(dataset, batch_size, false);
    
    std::vector<int> correct(10, 0);
    std::vector<int> total(10, 0);
    
    while (loader.has_next()) {
        auto [x_batch, y_batch] = loader.next_batch();
        Variable x(x_batch, false);
        Variable out = model(x);
        
        for (size_t i = 0; i < x_batch.shape().front(); ++i) {
            float max_logit = -1e30f;
            int pred_class = 0;
            for (size_t j = 0; j < 10; ++j) {
                float val = out.data().data()[i * 10 + j];
                if (val > max_logit) {
                    max_logit = val;
                    pred_class = static_cast<int>(j);
                }
            }
            int true_class = static_cast<int>(y_batch.data()[i]);
            if (pred_class == true_class) {
                ++correct[true_class];
            }
            ++total[true_class];
        }
    }
    
    return {correct, total};
}

int main(int argc, char** argv) {
    const std::string train_path = "datasets/mnist/mnist_train.csv";
    const std::string test_path = "datasets/mnist/mnist_test.csv";
    
    size_t max_samples = 0;
    if (argc > 1) {
        max_samples = static_cast<size_t>(std::atoi(argv[1]));
    }
    
    const int batch_size = 64;
    const int epochs = 10;
    const float lr = 0.001f;

    std::cout << "Step 5.12 - End-to-end MLP on MNIST\n";
    std::cout << "Loading training dataset from: " << train_path;
    if (max_samples > 0) std::cout << " (max " << max_samples << " samples)";
    std::cout << "\n";

    MNISTDataset train_dataset(train_path, max_samples);
    DataLoader train_loader(train_dataset, batch_size, true);
    MNISTDataset test_dataset(test_path, max_samples);

    Sequential model;
    model.add(std::make_unique<Linear>(784, 32, 0.0f));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(32, 32, 0.0f));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(32, 10, 0.0f));

    auto params = model.parameters();
    AdamW optimizer(params, lr);
    CrossEntropyLoss loss_fn;

    std::ofstream loss_file("examples/mnist_mlp/loss_history.csv");
    loss_file << "epoch,loss\n";
    std::ofstream acc_file("examples/mnist_mlp/per_class_accuracy.csv");
    acc_file << "epoch";
    for (int c = 0; c < 10; ++c) acc_file << ",class_" << c;
    acc_file << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_loader.reset();
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        int step = 1;

        while (train_loader.has_next()) {
            auto [x_batch, y_batch] = train_loader.next_batch();

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

            std::cout << "Step " << step++ << ", (" << epoch_loss / static_cast<float>(num_batches) << ")" << "\n";
        }

        epoch_loss /= static_cast<float>(num_batches);
        loss_file << epoch << "," << epoch_loss << "\n";

        auto [train_correct, train_total] = evaluate(model, train_dataset, batch_size);
        auto [test_correct, test_total] = evaluate(model, test_dataset, batch_size);

        acc_file << epoch;
        for (int c = 0; c < 10; ++c) {
            float acc = test_total[c] > 0 ? (static_cast<float>(test_correct[c]) / static_cast<float>(test_total[c])) : 0.0f;
            acc_file << "," << acc;
        }
        acc_file << "\n";

        int train_total_correct = std::accumulate(train_correct.begin(), train_correct.end(), 0);
        int test_total_correct = std::accumulate(test_correct.begin(), test_correct.end(), 0);
        int train_total_samples = std::accumulate(train_total.begin(), train_total.end(), 0);
        int test_total_samples = std::accumulate(test_total.begin(), test_total.end(), 0);

        std::cout << "epoch " << (epoch + 1) << "/" << epochs
                  << " — loss: " << epoch_loss
                  << " — train acc: " << train_total_correct << "/" << train_total_samples
                  << " (" << (static_cast<float>(train_total_correct) / static_cast<float>(train_total_samples)) << ")"
                  << " — test acc: " << test_total_correct << "/" << test_total_samples
                  << " (" << (static_cast<float>(test_total_correct) / static_cast<float>(test_total_samples)) << ")"
                  << "\n";
    }

    loss_file.close();
    acc_file.close();

    std::cout << "\nSaved examples/mnist_mlp/loss_history.csv and per_class_accuracy.csv\n";
    std::cout << "Run 'python examples/mnist_mlp/plot_results.py' to visualize.\n";
    return 0;
}
