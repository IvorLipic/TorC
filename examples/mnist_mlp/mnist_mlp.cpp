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

using torc::Tensor;
using torc::Variable;
using torc::nn::Linear;
using torc::nn::ReLU;
using torc::nn::Sequential;
using torc::nn::CrossEntropyLoss;
using torc::optim::Adam;
using torc::data::MNISTDataset;
using torc::data::DataLoader;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mnist_train.csv>\n";
        std::cerr << "Download from: https://www.kaggle.com/c/digit-recognizer/data\n";
        return 1;
    }

    const std::string csv_path = argv[1];
    const int batch_size = 64;
    const int epochs = 3;
    const float lr = 0.001f;

    std::cout << "Step 5.12 — End-to-end MLP on MNIST\n";
    std::cout << "Loading dataset from: " << csv_path << "\n";

    MNISTDataset dataset(csv_path);
    DataLoader loader(dataset, batch_size, true);

    Sequential model;
    model.add(std::make_unique<Linear>(784, 128));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(128, 10));

    auto params = model.parameters();
    Adam optimizer(params, lr);
    CrossEntropyLoss loss_fn;

    std::ofstream loss_file("examples/mnist_mlp/loss_history.csv");
    loss_file << "epoch,loss\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        loader.reset();
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        size_t correct = 0;
        size_t total = 0;

        while (loader.has_next()) {
            auto [x_batch, y_batch] = loader.next_batch();

            Variable x(x_batch, true);
            Variable y(y_batch, false);

            Variable out = model(x);
            Variable loss = loss_fn(out, y);
            float batch_loss = loss.data().data()[0];
            epoch_loss += batch_loss;
            ++num_batches;

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
                if (pred_class == static_cast<int>(y_batch.data()[i])) {
                    ++correct;
                }
                ++total;
            }

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        epoch_loss /= static_cast<float>(num_batches);
        float accuracy = static_cast<float>(correct) / static_cast<float>(total);
        loss_file << epoch << "," << epoch_loss << "\n";

        std::cout << "epoch " << (epoch + 1) << "/" << epochs
                  << " — loss: " << epoch_loss
                  << " — acc: " << accuracy << "\n";
    }

    loss_file.close();

    std::cout << "\nSaved examples/mnist_mlp/loss_history.csv\n";
    return 0;
}
