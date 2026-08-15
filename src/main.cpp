#include "torc/tensor.hpp"
#include <iostream>
using namespace torc;

int main() {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});

    Tensor c = a.add(b);
    Tensor d = a.mul(b);

    std::cout << "c: ";
    for (int i = 0; i < c.numel(); ++i) std::cout << c.data()[i] << " ";
    std::cout << "\n";

    std::cout << "d: ";
    for (int i = 0; i < d.numel(); ++i) std::cout << d.data()[i] << " ";
    std::cout << "\n";

    return 0;
}
