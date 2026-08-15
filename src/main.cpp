#include "torc/tensor.hpp"
#include <iostream>
using namespace torc;

int main() {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});

    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";

    std::cout << "a + b = " << a.add(b) << "\n";
    std::cout << "a - b = " << a.sub(b) << "\n";
    std::cout << "a * b = " << a.mul(b) << "\n";
    std::cout << "a / b = " << a.div(b) << "\n";
    std::cout << "a + 10 = " << a.add(10.0f) << "\n";
    std::cout << "a * 2  = " << a.mul(2.0f) << "\n";
    std::cout << "-a    = " << -a << "\n";

    Tensor m({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    std::cout << "\nm: " << m << "\n";
    std::cout << "m.sum() = " << m.sum() << "\n";
    std::cout << "m.mean() = " << m.mean() << "\n";
    std::cout << "m.max() = " << m.max() << "\n";
    std::cout << "m.min() = " << m.min() << "\n";
    std::cout << "m.sum(0) = " << m.sum(0) << "\n";
    std::cout << "m.sum(1) = " << m.sum(1) << "\n";
    std::cout << "m.mean(1) = " << m.mean(1) << "\n";
    std::cout << "m.max(1) = " << m.max(1) << "\n";
    std::cout << "m.min(0) = " << m.min(0) << "\n";

    return 0;
}
