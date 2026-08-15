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

    return 0;
}
