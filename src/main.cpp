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

    Tensor n({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    std::cout << "\nn: " << n << "\n";
    std::cout << "n.reshape({4}) = " << n.reshape({4}) << "\n";
    std::cout << "n.view({1, 4}) = " << n.view({1, 4}) << "\n";

    Tensor p({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    std::cout << "\np: " << p << "\n";
    std::cout << "p(0, 0) = " << p(0, 0) << "\n";
    p(1, 1) = 99.0f;
    std::cout << "p(1, 1) = " << p(1, 1) << "\n";

    Tensor q({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    std::cout << "\nq: " << q << "\n";
    std::cout << "q.transpose({}) = " << q.transpose({}) << "\n";
    std::cout << "q.slice({0,2}, {0,2}) = " << q.slice({Tensor::Slice{0, 2}, Tensor::Slice{0, 2}}) << "\n";

    Tensor x({1.0f, 2.0f, 3.0f}, {3});
    Tensor y({10.0f, 20.0f, 30.0f}, {3, 1});
    std::cout << "\nx: " << x << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "x + y = " << x.add(y) << "\n";

    return 0;
}
