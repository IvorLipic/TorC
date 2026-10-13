// Minimal torc consumer: y = sum(a * b), prints dy/da (== b) and dy/db (== a).
#include <iostream>
#include <torc/autograd.hpp>

int main() {
    using torc::Tensor;
    using torc::Variable;

    Variable a(Tensor({1.0f, 2.0f, 3.0f}, std::vector<int>{3}), true);
    Variable b(Tensor({4.0f, 5.0f, 6.0f}, std::vector<int>{3}), true);
    // Keep every tracked intermediate alive until backward() (the tape holds non-owning pointers).
    Variable prod = torc::mul(a, b);
    Variable y = torc::sum(prod);
    y.backward();

    std::cout << "y = " << y.data().data()[0] << "\n";
    std::cout << "dy/da =";
    for (int i = 0; i < a.grad().numel(); ++i) std::cout << ' ' << a.grad().data()[i];
    std::cout << "\ndy/db =";
    for (int i = 0; i < b.grad().numel(); ++i) std::cout << ' ' << b.grad().data()[i];
    std::cout << "\n";
    return 0;
}
