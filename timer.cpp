#include "timer.h"

#include <iostream>
#include <iomanip>

template<>
ScopedTimer<true>::ScopedTimer(const char *name, int index) : name(name), index(index) {
    start = std::chrono::steady_clock::now();
}

template<>
ScopedTimer<true>::~ScopedTimer() {
    end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>
    >(end - start);
    std::cout << ">> " << name;
    if (index >= 0) std::cout << " [" << index << "]";
    std::cout << ": " << std::fixed << std::setprecision(3)
        << duration.count() << " ms" << std::endl;
}
