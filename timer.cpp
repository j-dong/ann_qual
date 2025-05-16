#include "timer.h"

#include <iostream>
#include <iomanip>

template<>
ScopedTimer<true>::ScopedTimer(const char *name, int index) : name(name), index(index) {
    start = std::chrono::steady_clock::now();
}

template<>
void ScopedTimer<true>::print_timer_message(const char *message, int index, double ms) {
    std::cout << ">> " << message;
    if (index >= 0) std::cout << " [" << index << "]";
    std::cout << ": " << std::fixed << std::setprecision(3)
        << ms << " ms" << std::endl;
}

template<>
ScopedTimer<true>::~ScopedTimer() {
    end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>
    >(end - start);
    print_timer_message(name, index, duration.count());
}

template<>
double ScopedTimer<true>::get_ms() {
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>
    >(end - start);
    return duration.count();
}
