#pragma once

#include <chrono>

template<bool enabled=true>
class ScopedTimer {
    const char *name;
    int index;
    std::chrono::time_point<std::chrono::steady_clock> start, end;
public:
    ScopedTimer(const char *name, int index);
    ScopedTimer(const char *name) : ScopedTimer(name, -1) {}
    ~ScopedTimer();
};

template<>
class ScopedTimer<false> {
public:
    ScopedTimer(const char *name, int index) {}
    ScopedTimer(const char *name) {}
};

template class ScopedTimer<true>;
