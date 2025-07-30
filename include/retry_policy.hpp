#pragma once
#include <chrono>
#include <thread>
#include <atomic>

template <std::size_t MaxAttempts>
class RetryPolicy {
public:
    using Clock = std::chrono::steady_clock;

    explicit RetryPolicy(std::chrono::milliseconds base = std::chrono::milliseconds(100))
        : base_(base) {}

    template <typename Fn>
    void invoke(Fn&& fn) {
        for (std::size_t i = 0; i < MaxAttempts; ++i) {
            try {
                fn();
                return;
            } catch (...) {
                if (i + 1 == MaxAttempts) throw;
                std::this_thread::sleep_for(delay(i));
            }
        }
    }

private:
    std::chrono::milliseconds delay(std::size_t attempt) const {
        return base_ * (1 << attempt);  // exponential backoff
    }
    std::chrono::milliseconds base_;
};