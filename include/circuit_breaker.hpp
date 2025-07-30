#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

class CircuitBreaker {
public:
    explicit CircuitBreaker(std::size_t max_failures,
                            std::chrono::milliseconds timeout = std::chrono::seconds(5));

    bool is_open() const;
    void on_success();
    void on_failure();
    template <typename Fn> bool call(Fn&& fn);

private:
    void reset();               // reset to closed
    void trip();                // open -> half-open after timeout

    mutable std::mutex mtx_;
    std::size_t max_failures_;
    std::chrono::milliseconds timeout_;
    std::size_t failures_ = 0;
    std::chrono::steady_clock::time_point last_fail_{};
    enum State { CLOSED, OPEN, HALF_OPEN } state_ = CLOSED;
};

/* Inline definitions */
inline CircuitBreaker::CircuitBreaker(std::size_t max_failures,
                                      std::chrono::milliseconds timeout)
    : max_failures_(max_failures), timeout_(timeout) {}

inline bool CircuitBreaker::is_open() const {
    std::lock_guard lg(mtx_);
    if (state_ == OPEN && (std::chrono::steady_clock::now() - last_fail_) > timeout_) {
        const_cast<CircuitBreaker*>(this)->state_ = HALF_OPEN;
    }
    return state_ == OPEN;
}

inline void CircuitBreaker::on_success() {
    std::lock_guard lg(mtx_);
    failures_ = 0;
    state_ = CLOSED;
}

inline void CircuitBreaker::on_failure() {
    std::lock_guard lg(mtx_);
    failures_++;
    last_fail_ = std::chrono::steady_clock::now();
    if (failures_ >= max_failures_)
        state_ = OPEN;
}

inline void CircuitBreaker::trip()  { /* noop – half-open moves us forward */ }
inline void CircuitBreaker::reset() { on_success(); }

template <typename Fn>
inline bool CircuitBreaker::call(Fn&& fn) try {
    if (is_open()) return false;
    try { fn(); } catch (...) { on_failure(); throw; }
    on_success();
    return true;
} catch (...) { return false; }