// src/thread_pool_worker.cpp
#include "thread_pool_worker.hpp"
#include <stdexcept>

ThreadPool::ThreadPool(size_t num)
    : shutdown_{ false } {
    for (size_t i = 0; i < num; ++i)
        workers_.emplace_back(&ThreadPool::worker, this);
}

ThreadPool::~ThreadPool() {
    shutdown_ = true;
    cv_.notify_all();
    for (auto& w : workers_) if (w.joinable()) w.join();
}

/* public methods */
void ThreadPool::submit(std::function<void()> f) {
    std::lock_guard lg(mtx_);
    if (shutdown_) throw std::runtime_error("ThreadPool stopped");
    tasks_.push(std::move(f));
    cv_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock lg(mtx_);
    cv_.wait(lg, [&]{ return tasks_.empty() && tasks_running_.load() == 0; });
}

void ThreadPool::worker() {
    while (!shutdown_.load()) {
        std::function<void()> task;
        {
            std::unique_lock lg(mtx_);
            cv_.wait(lg, [&]{ return !tasks_.empty() || shutdown_.load(); });
            if (shutdown_.load()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        tasks_running_++;
        try {
            task();
        } catch (...) {
            // Ignore exceptions from tasks
        }
        tasks_running_--;
        cv_.notify_all();
    }
}