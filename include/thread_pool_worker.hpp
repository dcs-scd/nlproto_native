#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <atomic>
#include <condition_variable>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    void submit(std::function<void()> task);
    void wait();
    void drain() {               // waits until queue empty
        std::unique_lock lg(mtx_);
        cv_.wait(lg, [this]{ return tasks_.empty() && tasks_running_ == 0; });
    }

private:
    void worker();
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<std::size_t> tasks_running_{0};
};