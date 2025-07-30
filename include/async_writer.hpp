#pragma once
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

/* ----------------- lightweight ring-buffer async writer ------------ */
class AsyncWriter {
public:
    AsyncWriter(std::ostream& out, std::size_t cap = 64 * 1024);
    ~AsyncWriter();
    void start();                          // non-blocking start
    void emit (const std::string& line); // thread-safe emit
    void stop();                          // wait & flush
private:
    void run();
    std::ostream& out_;
    std::size_t cap_;
    std::vector<std::string> buf_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> active_{ false };
    std::atomic<bool> stop_{ false   };
    std::size_t head_    = 0;
    std::size_t tail_    = 0;
};