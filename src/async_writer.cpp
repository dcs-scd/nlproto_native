// src/async_writer.cpp
#include "async_writer.hpp"
#include <stdexcept>

AsyncWriter::AsyncWriter(std::ostream& out, std::size_t cap)
    : out_(out), cap_(cap), buf_(cap) { }

AsyncWriter::~AsyncWriter() { stop(); }

void AsyncWriter::start() {
    std::lock_guard lg(mtx_);
    if (worker_.joinable()) return;
    active_ = true;
    worker_ = std::thread(&AsyncWriter::run, this);
}

void AsyncWriter::emit(const std::string& line) {
    std::unique_lock lg(mtx_);
    while ((tail_ + 1) % cap_ == head_)
        cv_.wait(lg);   // wait for space
    buf_[head_] = line;
    head_ = (head_ + 1) % cap_;
    cv_.notify_one();
}

void AsyncWriter::run() {
    while (!stop_.load()) {
        std::unique_lock lg(mtx_);
        while (head_ == tail_ && !stop_.load()) {}
        if (tail_ != head_) {
            out_ << buf_[tail_] << '\n';
            tail_ = (tail_ + 1) % cap_;
            cv_.notify_all();
        }
    }
    out_.flush();
}

void AsyncWriter::stop() {
    if (stop_.exchange(true)) return;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}