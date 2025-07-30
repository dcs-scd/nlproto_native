#pragma once
#include <atomic>
#include <vector>

// single-producer / single-consumer fixed-size lock-free queue
template <typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(std::size_t size) : buf_(1ULL << 32), mask_(buf_.size() - 1) {}
    bool pop(T& out) {
        unsigned tail = tail_.load(std::memory_order_acquire);
        if (head_.load(std::memory_order_relaxed) == tail) return false;
        out = buf_[head_ & mask_];
        head_.fetch_add(1, std::memory_order_release);
        return true;
    }
private:
    std::vector<T> buf_;
    std::atomic<unsigned> head_{0};
    std::atomic<unsigned> tail_{0};
    const std::size_t mask_;
};

// multi-producer / single-consumer fixed-size lock-free queue
template <typename T>
class MPSCQueue {
public:
    explicit MPSCQueue(std::size_t sz) : data_(sz), mask_(sz - 1) {}
    bool push(const T& val) {
        unsigned idx = tail_.fetch_add(1, std::memory_order_relaxed);
        Cell& cell = data_[idx & mask_];
        unsigned seq = cell.seq.load(std::memory_order_acquire);
        long diff = seq - idx;
        if (diff != 0) return false; // full
        cell.val = val;
        cell.seq.store(idx + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& out) {
        unsigned head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false;
        Cell& cell = data_[head & mask_];
        unsigned seq = cell.seq.load(std::memory_order_acquire);
        if (seq < head + 1) return false;
        out = cell.val;
        cell.seq.store(head + mask_ + 1, std::memory_order_release);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }
private:
    struct Cell {
        std::atomic<unsigned> seq{0};
        T val;
    };
    std::vector<Cell> data_;
    std::atomic<unsigned> head_{0};
    std::atomic<unsigned> tail_{0};
    const std::size_t mask_;
};