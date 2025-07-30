#pragma once
#include <string>
#include <fstream>
#include <mutex>

class OutputManager {
public:
    OutputManager(const std::string& output_dir);
    void open_result_file(const std::string& name);
    void write_result(const std::string& line);
    void flush();
private:
    std::string out_dir;
    std::ofstream out;           // single file — we rotate inside `main`
    std::mutex mtx;
};