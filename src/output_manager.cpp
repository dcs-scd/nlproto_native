#include "output_manager.hpp"
#include <filesystem>

namespace fs = std::filesystem;

OutputManager::OutputManager(const std::string& dir) : out_dir(dir) {
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);
}

void OutputManager::open_result_file(const std::string& name) {
    std::lock_guard lg(mtx);
    if (out.is_open()) out.close();
    out.open(out_dir + "/" + name);
    if (!out) throw std::runtime_error("cannot open " + out_dir + "/" + name);
}

void OutputManager::write_result(const std::string& line) {
    std::lock_guard lg(mtx);
    out << line << '\n';
}

void OutputManager::flush() {
    std::lock_guard lg(mtx);
    out.flush();
}