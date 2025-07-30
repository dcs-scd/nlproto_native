#include "config.hpp"
#include "benchmark_types.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iostream>

std::vector<JobParams> build_matrix(const ExperimentConfig& cfg) {
    std::vector<JobParams> jobs;

    // Use the sweep field from ExperimentConfig
    const auto& sweep = cfg.sweep;
    std::cout << "build_matrix: sweep has " << sweep.size() << " parameters" << std::endl;
    if (sweep.empty()) return jobs;   // nothing to vary

    std::vector<std::vector<std::string>> dims;
    for (const auto& [name, values] : sweep) {
        if (!values.empty()) {
            dims.push_back(values);
        }
    }

    /* ---------- cartesian product same as before ---------- */
    uint64_t prod = 1;
    for (auto &d : dims) prod *= d.size();
    if (prod == 0) prod = 1;

    std::vector<uint64_t> strides(dims.size());
    uint64_t stride = 1;
    for (size_t i=dims.size();i--;) {
        strides[i] = stride;
        stride *= dims[i].size();
    }

    std::cout << "build_matrix: generating " << prod << " job combinations" << std::endl;
    for (uint64_t seq = 0; seq < prod; ++seq) {
        JobParams jp{static_cast<std::size_t>(seq + 1), "", cfg};

        std::ostringstream cmd;
        size_t off = 0;
        size_t param_count = 0;
        for (const auto& [name, values] : sweep) {
            const size_t idx = (seq / strides[off] ) % dims[off].size();
            cmd << "set " << name << " " << dims[off][idx];
            if (param_count < sweep.size() - 1) cmd << '\n';
            off++;
            param_count++;
        }
        jp.param_string = cmd.str();
        jobs.emplace_back(std::move(jp));
    }
    return jobs;
}