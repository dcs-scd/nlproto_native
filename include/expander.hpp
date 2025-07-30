#pragma once
// expander.hpp
// MIT 2024 – transforms the *concise* YAML sweep into every concrete
// {pa,pb,pc…} job that the benchmark is required to run.

#include "config.hpp"
#include <vector>

struct Job {
    int           id;
    std::size_t   rep_index;
    std::vector<double> params; // idx 0 == param[0].name …
};

// JobParams defined in benchmark_types.hpp

std::vector<Job> expand_experiment(const Experiment& exp);