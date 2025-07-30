#pragma once
#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>

struct BenchmarkConfig {
    int threads   = 0;                // 0  ⇒ auto-detect
    std::string jvm_args;
};

struct ExperimentConfig {
    std::string name;
    int         ticks   = 1000;
    int         reps    = 1;
    int         threads = 0;          // runtime override (0 ⇒ auto-detect)
    std::string model_path;
    std::string nlpath;               // NetLogo installation path
    std::unordered_map<std::string, std::vector<std::string>> sweep;
};

struct JobParams {
    std::size_t job_id = 0;
    std::string param_string;  // Added missing member
    ExperimentConfig base;
};

BenchmarkConfig auto_benchmark(const ExperimentConfig&);   // impl in config.cpp

// Function declarations
ExperimentConfig load_experiment(const std::string& file);
std::vector<JobParams> build_matrix(const ExperimentConfig& cfg);
double run_netlogo_sim(const ExperimentConfig& cfg, const std::string& paramString, int seed);