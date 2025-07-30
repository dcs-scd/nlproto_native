#pragma once
#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <map>

// Parameter specification types
struct ParameterConfiguration {
    enum Type { RANGE, LIST, LOGARITHMIC, CUSTOM };
    Type type = RANGE;
    
    // Range specification (min, max, step)
    double min = 0.0, max = 1.0, step = 1.0;
    
    // List specification (explicit values)
    std::vector<double> values;
    
    // Logarithmic specification (n values, base)
    int n = 5;
    double base = 10.0;
    
    // Generate parameter values based on specification
    std::vector<double> generate() const;
    
private:
    std::vector<double> generate_range() const;
    std::vector<double> generate_logarithmic() const;
    std::vector<double> generate_custom() const;
};

// Execution configuration section
struct ExecutionConfig {
    int cores = 4;
    int chunk_size = 10;
    std::string executor_backend = "original";
    int estimated_memory_per_run_mb = 384;
    double estimated_task_duration = 15.0;
};

// Benchmark configuration section  
struct BenchmarkConfig {
    int threads = 0;                    // Legacy field
    std::string jvm_args;               // Legacy field
    
    // Enhanced benchmark configuration
    std::string benchmark_mode = "use_config_only";
    bool force_rebenchmark = false;
    int max_age_hours = 24;
    std::string benchmark_backend = "unified_suite";
    std::vector<int> seeds_to_test;
    int min_seeds_per_combo = 3;
    int max_seeds_per_combo = 8;
};

// Output configuration section
struct OutputConfig {
    std::string base_directory = "results";
    bool save_individual_results = true;
    bool save_combined_results = true;
    bool save_parameter_summary = true;
    std::vector<std::string> formats = {"csv"};
    bool compress_results = false;
    std::string compression_format = "gzip";
};

// Logging configuration section
struct LoggingConfig {
    std::string level = "INFO";
    std::string directory = "results/logs";
    bool log_performance_metrics = true;
    std::string performance_metrics_file = "results/performance_metrics.json";
};

// Health check configuration
struct HealthCheckConfig {
    bool check_netlogo_installation = true;
    bool check_model_file = true;
    bool check_available_memory = true;
    bool check_disk_space = true;
    double min_available_memory_gb = 2.0;
    double min_available_disk_gb = 1.0;
};

// Advanced configuration section
struct AdvancedConfig {
    bool force_restart_simulation = false;
    bool enable_checkpointing = true;
    int checkpoint_interval_minutes = 30;
    bool continue_on_errors = true;
    double max_failed_tasks_percent = 10.0;
    int max_simulation_time_hours = 24;
    double memory_limit_simulation_gb = 16.0;
};

// Enhanced experiment configuration
struct ExperimentConfig {
    // Basic information
    std::string name;
    std::string description;
    
    // Model configuration
    std::string model_path;           // modelpath in YAML
    std::string nlpath;               // NetLogo installation path
    
    // Experiment parameters
    int ticks = 100;                  // experiment.runtime
    int base_seed = 42;               // experiment.base_seed
    int seeds_per_combo = 1;          // experiment.seeds_per_combo
    std::vector<std::string> metrics; // experiment.metrics
    
    // Enhanced parameter specifications
    std::map<std::string, ParameterConfiguration> variables;  // experiment.variables
    
    // Legacy sweep support (for backward compatibility)
    std::unordered_map<std::string, std::vector<std::string>> sweep;
    
    // Configuration sections
    ExecutionConfig execution;
    BenchmarkConfig benchmark;
    OutputConfig output;
    LoggingConfig logging;
    HealthCheckConfig health_check;
    AdvancedConfig advanced;
    
    // Legacy fields (for backward compatibility)
    int reps = 1;
    int threads = 0;                  // runtime override (0 ⇒ auto-detect)
    bool perform_health_check_on_startup = true;
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