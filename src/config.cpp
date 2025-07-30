// src/config.cpp
#include "config.hpp"
#include "benchmark_types.hpp"
// #include "experiment_config.hpp"  // Not needed - ExperimentConfig is in benchmark_types.hpp
#include <yaml-cpp/yaml.h>
#include <thread>
#include <string>
#include <iostream>
#include <cmath>

BenchmarkConfig auto_benchmark(const ExperimentConfig& cfg) {
    BenchmarkConfig out;
    
    // Build classpath from nlpath if available
    std::string classpath = "-Xmx2g";
    if (!cfg.nlpath.empty()) {
        std::string nl_app = cfg.nlpath + "/app";
        classpath += " -Djava.library.path=\"" + nl_app + "\"";
        classpath += " -Djava.class.path=\"target/classes";
        classpath += ":" + nl_app + "/netlogo-6.4.0.jar";
        classpath += ":" + nl_app + "/scala-library.jar";
        classpath += ":" + nl_app + "/behaviorsearch.jar";
        classpath += ":" + nl_app + "/extensions";
        classpath += ":" + nl_app + "/*\"";
    }
    
    out.jvm_args = classpath;
    out.threads  = (cfg.threads == 0)
               ? unsigned(std::thread::hardware_concurrency())
               : cfg.threads;
    return out;
}

/* YAML loader  */
ExperimentConfig load_experiment(const std::string& f) {
    YAML::Node n = YAML::LoadFile(f);
    ExperimentConfig e;
    
    // Basic information
    e.name = n["name"].as<std::string>("run");
    e.description = n["description"].as<std::string>("");
    
    // Model configuration
    if (n["model"]) {
        e.model_path = n["model"].as<std::string>();
    } else if (n["modelpath"]) {
        e.model_path = n["modelpath"].as<std::string>();
    }
    if (n["nlpath"]) {
        e.nlpath = n["nlpath"].as<std::string>();
    }
    
    // Legacy fields
    e.ticks = n["ticks"].as<int>(1000);
    e.reps = n["reps"].as<int>(1);
    e.threads = n["parallel"].as<int>(0);
    
    // Experiment section
    if (n["experiment"]) {
        const auto& exp = n["experiment"];
        e.ticks = exp["runtime"].as<int>(e.ticks);
        e.base_seed = exp["base_seed"].as<int>(42);
        e.seeds_per_combo = exp["seeds_per_combo"].as<int>(1);
        
        if (exp["metrics"]) {
            for (const auto& metric : exp["metrics"]) {
                if (metric.IsScalar()) {
                    // Simple string format
                    e.metrics.push_back(metric.as<std::string>());
                } else if (metric["name"]) {
                    // Object format with name field
                    e.metrics.push_back(metric["name"].as<std::string>());
                }
            }
        }
    }
    
    // Top-level metrics (for advanced YAML configs)
    if (n["metrics"] && e.metrics.empty()) {
        for (const auto& metric : n["metrics"]) {
            if (metric.IsScalar()) {
                // Simple string format
                e.metrics.push_back(metric.as<std::string>());
            } else if (metric["name"]) {
                // Object format with name field
                e.metrics.push_back(metric["name"].as<std::string>());
            }
        }
    }
    
    // Execution configuration
    if (n["execution"]) {
        const auto& exec = n["execution"];
        e.execution.cores = exec["cores"].as<int>(4);
        e.execution.chunk_size = exec["chunk_size"].as<int>(10);
        e.execution.executor_backend = exec["executor_backend"].as<std::string>("original");
        e.execution.estimated_memory_per_run_mb = exec["estimated_memory_per_run_mb"].as<int>(384);
        e.execution.estimated_task_duration = exec["estimated_task_duration"].as<double>(15.0);
    }
    
    // Benchmark configuration
    if (n["benchmark"]) {
        const auto& bench = n["benchmark"];
        e.benchmark.benchmark_mode = bench["benchmark_mode"].as<std::string>("use_config_only");
        e.benchmark.force_rebenchmark = bench["force_rebenchmark"].as<bool>(false);
        e.benchmark.max_age_hours = bench["max_age_hours"].as<int>(24);
        e.benchmark.benchmark_backend = bench["benchmark_backend"].as<std::string>("unified_suite");
        e.benchmark.min_seeds_per_combo = bench["min_seeds_per_combo"].as<int>(3);
        e.benchmark.max_seeds_per_combo = bench["max_seeds_per_combo"].as<int>(8);
        
        if (bench["seeds_to_test"]) {
            for (const auto& seed : bench["seeds_to_test"]) {
                e.benchmark.seeds_to_test.push_back(seed.as<int>());
            }
        }
    }
    
    // Output configuration
    if (n["output"]) {
        const auto& output = n["output"];
        e.output.base_directory = output["base_directory"].as<std::string>("results");
        e.output.save_individual_results = output["save_individual_results"].as<bool>(true);
        e.output.save_combined_results = output["save_combined_results"].as<bool>(true);
        e.output.save_parameter_summary = output["save_parameter_summary"].as<bool>(true);
        e.output.compress_results = output["compress_results"].as<bool>(false);
        e.output.compression_format = output["compression_format"].as<std::string>("gzip");
        
        if (output["formats"]) {
            e.output.formats.clear();
            for (const auto& format : output["formats"]) {
                e.output.formats.push_back(format.as<std::string>());
            }
        }
    }
    
    // Logging configuration
    if (n["logging"]) {
        const auto& logging = n["logging"];
        e.logging.level = logging["level"].as<std::string>("INFO");
        e.logging.directory = logging["directory"].as<std::string>("results/logs");
        e.logging.log_performance_metrics = logging["log_performance_metrics"].as<bool>(true);
        e.logging.performance_metrics_file = logging["performance_metrics_file"].as<std::string>("results/performance_metrics.json");
    }
    
    // Health check configuration
    if (n["health_check"]) {
        const auto& health = n["health_check"];
        e.health_check.check_netlogo_installation = health["check_netlogo_installation"].as<bool>(true);
        e.health_check.check_model_file = health["check_model_file"].as<bool>(true);
        e.health_check.check_available_memory = health["check_available_memory"].as<bool>(true);
        e.health_check.check_disk_space = health["check_disk_space"].as<bool>(true);
        e.health_check.min_available_memory_gb = health["min_available_memory_gb"].as<double>(2.0);
        e.health_check.min_available_disk_gb = health["min_available_disk_gb"].as<double>(1.0);
    }
    
    // Advanced configuration
    if (n["advanced"]) {
        const auto& advanced = n["advanced"];
        e.advanced.force_restart_simulation = advanced["force_restart_simulation"].as<bool>(false);
        e.advanced.enable_checkpointing = advanced["enable_checkpointing"].as<bool>(true);
        e.advanced.checkpoint_interval_minutes = advanced["checkpoint_interval_minutes"].as<int>(30);
        e.advanced.continue_on_errors = advanced["continue_on_errors"].as<bool>(true);
        e.advanced.max_failed_tasks_percent = advanced["max_failed_tasks_percent"].as<double>(10.0);
        e.advanced.max_simulation_time_hours = advanced["max_simulation_time_hours"].as<int>(24);
        e.advanced.memory_limit_simulation_gb = advanced["memory_limit_simulation_gb"].as<double>(16.0);
    }
    
    // Enhanced parameter parsing using ParameterSpec system
    if (n["experiment"] && n["experiment"]["variables"]) {
        // Handle experiment.variables format from full YAML configs
        const auto &variables = n["experiment"]["variables"];
        for (const auto& kv : variables) {
            std::string name = kv.first.as<std::string>();
            const auto& varConfig = kv.second;
            
            ParameterConfiguration spec;
            std::string type = varConfig["type"].as<std::string>("range");
            
            if (type == "range") {
                spec.type = ParameterConfiguration::RANGE;
                spec.min = varConfig["min"].as<double>();
                spec.max = varConfig["max"].as<double>();
                spec.step = varConfig["step"].as<double>();
            } else if (type == "list") {
                spec.type = ParameterConfiguration::LIST;
                if (varConfig["values"]) {
                    for (const auto& val : varConfig["values"]) {
                        spec.values.push_back(val.as<double>());
                    }
                }
            } else if (type == "logarithmic") {
                spec.type = ParameterConfiguration::LOGARITHMIC;
                spec.n = varConfig["n"].as<int>();
                spec.base = varConfig["base"].as<double>();
            } else if (type == "custom") {
                spec.type = ParameterConfiguration::CUSTOM;
                if (varConfig["values"]) {
                    for (const auto& val : varConfig["values"]) {
                        spec.values.push_back(val.as<double>());
                    }
                }
            }
            
            // Store in enhanced variables map
            e.variables[name] = spec;
            
            // Also populate legacy sweep for backward compatibility
            std::vector<double> values = spec.generate();
            for (double v : values) {
                e.sweep[name].push_back(std::to_string(v));
            }
            
            std::cout << "Parameter " << name << " (" << type << ") has " << values.size() << " values" << std::endl;
        }
    } else if (n["parameters"]) {
        // Handle parameters array format
        const auto &params = n["parameters"];
        for (const auto& param : params) {
            std::string name = param["name"].as<std::string>();
            const auto& spec_node = param["spec"];
            
            ParameterConfiguration spec;
            if (spec_node["values"]) {
                // Explicit values list
                spec.type = ParameterConfiguration::LIST;
                for (const auto& val : spec_node["values"]) {
                    spec.values.push_back(val.as<double>());
                }
            } else if (spec_node["min"] && spec_node["max"] && spec_node["step"]) {
                // Range specification
                spec.type = ParameterConfiguration::RANGE;
                spec.min = spec_node["min"].as<double>();
                spec.max = spec_node["max"].as<double>();
                spec.step = spec_node["step"].as<double>();
            } else if (spec_node["n"] && spec_node["base"]) {
                // Logarithmic specification
                spec.type = ParameterConfiguration::LOGARITHMIC;
                spec.n = spec_node["n"].as<int>();
                spec.base = spec_node["base"].as<double>();
            }
            
            // Store in enhanced variables map
            e.variables[name] = spec;
            
            // Also populate legacy sweep for backward compatibility
            std::vector<double> values = spec.generate();
            for (double v : values) {
                e.sweep[name].push_back(std::to_string(v));
            }
            
            std::cout << "Parameter " << name << " has " << values.size() << " values" << std::endl;
        }
    } else if (n["sweep"]) {
        // Legacy sweep format
        const auto &sweep = n["sweep"];
        for (const auto& kv : sweep) {
            std::string key = kv.first.as<std::string>();
            const auto& arr = kv.second;
            
            ParameterConfiguration spec;
            spec.type = ParameterConfiguration::LIST;
            for (const auto &v : arr) {
                e.sweep[key].push_back(v.as<std::string>());
                spec.values.push_back(std::stod(v.as<std::string>()));
            }
            e.variables[key] = spec;
        }
    }
    return e;
}