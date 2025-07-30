// src/config.cpp
#include "config.hpp"
#include "benchmark_types.hpp"
// #include "experiment_config.hpp"  // Not needed - ExperimentConfig is in benchmark_types.hpp
#include <yaml-cpp/yaml.h>
#include <thread>
#include <string>
#include <iostream>

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
    e.name       = n["name"].as<std::string>("run");
    e.ticks      = n["ticks"].as<int>(1000);
    e.reps       = n["reps"].as<int>(1);
    e.threads    = n["parallel"].as<int>(0);
    
    // Handle experiment.runtime field
    if (n["experiment"] && n["experiment"]["runtime"]) {
        e.ticks = n["experiment"]["runtime"].as<int>();
    }
    // Handle both simple model field and complex configuration
    if (n["model"]) {
        e.model_path = n["model"].as<std::string>();
    } else if (n["modelpath"]) {
        e.model_path = n["modelpath"].as<std::string>();
    }
    if (n["nlpath"]) {
        e.nlpath = n["nlpath"].as<std::string>();
    }
    
    // Handle both simple sweep and complex parameters formats
    if (n["sweep"]) {
        const auto &sweep = n["sweep"];
        for (const auto& kv : sweep) {
            std::string key = kv.first.as<std::string>();
            const auto& arr = kv.second;
            for (const auto &v : arr) e.sweep[key].push_back(v.as<std::string>());
        }
    } else if (n["experiment"] && n["experiment"]["variables"]) {
        // Handle experiment.variables format from full YAML configs
        const auto &variables = n["experiment"]["variables"];
        for (const auto& kv : variables) {
            std::string name = kv.first.as<std::string>();
            const auto& varConfig = kv.second;
            
            std::vector<std::string> values;
            if (varConfig["type"].as<std::string>() == "range") {
                double min = varConfig["min"].as<double>();
                double max = varConfig["max"].as<double>();
                double step = varConfig["step"].as<double>();
                for (double v = min; v <= max + 1e-9; v += step) {
                    values.push_back(std::to_string(static_cast<int>(v)));
                }
            }
            e.sweep[name] = values;
            std::cout << "Parameter " << name << " has " << values.size() << " values" << std::endl;
        }
    } else if (n["parameters"]) {
        const auto &params = n["parameters"];
        for (const auto& param : params) {
            std::string name = param["name"].as<std::string>();
            const auto& spec = param["spec"];
            
            std::vector<std::string> values;
            if (spec["values"]) {
                // Explicit values list
                for (const auto& val : spec["values"]) {
                    values.push_back(std::to_string(val.as<double>()));
                }
            } else if (spec["min"] && spec["max"] && spec["step"]) {
                // Range specification
                double min = spec["min"].as<double>();
                double max = spec["max"].as<double>();
                double step = spec["step"].as<double>();
                for (double v = min; v <= max + 1e-9; v += step) {
                    values.push_back(std::to_string(v));
                }
            } else if (spec["n"] && spec["base"]) {
                // Logarithmic specification
                int n = spec["n"].as<int>();
                double base = spec["base"].as<double>();
                for (int i = 0; i < n; i++) {
                    double val = std::pow(base, i - n + 1);
                    values.push_back(std::to_string(val));
                }
            }
            e.sweep[name] = values;
            // Debug output
            std::cout << "Parameter " << name << " has " << values.size() << " values: ";
            for (const auto& v : values) std::cout << v << " ";
            std::cout << std::endl;
        }
    }
    return e;
}