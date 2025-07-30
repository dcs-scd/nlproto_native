#pragma once

// jni_bridge.h
#include <string>
#include <vector>

struct ExperimentConfig;  // forward declaration (included via benchmark_types.hpp)

//----------------------------------------------------
// High-level JNI entry points exposed to C++
//----------------------------------------------------
double jni_run(int ticks,
               const std::string& modelPath,
               const std::string& paramString,
               int seed,
               bool headless = true);

// New function for metrics collection
std::vector<double> jni_run_with_metrics(int ticks,
                                        const std::string& modelPath,
                                        const std::string& commands,
                                        int seed,
                                        const std::vector<std::string>& metrics,
                                        bool headless = true);

// Wrapper launching a NetLogo headless simulation and returning numeric result
double run_netlogo_sim(const ExperimentConfig& cfg, const std::string& paramString, int seed);

// New wrapper for metrics collection
std::vector<double> run_netlogo_sim_with_metrics(const ExperimentConfig& cfg, const std::string& paramString, int seed);