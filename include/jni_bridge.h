#pragma once

// jni_bridge.h
#include <string>

struct ExperimentConfig;  // forward declaration (included via benchmark_types.hpp)

//----------------------------------------------------
// High-level JNI entry points exposed to C++
//----------------------------------------------------
double jni_run(int ticks,
               const std::string& modelPath,
               const std::string& paramString,
               int seed,
               bool headless = true);

// Wrapper launching a NetLogo headless simulation and returning numeric result