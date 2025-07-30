#pragma once
// config.hpp
// MIT 2024 — all immutable configuration structs parsed from the YAML sweep file.

#include <array>
#include <string>
#include <variant>
#include <vector>

// ------------------------------------------------------------------
struct RangeSpec {
    double min;
    double max;
    double step;
};
struct LogSpec {
    int n;
    double base = 10.0;
};
struct Vector {
    std::vector<double> values;
};
using ParameterSpec = std::variant<RangeSpec, LogSpec, Vector>;

// ------------------------------------------------------------------
struct Parameter {
    std::string     name;
    ParameterSpec   spec;
};

// ------------------------------------------------------------------
struct Repetitions {
    std::vector<int> reps;               // future: adaptive vector
};

// ------------------------------------------------------------------
struct Metrics {
    std::string name;
    std::string aggregate = "mean";      // reserved
};

// ------------------------------------------------------------------
struct Experiment {
    std::string model;           // *.nlogo (path relative to YAML)
    int         ticks;
    std::vector<Parameter> parameters;
    Repetitions repetitions;
    std::vector<Metrics> metrics;
};

// ------------------------------------------------------------------
// extern helper — see config.cpp
Experiment parse_yaml(const std::string& file);