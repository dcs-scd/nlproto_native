# NetLogo Parallel Simulation System - Command Examples

This document provides practical command-line examples for running the NetLogo parallel simulation system.

## Basic Commands

### 1. Simple Parameter Sweep (Beginner)
```bash
# Run basic parameter sweep
./nlproto_native examples/01_beginner_basic_sweep.yaml

# Expected output:
# [INFO] Sweep total 9 parameter tuples | ticks = 100
# [INFO] Done in 2.3s
```

### 2. Intermediate Multi-Parameter Sweep
```bash
# Run with multiple parameter types
./nlproto_native examples/02_intermediate_multiple_params.yaml

# Expected output:
# [INFO] Sweep total 240 parameter tuples | ticks = 200
# [INFO] Done in 45.7s
```

### 3. High-Resolution Research Sweep
```bash
# Large-scale parameter exploration
./nlproto_native examples/03_advanced_high_resolution.yaml

# Expected output:
# [INFO] Sweep total 32800 parameter tuples | ticks = 500
# [INFO] Done in 1847.2s
```

### 4. Performance Benchmarking
```bash
# Quick performance test
./nlproto_native examples/04_performance_benchmark.yaml

# Expected output:
# [INFO] Sweep total 25 parameter tuples | ticks = 50
# [INFO] Done in 0.8s
```

## Python Interface Commands

### Using the Python runner (alternative approach)
```bash
# Run via Python interface
python run_netlogo.py examples/01_beginner_basic_sweep.yaml

# This generates:
# - examples/01_beginner_basic_sweep.xml (NetLogo BehaviorSpace XML)
# - examples/01_beginner_basic_sweep.csv (Results spreadsheet)
# - examples/01_beginner_basic_sweep_rows.csv (Detailed row data)
```

## Build Commands

### Compiling the Native System
```bash
# Single build script
./build-single.sh

# Manual compilation (if needed)
g++ -std=c++20 -O3 -DNDEBUG \
    -I/usr/local/include -L/usr/local/lib \
    -ljni -lhwloc -lzstd -lyaml-cpp \
    main.cpp benchmark.cpp csv_json_writer.cpp config.cpp \
    -o nlproto_native
```

### Java Component Compilation
```bash
# Compile Java NetLogo interface
javac -cp "/path/to/netlogo-6.4.0.jar" \
      NetLogoRunner.java NativeRuntime.java

# Create JAR if needed
jar cf netlogo-interface.jar *.class
```

## Output Analysis Commands

### Examining Results
```bash
# Decompress and view CSV results
zstd -d results.csv.zst
head -20 results.csv

# Decompress and format JSON results  
zstd -d results.json.zst
jq '.' results.json | head -50

# Quick statistics
zstd -dc results.csv.zst | tail -n +2 | wc -l  # Count data rows
```

### Performance Analysis
```bash
# Time the execution
time ./nlproto_native examples/03_advanced_high_resolution.yaml

# Monitor system resources during execution
htop &
./nlproto_native examples/03_advanced_high_resolution.yaml

# Memory usage tracking
valgrind --tool=massif ./nlproto_native examples/01_beginner_basic_sweep.yaml
```

## Debugging Commands

### Verbose Execution
```bash
# Enable debug output (if available)
export NLPROTO_DEBUG=1
./nlproto_native examples/01_beginner_basic_sweep.yaml

# Check JVM arguments being used
export NLPROTO_SHOW_JVM_ARGS=1
./nlproto_native examples/01_beginner_basic_sweep.yaml
```

### Validation Commands
```bash
# Validate YAML configuration
python -c "import yaml; yaml.safe_load(open('examples/01_beginner_basic_sweep.yaml'))"

# Check NetLogo model exists
ls -la models/sample.nlogo

# Verify Java classpath
java -cp "/path/to/netlogo-6.4.0.jar" -version
```

## System Requirements Check

### Hardware Information
```bash
# Check CPU cores (affects thread count)
nproc                    # Linux
sysctl hw.ncpu          # macOS

# Check available memory (affects heap sizing)
free -h                 # Linux  
vm_stat                 # macOS

# Check disk space for large result files
df -h .
```

### Software Dependencies
```bash
# Check required libraries
ldd nlproto_native      # Linux
otool -L nlproto_native # macOS

# Verify NetLogo installation
java -jar /path/to/netlogo-6.4.0.jar --version

# Check YAML parser
pkg-config --cflags --libs yaml-cpp
```

## Integration Examples

### With R Analysis
```bash
# Run simulation and import to R
./nlproto_native examples/02_intermediate_multiple_params.yaml
zstd -d results.csv.zst

# In R:
# data <- read.csv("results.csv")
# summary(data)
# plot(data$pa, data$mean)
```

### With Python Analysis
```bash
# Run simulation
./nlproto_native examples/03_advanced_high_resolution.yaml

# Python analysis script:
# import pandas as pd
# import json
# 
# # Load compressed JSON
# import zstandard as zstd
# with open('results.json.zst', 'rb') as f:
#     data = json.loads(zstd.decompress(f.read()))
```

### Batch Processing Multiple Experiments
```bash
# Run multiple experiments in sequence
for config in examples/*.yaml; do
    echo "Running $config..."
    ./nlproto_native "$config"
    mv results.csv.zst "results_$(basename $config .yaml).csv.zst"
    mv results.json.zst "results_$(basename $config .yaml).json.zst"
done
```

## Error Handling

### Common Issues and Solutions
```bash
# Issue: "Model file not found"
# Solution: Check model path in YAML
ls -la $(grep "model:" examples/01_beginner_basic_sweep.yaml | cut -d' ' -f2)

# Issue: "JVM failed to start"  
# Solution: Check Java installation and memory
java -version
java -Xmx1g -version

# Issue: "Out of memory"
# Solution: Reduce parameter space or increase system memory
# Edit YAML to reduce tick count or parameter ranges

# Issue: "Segmentation fault"
# Solution: Run with debugger
gdb --args ./nlproto_native examples/01_beginner_basic_sweep.yaml
```

## Performance Tuning Tips

### Optimal Thread Configuration
```bash
# The system auto-detects optimal thread count, but you can influence it:
# - Ensure sufficient memory per thread
# - Consider NUMA topology on large systems
# - Monitor CPU utilization during runs

# Check NUMA information
numactl --hardware
```

### Memory Optimization
```bash
# For large parameter sweeps:
# 1. Monitor memory usage: watch -n 1 'free -h'
# 2. Adjust JVM heap size in benchmark.cpp if needed
# 3. Use compressed output to save disk space
# 4. Consider splitting large sweeps into chunks
```