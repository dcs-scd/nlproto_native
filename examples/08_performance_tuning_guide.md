# Performance Tuning Guide for NetLogo Parallel Simulation System

## Understanding the Performance Architecture

The nlproto_native system uses a sophisticated multi-layer performance optimization strategy:

### 1. Automatic Hardware Detection
- **CPU Topology**: Uses hwloc library to detect physical vs logical cores
- **Memory Configuration**: Reads `/proc/meminfo` for optimal heap sizing
- **NUMA Awareness**: Optimizes thread affinity on multi-socket systems

### 2. JVM Optimization Stack
- **EpsilonGC**: Zero-overhead garbage collection for allocation-light workloads
- **Tiered Compilation**: Fast JIT compilation (TieredStopAtLevel=1)
- **Heap Preallocation**: AlwaysPreTouch for consistent memory performance
- **Transparent Huge Pages**: Reduces TLB pressure on large heaps

### 3. Parallel Execution Engine
- **Worker Pool**: Thread count automatically set to physical cores (max 16)
- **Job Stealing**: Lock-free work distribution for load balancing
- **JNI Bridge**: Minimal overhead native-to-Java interface
- **Workspace Reuse**: NetLogo model loaded once, reused across all runs

## Performance Profiling Examples

### Benchmark Different Parameter Space Sizes
```bash
# Small space (quick baseline)
echo "model: models/sample.nlogo
ticks: 100
parameters:
  - name: param1
    spec: {min: 0, max: 1, step: 0.5}
  - name: param2  
    spec: {values: [0.1, 0.5, 0.9]}
repetitions: {reps: [1]}
metrics:
  - name: metric1" > small_space.yaml

time ./nlproto_native small_space.yaml

# Medium space 
echo "model: models/sample.nlogo
ticks: 100
parameters:
  - name: param1
    spec: {min: 0, max: 1, step: 0.1}  # 11 values
  - name: param2
    spec: {min: 0, max: 1, step: 0.1}  # 11 values
repetitions: {reps: [1]}
metrics:
  - name: metric1" > medium_space.yaml

time ./nlproto_native medium_space.yaml

# Large space
echo "model: models/sample.nlogo
ticks: 100  
parameters:
  - name: param1
    spec: {min: 0, max: 1, step: 0.02}  # 51 values
  - name: param2
    spec: {min: 0, max: 1, step: 0.02}  # 51 values
repetitions: {reps: [1]}
metrics:
  - name: metric1" > large_space.yaml

time ./nlproto_native large_space.yaml
```

### Memory Usage Profiling
```bash
# Monitor memory during execution
echo "#!/bin/bash
exec > memory_profile.log 2>&1
while true; do
    echo \"\$(date): \$(free -h | grep Mem)\"
    sleep 1
done" > monitor_memory.sh
chmod +x monitor_memory.sh

# Run memory monitor in background
./monitor_memory.sh &
MONITOR_PID=$!

# Run simulation
./nlproto_native examples/03_advanced_high_resolution.yaml

# Stop monitoring
kill $MONITOR_PID

# Analyze memory usage
grep "Mem:" memory_profile.log | tail -20
```

### CPU Utilization Analysis
```bash
# Use htop with logging
htop -d 10 > cpu_usage.log &
HTOP_PID=$!

./nlproto_native examples/03_advanced_high_resolution.yaml

kill $HTOP_PID

# Alternative: Use sar for detailed CPU metrics
sar -u 1 > cpu_detailed.log &
SAR_PID=$!

./nlproto_native examples/03_advanced_high_resolution.yaml

kill $SAR_PID
```

## Optimization Strategies by Use Case

### 1. Maximizing Throughput (HPC Clusters)
```yaml
# hpc_optimized.yaml
model: "models/sample.nlogo"
ticks: 200  # Balance between detail and speed

parameters:
  # Large parameter space for cluster efficiency
  - name: "param1"
    spec:
      min: 0.0
      max: 10.0
      step: 0.1  # 101 values
  - name: "param2"  
    spec:
      min: 0.0
      max: 10.0
      step: 0.1  # 101 values

# Fixed repetitions for predictable runtime
repetitions:
  reps: [5]

# Minimal metrics for I/O efficiency
metrics:
  - name: "primary_metric"
```

```bash
# Run with maximum efficiency
export OMP_NUM_THREADS=$(nproc)
export MALLOC_ARENA_MAX=4
./nlproto_native hpc_optimized.yaml
```

### 2. Memory-Constrained Systems
```yaml
# memory_efficient.yaml
model: "models/sample.nlogo"
ticks: 50   # Shorter runs to reduce memory pressure

parameters:
  # Smaller parameter space
  - name: "param1"
    spec:
      values: [0.1, 0.3, 0.5, 0.7, 0.9]  # Explicit values only
  - name: "param2"
    spec:
      min: 0.0
      max: 1.0
      step: 0.2  # Larger steps = fewer combinations

repetitions:
  reps: [3]  # Fewer repetitions

metrics:
  - name: "essential_metric"  # Single metric only
```

### 3. Development and Debugging
```yaml
# debug_fast.yaml
model: "models/sample.nlogo" 
ticks: 25   # Very short runs for quick iteration

parameters:
  - name: "debug_param"
    spec:
      values: [0.0, 0.5, 1.0]  # Just a few test values

repetitions:
  reps: [1]  # Single run for speed

metrics:
  - name: "debug_output"
```

## System-Specific Optimizations

### Linux (Ubuntu/CentOS/RHEL)
```bash
# Enable transparent huge pages
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Set CPU governor to performance mode
sudo cpupower frequency-set -g performance

# Increase file descriptor limits for large parameter sweeps
ulimit -n 65536

# NUMA optimization for multi-socket systems
numactl --interleave=all ./nlproto_native config.yaml
```

### macOS
```bash
# Increase shared memory limits
sudo sysctl -w kern.sysv.shmmax=134217728
sudo sysctl -w kern.sysv.shmall=32768

# Set energy preferences for maximum performance
sudo pmset -a lowpowermode 0
sudo pmset -a powernap 0
```

### Windows (if supported)
```cmd
REM Set Windows performance mode
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

REM Increase virtual memory if needed
wmic computersystem where name="%computername%" set AutomaticManagedPagefile=False
```

## Monitoring and Diagnostics

### Real-time Performance Dashboard
```bash
# Create simple performance monitor
echo "#!/bin/bash
while true; do
    clear
    echo \"=== NetLogo Simulation Performance Monitor ===\"
    echo \"Time: \$(date)\"
    echo \"CPU Usage: \$(top -bn1 | grep 'Cpu(s)' | awk '{print \$2}')\"
    echo \"Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
    echo \"Processes: \$(pgrep -c nlproto_native) nlproto_native instances\"
    echo \"Load Average: \$(uptime | awk -F'load average:' '{print \$2}')\"
    echo \"\"
    echo \"Press Ctrl+C to stop monitoring\"
    sleep 2
done" > perf_monitor.sh
chmod +x perf_monitor.sh

# Run monitor in separate terminal
./perf_monitor.sh
```

### Benchmark Comparison Script
```bash
# benchmark_configs.sh
#!/bin/bash

configs=("small_space.yaml" "medium_space.yaml" "large_space.yaml")
results_file="benchmark_results.txt"

echo "Configuration,Total_Jobs,Runtime_Seconds,Jobs_Per_Second,Peak_Memory_MB" > $results_file

for config in "${configs[@]}"; do
    echo "Benchmarking $config..."
    
    # Get job count
    jobs=$(./nlproto_native $config 2>&1 | grep "Sweep total" | awk '{print $3}')
    
    # Time execution and capture memory
    /usr/bin/time -v ./nlproto_native $config 2>&1 | \
    awk -v conf="$config" -v jobs="$jobs" '
    /User time/ { user_time = $4 }
    /Maximum resident set size/ { max_memory = $6/1024 }  # Convert to MB
    END { 
        throughput = jobs / user_time
        print conf "," jobs "," user_time "," throughput "," max_memory 
    }' >> $results_file
done

echo "Benchmark completed. Results in $results_file"
column -t -s, $results_file
```

## Performance Troubleshooting

### Common Performance Issues

#### Issue: Low CPU Utilization
```bash
# Check if limited by thread count
htop -t  # Show threads

# Verify all cores are being used
mpstat -P ALL 1

# Solution: The system auto-detects optimal thread count,
# but you can verify it's working correctly:
./nlproto_native config.yaml 2>&1 | grep -i thread
```

#### Issue: Memory Exhaustion
```bash
# Monitor memory usage over time
vmstat 1 > memory_usage.log &
./nlproto_native large_config.yaml
killall vmstat

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./nlproto_native config.yaml

# Solution: Reduce parameter space or increase system memory
```

#### Issue: I/O Bottlenecks
```bash
# Monitor disk I/O
iostat -x 1 > io_usage.log &
./nlproto_native config.yaml  
killall iostat

# Check if results compression helps
ls -lh results.*
zstd -d results.csv.zst && ls -lh results.csv

# Solution: Use SSD storage, enable compression
```

### Performance Validation
```bash
# Quick performance validation script
echo "#!/bin/bash
echo 'Running performance validation...'

# Test 1: Small workload baseline
echo 'Test 1: Baseline (should complete in <5 seconds)'
time timeout 10s ./nlproto_native examples/01_beginner_basic_sweep.yaml

# Test 2: CPU scaling test  
echo 'Test 2: CPU utilization check'
./nlproto_native examples/02_intermediate_multiple_params.yaml &
PID=\$!
sleep 5
cpu_usage=\$(top -bn1 -p \$PID | tail -1 | awk '{print \$9}')
echo \"CPU usage: \$cpu_usage%\"
wait \$PID

# Test 3: Memory efficiency
echo 'Test 3: Memory usage check'
/usr/bin/time -v ./nlproto_native examples/01_beginner_basic_sweep.yaml 2>&1 | grep 'Maximum resident set size'

echo 'Performance validation complete'
" > validate_performance.sh
chmod +x validate_performance.sh

./validate_performance.sh
```