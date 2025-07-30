# Troubleshooting Guide for NetLogo Parallel Simulation System

## Quick Diagnostic Commands

### 1. System Health Check
```bash
# Check if the binary exists and is executable
ls -la nlproto_native
file nlproto_native

# Test with minimal configuration
echo "model: models/sample.nlogo
ticks: 10
parameters:
  - name: test-param
    spec: {values: [1, 2]}
repetitions: {reps: [1]}
metrics:
  - name: test-metric" > test_minimal.yaml

./nlproto_native test_minimal.yaml
```

### 2. Dependency Check
```bash
# Check required libraries
ldd nlproto_native  # Linux
otool -L nlproto_native  # macOS

# Verify specific dependencies
pkg-config --exists yaml-cpp && echo "yaml-cpp: OK" || echo "yaml-cpp: MISSING"
pkg-config --exists zstd && echo "zstd: OK" || echo "zstd: MISSING"
ldconfig -p | grep hwloc  # Linux
```

### 3. Java Environment Check
```bash
# Verify Java installation
java -version
javac -version

# Check NetLogo JAR accessibility
find /Applications -name "netlogo-*.jar" 2>/dev/null  # macOS
find /opt -name "netlogo-*.jar" 2>/dev/null  # Linux

# Test Java classpath
java -cp "/path/to/netlogo-6.4.0.jar" org.nlogo.headless.Main --version
```

## Common Error Scenarios and Solutions

### Error 1: "Model file not found"
```
Error: Model file not found: models/sample.nlogo
```

**Diagnosis:**
```bash
# Check if model file exists
ls -la models/sample.nlogo

# Check YAML configuration
grep "model:" your_config.yaml

# Verify path is relative to YAML file location
pwd
realpath models/sample.nlogo
```

**Solutions:**
```bash
# Solution A: Fix path in YAML
sed -i 's|model: "models/sample.nlogo"|model: "/full/path/to/model.nlogo"|' config.yaml

# Solution B: Create missing model directory
mkdir -p models
cp /path/to/existing/model.nlogo models/

# Solution C: Use absolute path
echo 'model: "/Applications/NetLogo 6.4.0/models/Sample Models/Biology/Evolution/Peppered Moths.nlogo"' > fixed_config.yaml
```

### Error 2: "JVM failed to start"
```
Error: JVM initialization failed
```

**Diagnosis:**
```bash
# Check Java installation
which java
java -version

# Test JVM startup manually
java -Xmx1G -version

# Check available memory
free -h  # Linux
vm_stat  # macOS

# Check JVM arguments
echo $JAVA_OPTS
```

**Solutions:**
```bash
# Solution A: Reduce memory requirements
export JAVA_OPTS="-Xmx512m"
./nlproto_native config.yaml

# Solution B: Install/update Java
# Ubuntu/Debian:
sudo apt update && sudo apt install openjdk-11-jdk

# macOS:
brew install openjdk@11

# Solution C: Fix Java path
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"  # Linux
export JAVA_HOME="/usr/local/opt/openjdk@11"  # macOS
```

### Error 3: "YAML parsing error"
```
Error: Failed to parse YAML configuration
```

**Diagnosis:**
```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for common YAML issues
cat -A config.yaml | head -20  # Show hidden characters

# Validate with online tool
curl -X POST -H "Content-Type: application/x-yaml" \
     --data-binary @config.yaml \
     https://yaml-online-parser.appspot.com/
```

**Solutions:**
```bash
# Solution A: Fix indentation (use spaces, not tabs)
sed -i 's/\t/  /g' config.yaml

# Solution B: Fix common YAML errors
# Remove trailing spaces
sed -i 's/[[:space:]]*$//' config.yaml

# Solution C: Use YAML validator
pip install yamllint
yamllint config.yaml

# Solution D: Recreate from template
cp examples/01_beginner_basic_sweep.yaml my_fixed_config.yaml
# Edit my_fixed_config.yaml with your parameters
```

### Error 4: "Out of memory"
```
Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
```

**Diagnosis:**
```bash
# Check available system memory
free -h
top -o MEM  # Check current memory usage

# Estimate memory requirements
wc -l config.yaml  # Check parameter space size
echo "Estimated combinations: (calculate from your parameters)"

# Check current JVM settings
ps aux | grep java | grep nlproto
```

**Solutions:**
```bash
# Solution A: Reduce parameter space
echo "model: models/sample.nlogo
ticks: 50  # Reduced from higher value
parameters:
  - name: param1
    spec: {min: 0, max: 1, step: 0.2}  # Larger step = fewer values
repetitions: {reps: [1]}  # Fewer repetitions
metrics:
  - name: metric1" > memory_efficient.yaml

# Solution B: Split large experiments
# Create multiple smaller YAML files
split_experiment() {
    echo "Creating smaller experiment chunks..."
    # Split parameter ranges into smaller pieces
}

# Solution C: Increase system swap (if needed)
sudo fallocate -l 4G /swapfile  # Linux
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Error 5: "Segmentation fault"
```
Segmentation fault (core dumped)
```

**Diagnosis:**
```bash
# Enable core dumps
ulimit -c unlimited

# Run with debugger
gdb --args ./nlproto_native config.yaml
# In gdb: run, then when it crashes: bt

# Check for memory issues
valgrind --tool=memcheck ./nlproto_native config.yaml

# Verify binary integrity
md5sum nlproto_native
strip --only-keep-debug nlproto_native -o nlproto_native.debug
```

**Solutions:**
```bash
# Solution A: Rebuild the binary
make clean
make all

# Solution B: Update system libraries
sudo apt update && sudo apt upgrade  # Ubuntu/Debian
brew update && brew upgrade  # macOS

# Solution C: Check for library conflicts
ldd nlproto_native | grep "not found"

# Solution D: Use minimal test case
echo "model: models/sample.nlogo
ticks: 1
parameters:
  - name: test
    spec: {values: [1]}
repetitions: {reps: [1]}  
metrics:
  - name: test" > debug_minimal.yaml

./nlproto_native debug_minimal.yaml
```

### Error 6: "Permission denied"
```
bash: ./nlproto_native: Permission denied
```

**Solutions:**
```bash
# Make binary executable
chmod +x nlproto_native

# Check ownership
ls -la nlproto_native
sudo chown $USER:$USER nlproto_native

# Run with explicit interpreter (if needed)
bash -x ./nlproto_native config.yaml
```

### Error 7: "Library not found"
```
./nlproto_native: error while loading shared libraries: libyaml-cpp.so.0.6: cannot open shared object file
```

**Solutions:**
```bash
# Solution A: Install missing libraries
# Ubuntu/Debian:
sudo apt install libyaml-cpp-dev libzstd-dev libhwloc-dev

# CentOS/RHEL:
sudo yum install yaml-cpp-devel libzstd-devel hwloc-devel

# macOS:
brew install yaml-cpp zstd hwloc

# Solution B: Update library cache
sudo ldconfig  # Linux

# Solution C: Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
./nlproto_native config.yaml
```

### Error 8: "NetLogo model won't load"
```
Error: Unable to open NetLogo model: Virus_Spread.nlogo
```

**Diagnosis:**
```bash
# Check model file format
file models/Virus_Spread.nlogo
head -5 models/Virus_Spread.nlogo

# Verify NetLogo version compatibility
grep "NetLogo" models/Virus_Spread.nlogo

# Test model independently
java -cp "/path/to/netlogo.jar" org.nlogo.headless.Main \
     --model models/Virus_Spread.nlogo \
     --experiment test
```

**Solutions:**
```bash
# Solution A: Download correct model
wget "https://github.com/NetLogo/models/raw/master/Sample%20Models/Biology/Virus.nlogo" \
     -O models/Virus_Spread.nlogo

# Solution B: Create simple test model
echo 'NetLogo 6.4.0

turtles-own []

to setup
  clear-all
  create-turtles 10
  reset-ticks
end

to go
  ask turtles [ forward 1 ]
  tick
end' > models/test_model.nlogo

# Solution C: Use absolute path to known good model
find /Applications -name "*.nlogo" | head -5  # macOS
find /opt -name "*.nlogo" | head -5  # Linux
```

## Performance Issues

### Issue: Slow execution
```bash
# Diagnosis
time ./nlproto_native config.yaml

# Profile CPU usage
htop &
./nlproto_native config.yaml

# Check I/O bottlenecks
iostat -x 1 &
./nlproto_native config.yaml
```

**Solutions:**
```bash
# Solution A: Optimize configuration
echo "# Use shorter simulations for testing
ticks: 50  # Instead of 500

# Reduce parameter resolution
step: 0.1  # Instead of 0.01

# Fewer repetitions initially
reps: [3]  # Instead of [10, 15, 20]" >> optimization_notes.txt

# Solution B: Check system resources
free -h
nproc
df -h .

# Solution C: Use performance mode
sudo cpupower frequency-set -g performance  # Linux
```

### Issue: Results files not created
```bash
# Check disk space
df -h .

# Check permissions
ls -la .
touch test_write_permission && rm test_write_permission

# Verify output location
strace -e trace=openat ./nlproto_native config.yaml 2>&1 | grep results
```

## Advanced Debugging

### Memory Debugging
```bash
# Detailed memory analysis
valgrind --tool=massif ./nlproto_native config.yaml

# Analyze memory usage over time
ms_print massif.out.* > memory_analysis.txt

# Check for memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./nlproto_native config.yaml
```

### Performance Profiling
```bash
# CPU profiling with perf (Linux)
perf record -g ./nlproto_native config.yaml
perf report

# System call tracing
strace -c ./nlproto_native config.yaml

# Function call profiling
gprof ./nlproto_native gmon.out > profile_analysis.txt
```

### Network/Cluster Debugging
```bash
# Test on different machines
rsync -av nlproto_native config.yaml user@remote:/tmp/
ssh user@remote "cd /tmp && ./nlproto_native config.yaml"

# Check NFS/shared storage issues
df -T .
mount | grep nfs

# MPI/cluster specific (if applicable)
mpirun --version
srun --version
```

## Emergency Recovery Procedures

### Data Recovery
```bash
# Recover partial results from interrupted runs
ls -la results*
zstd -t results.csv.zst  # Test compressed file integrity

# Extract what's available
zstd -d results.csv.zst
wc -l results.csv

# Resume from checkpoint (if supported)
tail -1 results.csv  # Check last completed run
```

### System Recovery
```bash
# Clean up after crashes
killall -9 java  # Kill hanging Java processes
rm -f /tmp/.java_pid*  # Clean temporary files
ipcs -m | grep $USER | awk '{print $2}' | xargs -r ipcrm -m  # Clean shared memory

# Reset environment
unset JAVA_OPTS
unset LD_LIBRARY_PATH
source ~/.bashrc
```

### Complete Rebuild
```bash
# Clean rebuild process
make clean
rm -f nlproto_native
git status  # Check for uncommitted changes

# Rebuild with debug symbols
g++ -g -O0 -DDEBUG \
    -I/usr/local/include -L/usr/local/lib \
    -ljni -lhwloc -lzstd -lyaml-cpp \
    main.cpp benchmark.cpp csv_json_writer.cpp config.cpp \
    -o nlproto_native_debug

# Test debug version
./nlproto_native_debug config.yaml
```

## Getting Help

### Information Gathering
```bash
# System information script
echo "=== System Information ===" > debug_info.txt
uname -a >> debug_info.txt
cat /etc/os-release >> debug_info.txt  # Linux
sw_vers >> debug_info.txt  # macOS

echo -e "\n=== Java Information ===" >> debug_info.txt
java -version 2>&1 >> debug_info.txt

echo -e "\n=== Dependencies ===" >> debug_info.txt
ldd nlproto_native >> debug_info.txt 2>&1

echo -e "\n=== Configuration ===" >> debug_info.txt
head -20 config.yaml >> debug_info.txt

echo -e "\n=== Error Output ===" >> debug_info.txt
./nlproto_native config.yaml 2>&1 | head -50 >> debug_info.txt
```

### Minimal Reproducible Example
```bash
# Create minimal failing example
cat > minimal_failing.yaml << EOF
model: "models/sample.nlogo"
ticks: 10
parameters:
  - name: "test-param"
    spec:
      values: [1.0, 2.0]
repetitions:
  reps: [1]
metrics:
  - name: "count-turtles"
EOF

./nlproto_native minimal_failing.yaml > output.log 2>&1
```

### Community Resources
- Check GitHub issues: Search for similar problems
- NetLogo documentation: Official NetLogo guides
- Performance benchmarks: Compare with known good configurations
- System requirements: Verify minimum hardware/software requirements