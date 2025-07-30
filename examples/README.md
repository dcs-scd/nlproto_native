# NetLogo Parallel Simulation System - Usage Examples

This directory contains comprehensive usage examples for the NetLogo parallel simulation system, ranging from beginner to advanced scenarios.

## 📚 Example Categories

### **Basic Examples (Start Here)**
1. `01_beginner_basic_sweep.yaml` - Simple 3x3 parameter grid (9 combinations)
2. `02_intermediate_multiple_params.yaml` - Multiple parameter types and metrics
3. `example_commands.md` - Command-line usage examples

### **Advanced Configuration**
4. `03_advanced_high_resolution.yaml` - Large-scale experiment (1,000+ combinations)
5. `04_performance_benchmark.yaml` - Benchmarking and optimization
6. `05_parameter_specification_showcase.yaml` - All parameter specification methods
7. `06_research_design_patterns.yaml` - Research methodology examples

### **Execution Modes**
8. `07_execution_modes_demo.yaml` - Different execution backends
9. `08_performance_tuning_guide.md` - Performance optimization guide

### **Domain-Specific Examples**
10. `09_ecology_predator_prey.yaml` - Ecological modeling
11. `10_social_segregation_schelling.yaml` - Social science research
12. `11_economics_market_dynamics.yaml` - Economic modeling
13. `12_epidemiology_disease_spread.yaml` - Public health modeling

### **Analysis & Integration**
14. `13_output_analysis_guide.md` - Data analysis workflows
15. `14_troubleshooting_guide.md` - Common issues and solutions
16. `15_integration_workflows.md` - Integration with R, Python, Docker
17. `16_user_personas_examples.md` - Examples by user type

## 🚀 Quick Start

**For Beginners:**
```bash
# Start with basic parameter sweep
python3 run_netlogo.py examples/01_beginner_basic_sweep.yaml

# Or use the high-performance C++ runner
./netlogo_runner examples/01_beginner_basic_sweep.yaml
```

**For Researchers:**
```bash
# High-resolution experiment with benchmarking
./netlogo_runner examples/03_advanced_high_resolution.yaml
```

**For Production:**
```bash
# Production deployment with monitoring
docker run -v $(pwd):/workspace netlogo-runner examples/04_performance_benchmark.yaml
```

## 📊 Complexity Progression

| Example | Parameters | Combinations | Est. Runtime | Use Case |
|---------|------------|--------------|--------------|----------|
| 01_beginner | 2 | 9 | 1-2 min | Learning |
| 02_intermediate | 3 | 120 | 10-15 min | Development |
| 03_advanced | 4 | 1,000+ | 2-4 hours | Research |
| 10_social | 5 | 32,800 | 8-12 hours | Publication |

## 🔧 System Requirements by Example

**Basic Examples (01-02):**
- 4+ CPU cores
- 8GB RAM
- NetLogo 6.4.0+
- Java 11+

**Advanced Examples (03-12):**
- 8+ CPU cores
- 16GB+ RAM
- SSD storage recommended
- Consider cluster/cloud deployment

## 📖 Learning Path

1. **Start with `01_beginner_basic_sweep.yaml`** - Learn basic concepts
2. **Progress to `02_intermediate_multiple_params.yaml`** - Multiple parameters
3. **Try `05_parameter_specification_showcase.yaml`** - All parameter types
4. **Explore domain examples (09-12)** - Your research area
5. **Scale up with `03_advanced_high_resolution.yaml`** - Production scale
6. **Optimize with `04_performance_benchmark.yaml`** - Performance tuning

## 🆘 Getting Help

- Check `14_troubleshooting_guide.md` for common issues
- Review `08_performance_tuning_guide.md` for optimization
- See `15_integration_workflows.md` for tool integration
- Consult `16_user_personas_examples.md` for role-specific examples

## 📝 Example Naming Convention

- `01-09`: Complexity/feature progression
- `10-12`: Domain-specific applications
- `13-17`: Analysis and support guides
- All YAML files are complete, runnable configurations
- All MD files contain step-by-step instructions

Happy simulating! 🎯