# User Persona Examples for NetLogo Parallel Simulation System

## Persona 1: Graduate Student (First-time User)

**Background**: Sarah is a PhD student in ecology studying predator-prey dynamics. She's familiar with NetLogo GUI but new to command-line tools and parameter sweeps.

**Needs**: Simple setup, clear documentation, quick results for thesis chapter.

### Sarah's First Simulation
```yaml
# sarah_first_experiment.yaml
# Simple predator-prey parameter sweep for thesis research

model: "models/Wolf_Sheep_Predation.nlogo"
ticks: 200  # Moderate length for meaningful dynamics

# Start with just two key parameters
parameters:
  - name: "initial-number-sheep"
    spec:
      values: [50, 100, 150]  # Just 3 values to start
  
  - name: "initial-number-wolves"  
    spec:
      values: [10, 20, 30]  # 3x3 = 9 combinations total

# Single repetition for quick results
repetitions:
  reps: [1]

# Basic population metrics
metrics:
  - name: "count-sheep"
    aggregate: "mean"
  - name: "count-wolves" 
    aggregate: "mean"
```

### Sarah's Command Sequence
```bash
# Sarah's step-by-step workflow

# Step 1: Test the system works
./nlproto_native examples/01_beginner_basic_sweep.yaml

# Step 2: Run her experiment
./nlproto_native sarah_first_experiment.yaml

# Step 3: Look at results (she prefers CSV)
zstd -d results.csv.zst
head -20 results.csv

# Step 4: Simple analysis in Excel/Google Sheets
# (She copies results.csv into spreadsheet software)
```

### Sarah's Learning Path
1. **Week 1**: Basic parameter sweeps with 2-3 parameters
2. **Week 2**: Adding more repetitions for statistical validity  
3. **Week 3**: Exploring different parameter specification methods
4. **Month 2**: Using R for automated analysis (following guide)

---

## Persona 2: Research Scientist (Expert User)

**Background**: Dr. Martinez is a computational social scientist studying urban segregation. He needs sophisticated parameter sweeps with thousands of combinations for publication-quality research.

**Needs**: High-resolution parameter spaces, robust statistics, automated analysis pipelines.

### Dr. Martinez's Research Configuration
```yaml
# martinez_segregation_study.yaml
# High-resolution Schelling segregation analysis for Nature paper

model: "models/Segregation.nlogo"
ticks: 500  # Long enough for full convergence

parameters:
  # High-resolution similarity preference sweep
  - name: "%-similar-wanted"
    spec:
      min: 10.0
      max: 90.0  
      step: 2.0  # 41 values for smooth curves
  
  # Population density effects
  - name: "density"
    spec:
      min: 60.0
      max: 95.0
      step: 5.0  # 8 values
  
  # Neighborhood size sensitivity
  - name: "neighborhood-radius"
    spec:
      values: [1, 2, 3, 4, 5]  # 5 values
  
  # Mobility constraints
  - name: "mobility-rate"
    spec:
      n: 6
      base: 2.0  # [0.5, 1, 2, 4, 8, 16] exponential spacing

# High statistical power with adaptive repetitions
repetitions:
  reps: [15, 25, 35]  # Start with 15, adapt up to 35 if needed

# Comprehensive segregation metrics
metrics:
  - name: "percent-similar"
    aggregate: "mean"
  - name: "segregation-index" 
    aggregate: "mean"
  - name: "convergence-time"
    aggregate: "mean"
  - name: "spatial-autocorrelation"
    aggregate: "mean"
  - name: "ghetto-formation-rate"
    aggregate: "mean"
```

### Dr. Martinez's Advanced Workflow
```bash
# Dr. Martinez's automated research pipeline

# Step 1: Run large-scale simulation (overnight)
nohup ./nlproto_native martinez_segregation_study.yaml > run.log 2>&1 &

# Step 2: Monitor progress
tail -f run.log

# Step 3: Automated analysis pipeline
python3 << EOF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import zstandard as zstd
import io

# Load results
with open('results.csv.zst', 'rb') as f:
    decompressed = zstd.decompress(f.read())
    data = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))

print(f"Analyzed {len(data)} simulation runs")

# Publication-quality analysis
# 1. Tipping point detection
tipping_analysis = data.groupby(['density', 'neighborhood-radius'])['segregation-index'].apply(
    lambda x: np.where(np.diff(x) > 0.1)[0]  # Detect sharp increases
)

# 2. Phase space mapping
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Critical similarity threshold vs density
pivot = data.pivot_table(values='segregation-index', 
                        index='density', 
                        columns='%-similar-wanted')
sns.heatmap(pivot, ax=axes[0,0], cmap='RdYlBu_r')
axes[0,0].set_title('Segregation Phase Space')

# Save publication figures
plt.savefig('segregation_analysis_nature.png', dpi=300, bbox_inches='tight')
print("Publication figures saved")
EOF

# Step 4: Generate LaTeX tables for paper
python3 generate_latex_tables.py results.csv.zst > tables.tex
```

---

## Persona 3: Undergraduate Researcher (Learning User)

**Background**: Alex is an undergraduate studying computer science with interest in agent-based modeling. He's comfortable with programming but new to research methodology.

**Needs**: Educational examples, progressive complexity, clear explanations.

### Alex's Learning Sequence

#### Week 1: Basic Concepts
```yaml
# alex_week1_simple.yaml
# Learning basic parameter sweeps

model: "models/sample.nlogo"
ticks: 50  # Short for quick experiments

parameters:
  - name: "population-size"
    spec:
      values: [10, 50, 100]  # Clear discrete values

repetitions:
  reps: [3]  # Understand why we need multiple runs

metrics:
  - name: "final-population"
    aggregate: "mean"
```

#### Week 2: Parameter Types
```yaml
# alex_week2_parameters.yaml
# Learning different parameter specification methods

model: "models/sample.nlogo"
ticks: 100

parameters:
  # Linear range
  - name: "param-linear"
    spec:
      min: 0.0
      max: 1.0
      step: 0.25  # [0, 0.25, 0.5, 0.75, 1.0]
  
  # Logarithmic range  
  - name: "param-log"
    spec:
      n: 4
      base: 10.0  # [0.001, 0.01, 0.1, 1.0]

repetitions:
  reps: [5]

metrics:
  - name: "outcome-measure"
    aggregate: "mean"
```

#### Week 3: Statistical Thinking
```yaml
# alex_week3_statistics.yaml
# Understanding repetitions and variance

model: "models/sample.nlogo"
ticks: 150

parameters:
  - name: "noise-level"
    spec:
      values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Different repetition counts to see the effect
repetitions:
  reps: [1, 5, 10, 20]  # See how more reps reduce uncertainty

metrics:
  - name: "outcome-mean"
    aggregate: "mean"
  - name: "outcome-variance"
    aggregate: "mean"
```

### Alex's Analysis Learning Scripts
```python
# alex_learning_analysis.py
# Educational analysis with lots of comments

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data (Alex learns about compressed formats)
import zstandard as zstd
import io

print("Loading simulation results...")
with open('results.csv.zst', 'rb') as f:
    decompressed = zstd.decompress(f.read())
    data = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))

print(f"Loaded {len(data)} simulation results")

# Lesson 1: Basic data exploration
print("\n=== Data Exploration ===")
print("First 5 rows:")
print(data.head())

print("\nBasic statistics:")
print(data.describe())

# Lesson 2: Visualization basics
print("\n=== Creating Visualizations ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Simple scatter plot
axes[0].scatter(data['pa'], data['mean'])
axes[0].set_xlabel('Parameter A')
axes[0].set_ylabel('Mean Response')
axes[0].set_title('Parameter Effect')

# Distribution plot
data['mean'].hist(bins=20, ax=axes[1])
axes[1].set_xlabel('Mean Response')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Response Distribution')

plt.savefig('alex_learning_plots.png')
print("Plots saved as alex_learning_plots.png")

# Lesson 3: Statistical concepts
print("\n=== Statistical Analysis ===")
correlation = data['pa'].corr(data['mean'])
print(f"Correlation between Parameter A and response: {correlation:.3f}")

if correlation > 0.5:
    print("Strong positive relationship!")
elif correlation < -0.5:
    print("Strong negative relationship!")
else:
    print("Weak relationship - might be non-linear or noisy")

# Lesson 4: Understanding variance
print(f"\nMean of responses: {data['mean'].mean():.3f}")
print(f"Standard deviation: {data['mean'].std():.3f}")
print(f"Coefficient of variation: {data['mean'].std()/data['mean'].mean():.3f}")

print("\nAnalysis complete! Try changing parameters and re-running.")
```

---

## Persona 4: Data Scientist (Power User)

**Background**: Dr. Chen is a data scientist at a biotech company using NetLogo to model drug diffusion in tissues. She needs high-performance computing and machine learning integration.

**Needs**: Massive parameter spaces, optimization algorithms, cloud deployment, ML integration.

### Dr. Chen's Production Configuration
```yaml
# chen_drug_diffusion_optimization.yaml
# Large-scale pharmacokinetic parameter optimization

model: "models/Drug_Diffusion.nlogo"
ticks: 1000  # Detailed pharmacokinetic modeling

parameters:
  # Drug properties (log-scale for multiple orders of magnitude)
  - name: "diffusion-coefficient"
    spec:
      n: 15
      base: 10.0  # Wide range: 1e-15 to 1e-1
  
  - name: "binding-affinity"
    spec:
      n: 12
      base: 10.0
  
  - name: "clearance-rate"
    spec:
      n: 10
      base: 2.0
  
  # Tissue properties
  - name: "tissue-density"
    spec:
      min: 0.5
      max: 2.0
      step: 0.05  # 31 values
  
  - name: "vascularization"
    spec:
      min: 0.1
      max: 0.9
      step: 0.02  # 41 values
  
  # Dosing parameters
  - name: "dose-amount"
    spec:
      values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
  
  - name: "dose-frequency"
    spec:
      values: [6, 8, 12, 24, 48]  # Hours between doses

# Robust statistics for regulatory submission
repetitions:
  reps: [20, 30, 50]  # High statistical power

# Comprehensive pharmacokinetic metrics
metrics:
  - name: "peak-concentration"
    aggregate: "mean"
  - name: "time-to-peak"
    aggregate: "mean"
  - name: "auc-24h"  # Area under curve
    aggregate: "mean"
  - name: "clearance-observed"
    aggregate: "mean"
  - name: "steady-state-achieved"
    aggregate: "mean"
  - name: "toxicity-threshold-exceeded"
    aggregate: "mean"
```

### Dr. Chen's ML-Integrated Pipeline
```python
# chen_ml_optimization_pipeline.py
# Machine learning-driven parameter optimization

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna  # Bayesian optimization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import subprocess
import json
import logging

class DrugOptimizationPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.models = {}
        self.optimization_results = {}
        
    def run_initial_sweep(self):
        """Run comprehensive parameter sweep"""
        self.logger.info("Running initial parameter sweep...")
        
        # Run simulation
        result = subprocess.run(['./nlproto_native', 'chen_drug_diffusion_optimization.yaml'],
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")
            
        self.logger.info("Initial sweep completed")
        
    def load_and_preprocess_data(self):
        """Load simulation results and create ML features"""
        # Load data (compressed CSV)
        import zstandard as zstd
        import io
        
        with open('results.csv.zst', 'rb') as f:
            decompressed = zstd.decompress(f.read())
            self.data = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))
        
        # Create derived features for ML
        self.data['dose_intensity'] = self.data['dose-amount'] / self.data['dose-frequency']
        self.data['permeability_index'] = (self.data['diffusion-coefficient'] * 
                                          self.data['vascularization'])
        self.data['therapeutic_index'] = (self.data['peak-concentration'] / 
                                         self.data['toxicity-threshold-exceeded'])
        
        # Log-transform skewed variables
        log_vars = ['diffusion-coefficient', 'binding-affinity', 'clearance-rate']
        for var in log_vars:
            self.data[f'log_{var}'] = np.log10(self.data[var] + 1e-10)
            
        self.logger.info(f"Loaded {len(self.data)} simulation results")
        
    def build_surrogate_models(self):
        """Build ML models to predict outcomes from parameters"""
        
        # Features: all input parameters + derived features
        feature_cols = [col for col in self.data.columns 
                       if col.startswith('log_') or col in [
                           'tissue-density', 'vascularization', 'dose-amount', 
                           'dose-frequency', 'dose_intensity', 'permeability_index'
                       ]]
        
        X = self.data[feature_cols]
        
        # Multiple target variables
        targets = ['peak-concentration', 'auc-24h', 'time-to-peak', 'clearance-observed']
        
        for target in targets:
            y = self.data[target]
            
            # Try multiple algorithms
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
            }
            
            best_score = -np.inf
            best_model = None
            
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                avg_score = cv_scores.mean()
                
                self.logger.info(f"{target} - {name}: R² = {avg_score:.3f} ± {cv_scores.std():.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            # Train best model on full data
            best_model.fit(X, y)
            self.models[target] = {
                'model': best_model,
                'features': feature_cols,
                'score': best_score
            }
            
        self.logger.info("Surrogate models trained")
        
    def bayesian_optimization(self):
        """Use Bayesian optimization to find optimal parameters"""
        
        def objective(trial):
            # Define parameter space for optimization
            params = {
                'log_diffusion-coefficient': trial.suggest_float('log_diffusion', -15, -1),
                'log_binding-affinity': trial.suggest_float('log_binding', -12, -1),
                'log_clearance-rate': trial.suggest_float('log_clearance', -5, 5),
                'tissue-density': trial.suggest_float('tissue_density', 0.5, 2.0),
                'vascularization': trial.suggest_float('vascularization', 0.1, 0.9),
                'dose-amount': trial.suggest_float('dose_amount', 0.1, 50.0),
                'dose-frequency': trial.suggest_categorical('dose_frequency', [6, 8, 12, 24, 48])
            }
            
            # Create feature vector
            X_pred = pd.DataFrame([params])
            X_pred['dose_intensity'] = X_pred['dose-amount'] / X_pred['dose-frequency']
            X_pred['permeability_index'] = (10**X_pred['log_diffusion-coefficient'] * 
                                           X_pred['vascularization'])
            
            # Predict outcomes using surrogate models
            predictions = {}
            for target, model_info in self.models.items():
                X_aligned = X_pred[model_info['features']]
                pred = model_info['model'].predict(X_aligned)[0]
                predictions[target] = pred
            
            # Multi-objective optimization: maximize efficacy, minimize toxicity
            efficacy_score = predictions['auc-24h']  # Higher is better
            toxicity_penalty = max(0, predictions['peak-concentration'] - 100)  # Penalty if > threshold
            time_penalty = max(0, predictions['time-to-peak'] - 12)  # Penalty if too slow
            
            # Combined objective (to maximize)
            objective_value = efficacy_score - 10 * toxicity_penalty - time_penalty
            
            return objective_value
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=500)
        
        self.optimization_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
        
        self.logger.info(f"Optimization completed. Best value: {study.best_value:.3f}")
        
    def validate_optimal_parameters(self):
        """Run actual simulations with optimized parameters to validate"""
        
        # Create validation configuration
        optimal_params = self.optimization_results['best_params']
        
        validation_config = {
            'model': 'models/Drug_Diffusion.nlogo',
            'ticks': 1000,
            'parameters': [],
            'repetitions': {'reps': [50]},  # High repetitions for validation
            'metrics': [
                {'name': 'peak-concentration', 'aggregate': 'mean'},
                {'name': 'auc-24h', 'aggregate': 'mean'},
                {'name': 'time-to-peak', 'aggregate': 'mean'},
                {'name': 'clearance-observed', 'aggregate': 'mean'}
            ]
        }
        
        # Add optimized parameters
        for param, value in optimal_params.items():
            param_config = {
                'name': param.replace('log_', ''),
                'spec': {'values': [10**value if param.startswith('log_') else value]}
            }
            validation_config['parameters'].append(param_config)
        
        # Save validation config
        import yaml
        with open('validation_config.yaml', 'w') as f:
            yaml.dump(validation_config, f)
        
        # Run validation simulation
        result = subprocess.run(['./nlproto_native', 'validation_config.yaml'],
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            self.logger.info("Validation simulation completed successfully")
        else:
            self.logger.error(f"Validation failed: {result.stderr}")
            
    def generate_optimization_report(self):
        """Create comprehensive optimization report"""
        
        # Create interactive plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Parameter Importance', 'Optimization History', 
                           'Predicted vs Actual', 'Parameter Correlations']
        )
        
        # Feature importance for best model
        best_target = max(self.models.keys(), 
                         key=lambda x: self.models[x]['score'])
        importance = self.models[best_target]['model'].feature_importances_
        features = self.models[best_target]['features']
        
        fig.add_trace(
            go.Bar(x=features, y=importance, name='Importance'),
            row=1, col=1
        )
        
        # Optimization history
        study = self.optimization_results['study']
        values = [trial.value for trial in study.trials]
        
        fig.add_trace(
            go.Scatter(x=list(range(len(values))), y=values, name='Objective'),
            row=1, col=2
        )
        
        fig.update_layout(height=800, title="Drug Optimization Results")
        fig.write_html('optimization_report.html')
        
        # Save detailed results
        results_summary = {
            'optimal_parameters': self.optimization_results['best_params'],
            'model_performance': {k: v['score'] for k, v in self.models.items()},
            'optimization_value': self.optimization_results['best_value']
        }
        
        with open('optimization_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        self.logger.info("Optimization report generated")
        
    def run_complete_optimization(self):
        """Execute the complete optimization pipeline"""
        
        logging.basicConfig(level=logging.INFO)
        
        try:
            self.run_initial_sweep()
            self.load_and_preprocess_data()
            self.build_surrogate_models()
            self.bayesian_optimization()
            self.validate_optimal_parameters()
            self.generate_optimization_report()
            
            self.logger.info("Complete optimization pipeline finished successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

# Usage
if __name__ == "__main__":
    pipeline = DrugOptimizationPipeline()
    pipeline.run_complete_optimization()
```

---

## Persona 5: Systems Administrator (Operations User)

**Background**: Jamie manages computing resources for a research institute. They need to deploy, monitor, and maintain NetLogo simulations across multiple systems.

**Needs**: Deployment automation, monitoring, resource management, user support.

### Jamie's Deployment Scripts
```bash
#!/bin/bash
# jamie_deployment_script.sh
# Automated deployment across research cluster

# Configuration
INSTALL_DIR="/opt/netlogo-parallel"
USER_DIR="/shared/netlogo-examples"
LOG_DIR="/var/log/netlogo"

# System requirements check
check_requirements() {
    echo "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "✓ Linux OS detected"
    else
        echo "✗ Linux required"
        exit 1
    fi
    
    # Check dependencies
    for dep in gcc java zstd hwloc yaml-cpp; do
        if command -v $dep &> /dev/null || pkg-config --exists $dep; then
            echo "✓ $dep found"
        else
            echo "✗ $dep missing"
            MISSING_DEPS="$MISSING_DEPS $dep"
        fi
    done
    
    if [ ! -z "$MISSING_DEPS" ]; then
        echo "Installing missing dependencies: $MISSING_DEPS"
        sudo apt-get update
        sudo apt-get install -y gcc openjdk-11-jdk libzstd-dev libhwloc-dev libyaml-cpp-dev
    fi
}

# Install NetLogo parallel system
install_system() {
    echo "Installing NetLogo parallel system..."
    
    sudo mkdir -p $INSTALL_DIR
    sudo mkdir -p $USER_DIR
    sudo mkdir -p $LOG_DIR
    
    # Build and install
    cd /tmp
    git clone https://github.com/your-org/nlproto-native.git
    cd nlproto-native
    
    make clean
    make release
    
    sudo cp nlproto_native $INSTALL_DIR/
    sudo cp -r examples/ $USER_DIR/
    sudo cp -r models/ $USER_DIR/
    
    # Create wrapper script for users
    sudo tee $INSTALL_DIR/nlproto-wrapper.sh > /dev/null << 'EOF'
#!/bin/bash
# NetLogo simulation wrapper with logging and resource limits

USER=$(whoami)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/netlogo/user_${USER}_${TIMESTAMP}.log"

# Resource limits
ulimit -v 8388608  # 8GB virtual memory limit
ulimit -t 14400    # 4 hour CPU time limit

# Log execution
echo "$(date): User $USER starting simulation $1" >> $LOG_FILE

# Run simulation
exec /opt/netlogo-parallel/nlproto_native "$@" 2>&1 | tee -a $LOG_FILE
EOF

    sudo chmod +x $INSTALL_DIR/nlproto-wrapper.sh
    sudo ln -sf $INSTALL_DIR/nlproto-wrapper.sh /usr/local/bin/nlproto
    
    echo "Installation complete"
}

# Setup monitoring
setup_monitoring() {
    echo "Setting up monitoring..."
    
    # Create monitoring script
    sudo tee /opt/netlogo-parallel/monitor.sh > /dev/null << 'EOF'
#!/bin/bash
# NetLogo system monitoring

LOG_FILE="/var/log/netlogo/system_monitor.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Count active simulations
    ACTIVE_SIMS=$(pgrep -c nlproto_native)
    
    # System resources
    CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    # Log metrics
    echo "$TIMESTAMP,active_sims=$ACTIVE_SIMS,cpu_load=$CPU_LOAD,memory_pct=$MEMORY_USAGE,disk_pct=$DISK_USAGE" >> $LOG_FILE
    
    # Alert if resources high
    if (( $(echo "$CPU_LOAD > 20" | bc -l) )) || (( MEMORY_USAGE > 90 )); then
        echo "$(date): HIGH RESOURCE USAGE - CPU: $CPU_LOAD, Memory: $MEMORY_USAGE%" | \
            mail -s "NetLogo System Alert" admin@institution.edu
    fi
    
    sleep 60
done
EOF

    sudo chmod +x /opt/netlogo-parallel/monitor.sh
    
    # Create systemd service
    sudo tee /etc/systemd/system/netlogo-monitor.service > /dev/null << 'EOF'
[Unit]
Description=NetLogo System Monitor
After=network.target

[Service]
Type=simple
ExecStart=/opt/netlogo-parallel/monitor.sh
Restart=always
User=nobody

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl enable netlogo-monitor
    sudo systemctl start netlogo-monitor
    
    echo "Monitoring setup complete"
}

# User management
setup_user_environment() {
    echo "Setting up user environment..."
    
    # Create user template
    sudo tee /etc/skel/.netlogo_profile > /dev/null << 'EOF'
# NetLogo environment setup
export NETLOGO_HOME="/shared/netlogo-examples"
export NETLOGO_MODELS="$NETLOGO_HOME/models"

# Helpful aliases
alias nlproto='nlproto'
alias nlexamples='ls /shared/netlogo-examples/*.yaml'
alias nlstatus='pgrep -l nlproto_native'

# Function to check job status
nlps() {
    echo "Active NetLogo simulations:"
    ps aux | grep nlproto_native | grep -v grep
}

# Function to estimate completion time
nltime() {
    if [ $# -eq 0 ]; then
        echo "Usage: nltime config.yaml"
        return 1
    fi
    
    echo "Estimating runtime for $1..."
    # Simple estimation based on parameter space size
    python3 -c "
import yaml
with open('$1') as f:
    config = yaml.safe_load(f)
    
total_combos = 1
for param in config['parameters']:
    if 'min' in param['spec']:
        n = int((param['spec']['max'] - param['spec']['min']) / param['spec']['step']) + 1
    elif 'values' in param['spec']:
        n = len(param['spec']['values'])
    else:
        n = param['spec']['n']
    total_combos *= n

reps = config['repetitions']['reps'][0]
ticks = config['ticks']

# Rough estimate: 0.1 seconds per tick per repetition
estimated_seconds = total_combos * reps * ticks * 0.1
estimated_hours = estimated_seconds / 3600

print(f'Estimated combinations: {total_combos:,}')
print(f'Estimated runtime: {estimated_hours:.1f} hours')
"
}
EOF

    # Update existing users
    for user_home in /home/*; do
        if [ -d "$user_home" ]; then
            sudo cp /etc/skel/.netlogo_profile "$user_home/"
            sudo chown $(basename $user_home):$(basename $user_home) "$user_home/.netlogo_profile"
        fi
    done
    
    echo "User environment setup complete"
}

# Main installation
main() {
    echo "Starting NetLogo parallel system deployment..."
    
    check_requirements
    install_system
    setup_monitoring
    setup_user_environment
    
    echo ""
    echo "Deployment complete!"
    echo "Users can now run: nlproto config.yaml"
    echo "Examples available in: $USER_DIR"
    echo "Monitoring logs in: $LOG_DIR"
    echo ""
    echo "Test the installation:"
    echo "  nlproto /shared/netlogo-examples/01_beginner_basic_sweep.yaml"
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### Jamie's Monitoring Dashboard
```python
# jamie_monitoring_dashboard.py
# Web dashboard for NetLogo system monitoring

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import subprocess
import psutil
import time
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="NetLogo System Monitor", layout="wide")

def get_system_status():
    """Get current system status"""
    # Active simulations
    try:
        result = subprocess.run(['pgrep', '-c', 'nlproto_native'], 
                               capture_output=True, text=True)
        active_sims = int(result.stdout.strip()) if result.returncode == 0 else 0
    except:
        active_sims = 0
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'active_simulations': active_sims,
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_percent': disk.percent,
        'disk_free_gb': disk.free / (1024**3)
    }

def load_monitoring_logs():
    """Load historical monitoring data"""
    log_file = "/var/log/netlogo/system_monitor.log"
    
    if os.path.exists(log_file):
        try:
            # Read last 1000 lines
            result = subprocess.run(['tail', '-1000', log_file], 
                                   capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            data = []
            for line in lines:
                if line:
                    parts = line.split(',')
                    timestamp = parts[0]
                    metrics = {}
                    for part in parts[1:]:
                        key, value = part.split('=')
                        metrics[key] = float(value)
                    metrics['timestamp'] = pd.to_datetime(timestamp)
                    data.append(metrics)
            
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def get_user_activity():
    """Get user activity information"""
    try:
        # Get users currently running simulations
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        users = {}
        for line in lines:
            if 'nlproto_native' in line:
                parts = line.split()
                if len(parts) > 0:
                    user = parts[0]
                    users[user] = users.get(user, 0) + 1
        
        return users
    except:
        return {}

# Main dashboard
st.title("🧬 NetLogo Parallel System Monitor")

# Real-time status
status = get_system_status()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Active Simulations", status['active_simulations'])

with col2:
    st.metric("CPU Usage", f"{status['cpu_percent']:.1f}%")

with col3:
    st.metric("Memory Usage", f"{status['memory_percent']:.1f}%", 
              delta=f"{status['memory_available_gb']:.1f}GB free")

with col4:
    st.metric("Disk Usage", f"{status['disk_percent']:.1f}%",
              delta=f"{status['disk_free_gb']:.1f}GB free")

# Historical data
st.subheader("Historical Performance")

monitoring_data = load_monitoring_logs()

if not monitoring_data.empty:
    # Create time series plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Active Simulations', 'CPU Load', 'Memory Usage', 'Disk Usage'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Active simulations
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['active_sims'],
                   name='Active Sims', line=dict(color='blue')),
        row=1, col=1
    )
    
    # CPU load
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['cpu_load'],
                   name='CPU Load', line=dict(color='red')),
        row=1, col=2
    )
    
    # Memory usage
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['memory_pct'],
                   name='Memory %', line=dict(color='green')),
        row=2, col=1
    )
    
    # Disk usage  
    fig.add_trace(
        go.Scatter(x=monitoring_data['timestamp'], y=monitoring_data['disk_pct'],
                   name='Disk %', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monitoring data available yet")

# User activity
st.subheader("Current User Activity")

user_activity = get_user_activity()

if user_activity:
    user_df = pd.DataFrame(list(user_activity.items()), 
                          columns=['User', 'Active Jobs'])
    
    fig = px.bar(user_df, x='User', y='Active Jobs', 
                 title='Simulations by User')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No active simulations")

# System administration tools
st.subheader("System Administration")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("View System Logs"):
        try:
            result = subprocess.run(['tail', '-50', '/var/log/netlogo/system_monitor.log'],
                                   capture_output=True, text=True)
            st.text_area("Recent System Logs", result.stdout, height=300)
        except:
            st.error("Could not access system logs")

with col2:
    if st.button("Check Disk Space"):
        try:
            result = subprocess.run(['df', '-h'], capture_output=True, text=True)
            st.text_area("Disk Usage", result.stdout, height=300)
        except:
            st.error("Could not check disk space")

with col3:
    if st.button("List Running Jobs"):
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            netlogo_jobs = [line for line in result.stdout.split('\n') 
                           if 'nlproto' in line]
            st.text_area("NetLogo Jobs", '\n'.join(netlogo_jobs), height=300)
        except:
            st.error("Could not list jobs")

# Auto-refresh
if st.checkbox("Auto-refresh (30 seconds)"):
    time.sleep(30)
    st.experimental_rerun()
```

This comprehensive persona-based guide provides examples tailored to the specific needs, skill levels, and use cases of different user types, from beginners to system administrators.