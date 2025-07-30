# Integration Workflows and Tool Chains

## Research Pipeline Integration

### 1. Complete R Research Pipeline
```r
# complete_research_pipeline.R
# A full workflow from simulation to publication-ready figures

library(tidyverse)
library(ggplot2)
library(viridis)
library(patchwork)
library(broom)
library(knitr)
library(rmarkdown)

# Step 1: Run simulation
system("./nlproto_native examples/03_advanced_high_resolution.yaml")

# Step 2: Load and preprocess data
load_simulation_results <- function() {
  # Load compressed CSV
  system("zstd -d results.csv.zst")
  data <- read_csv("results.csv")
  
  # Add derived variables
  data <- data %>%
    mutate(
      param_ratio = pa / pb,
      outcome_efficiency = mean / (stdev + 1e-6),
      parameter_category = case_when(
        pa < 0.3 ~ "Low",
        pa < 0.7 ~ "Medium", 
        TRUE ~ "High"
      )
    )
  
  return(data)
}

# Step 3: Statistical analysis
perform_analysis <- function(data) {
  # Main effects model
  model_main <- lm(mean ~ pa + pb, data = data)
  
  # Interaction model
  model_interaction <- lm(mean ~ pa * pb + I(pa^2) + I(pb^2), data = data)
  
  # Model comparison
  anova_result <- anova(model_main, model_interaction)
  
  # Effect sizes
  eta_squared <- function(model) {
    aov_result <- aov(model)
    summary_aov <- summary(aov_result)[[1]]
    summary_aov$"Sum Sq" / sum(summary_aov$"Sum Sq")
  }
  
  results <- list(
    model_main = model_main,
    model_interaction = model_interaction,
    anova = anova_result,
    effect_sizes = eta_squared(model_interaction)
  )
  
  return(results)
}

# Step 4: Create publication figures
create_figures <- function(data, analysis) {
  # Figure 1: Parameter space heatmap
  p1 <- ggplot(data, aes(x = pa, y = pb, fill = mean)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Response") +
    labs(title = "A) Response Surface",
         x = "Parameter A", y = "Parameter B") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Figure 2: Main effects
  p2 <- data %>%
    pivot_longer(cols = c(pa, pb), names_to = "parameter", values_to = "value") %>%
    ggplot(aes(x = value, y = mean, color = parameter)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess") +
    scale_color_viridis_d(name = "Parameter") +
    labs(title = "B) Main Effects",
         x = "Parameter Value", y = "Response") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Figure 3: Interaction effects
  p3 <- ggplot(data, aes(x = pa, y = mean, color = factor(round(pb, 2)))) +
    geom_line() +
    geom_point() +
    scale_color_viridis_d(name = "Parameter B") +
    labs(title = "C) Interaction Effects",
         x = "Parameter A", y = "Response") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Figure 4: Model diagnostics
  p4 <- augment(analysis$model_interaction) %>%
    ggplot(aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_smooth(se = FALSE) +
    labs(title = "D) Model Diagnostics",
         x = "Fitted Values", y = "Residuals") +
    theme_minimal()
  
  # Combine figures
  combined_figure <- (p1 | p2) / (p3 | p4)
  
  return(list(
    individual = list(p1 = p1, p2 = p2, p3 = p3, p4 = p4),
    combined = combined_figure
  ))
}

# Step 5: Generate report
generate_report <- function(data, analysis, figures) {
  # Create R Markdown report
  rmd_content <- '
---
title: "NetLogo Simulation Analysis Report"
author: "Automated Pipeline"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    theme: flatly
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
```

## Executive Summary

This report presents the analysis of NetLogo simulation results with `r nrow(data)` parameter combinations tested across `r length(unique(data$pa))` × `r length(unique(data$pb))` parameter space.

## Key Findings

- **Parameter A Effect**: `r if(summary(analysis$model_main)$coefficients[2,4] < 0.05) "Significant" else "Non-significant"` (p = `r round(summary(analysis$model_main)$coefficients[2,4], 4)`)
- **Parameter B Effect**: `r if(summary(analysis$model_main)$coefficients[3,4] < 0.05) "Significant" else "Non-significant"` (p = `r round(summary(analysis$model_main)$coefficients[3,4], 4)`)
- **Interaction Effect**: `r if(summary(analysis$model_interaction)$coefficients[4,4] < 0.05) "Significant" else "Non-significant"` (p = `r round(summary(analysis$model_interaction)$coefficients[4,4], 4)`)

## Visualizations

```{r figures, fig.width=12, fig.height=8}
figures$combined
```

## Statistical Models

### Main Effects Model
```{r main-model}
summary(analysis$model_main) %>% tidy() %>% kable(digits = 4)
```

### Interaction Model
```{r interaction-model}  
summary(analysis$model_interaction) %>% tidy() %>% kable(digits = 4)
```

### Model Comparison
```{r model-comparison}
analysis$anova %>% tidy() %>% kable(digits = 4)
```

## Conclusions

Based on the analysis of simulation results, we found [conclusions would be data-dependent].

## Methodology

- **Simulation System**: NetLogo Parallel Simulation System (nlproto_native)
- **Parameter Space**: Grid search across specified ranges
- **Statistical Analysis**: Linear models with interaction terms
- **Software**: R `r R.version.string`
'
  
  writeLines(rmd_content, "simulation_report.Rmd")
  
  # Render report
  rmarkdown::render("simulation_report.Rmd")
  
  return("simulation_report.html")
}

# Execute complete pipeline
main <- function() {
  cat("Starting research pipeline...\n")
  
  # Load data
  cat("Loading simulation results...\n")
  data <- load_simulation_results()
  
  # Perform analysis
  cat("Performing statistical analysis...\n")
  analysis <- perform_analysis(data)
  
  # Create figures
  cat("Creating figures...\n")
  figures <- create_figures(data, analysis)
  
  # Save figures
  ggsave("research_figures.png", figures$combined, 
         width = 12, height = 8, dpi = 300)
  
  # Generate report
  cat("Generating report...\n")
  report_file <- generate_report(data, analysis, figures)
  
  cat("Pipeline complete! Report saved as:", report_file, "\n")
  
  return(list(
    data = data,
    analysis = analysis,
    figures = figures,
    report = report_file
  ))
}

# Run pipeline
if (interactive() || !exists("sourced")) {
  pipeline_results <- main()
}
```

### 2. Python Data Science Pipeline
```python
# python_research_pipeline.py
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import zstandard as zstd
import json
import io
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetLogoAnalysisPipeline:
    def __init__(self, config_file, output_dir="analysis_output"):
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data = None
        self.results = {}
        
    def run_simulation(self):
        """Execute NetLogo simulation"""
        logger.info(f"Running simulation with config: {self.config_file}")
        
        cmd = ["./nlproto_native", str(self.config_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")
            
        logger.info("Simulation completed successfully")
        return result.stdout
        
    def load_data(self):
        """Load and preprocess simulation results"""
        logger.info("Loading simulation results")
        
        # Load compressed CSV
        with open('results.csv.zst', 'rb') as f:
            decompressed = zstd.decompress(f.read())
            self.data = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))
            
        # Add derived variables
        self.data['param_ratio'] = self.data['pa'] / (self.data['pb'] + 1e-6)
        self.data['efficiency'] = self.data['mean'] / (self.data['stdev'] + 1e-6)
        self.data['param_sum'] = self.data['pa'] + self.data['pb']
        
        logger.info(f"Loaded {len(self.data)} simulation results")
        return self.data
        
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        logger.info("Performing statistical analysis")
        
        # Correlation analysis
        correlation_matrix = self.data[['pa', 'pb', 'mean', 'stdev', 'entropy']].corr()
        
        # ANOVA for parameter effects
        from scipy.stats import f_oneway
        
        # Group by parameter levels for ANOVA
        pa_groups = [group['mean'].values for name, group in self.data.groupby('pa')]
        pb_groups = [group['mean'].values for name, group in self.data.groupby('pb')]
        
        anova_pa = f_oneway(*pa_groups)
        anova_pb = f_oneway(*pb_groups)
        
        # Machine learning analysis
        X = self.data[['pa', 'pb']].values
        y = self.data['mean'].values
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf_model, X, y, cv=5)
        rf_model.fit(X, y)
        
        self.results['statistics'] = {
            'correlations': correlation_matrix,
            'anova_pa': anova_pa,
            'anova_pb': anova_pb,
            'rf_cv_score': cv_scores.mean(),
            'rf_feature_importance': dict(zip(['pa', 'pb'], rf_model.feature_importances_))
        }
        
        logger.info("Statistical analysis completed")
        return self.results['statistics']
        
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("Creating visualizations")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Parameter space heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Heatmap
        pivot_data = self.data.pivot_table(values='mean', index='pb', columns='pa', aggfunc='mean')
        sns.heatmap(pivot_data, ax=axes[0,0], cmap='viridis', cbar_kws={'label': 'Mean Response'})
        axes[0,0].set_title('Response Surface Heatmap')
        
        # Parameter effects
        self.data.groupby('pa')['mean'].mean().plot(ax=axes[0,1], marker='o')
        axes[0,1].set_title('Effect of Parameter A')
        axes[0,1].set_ylabel('Mean Response')
        
        # Variance analysis
        axes[1,0].scatter(self.data['mean'], self.data['stdev'], alpha=0.6, c=self.data['pa'], cmap='viridis')
        axes[1,0].set_xlabel('Mean')
        axes[1,0].set_ylabel('Standard Deviation')
        axes[1,0].set_title('Mean-Variance Relationship')
        
        # Distribution
        self.data['mean'].hist(bins=30, ax=axes[1,1], alpha=0.7)
        axes[1,1].set_title('Distribution of Responses')
        axes[1,1].set_xlabel('Mean Response')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive 3D surface plot
        fig_3d = go.Figure()
        
        # Create surface
        pa_unique = sorted(self.data['pa'].unique())
        pb_unique = sorted(self.data['pb'].unique())
        
        z_surface = np.zeros((len(pb_unique), len(pa_unique)))
        for i, pb_val in enumerate(pb_unique):
            for j, pa_val in enumerate(pa_unique):
                subset = self.data[(self.data['pa'] == pa_val) & (self.data['pb'] == pb_val)]
                if not subset.empty:
                    z_surface[i, j] = subset['mean'].mean()
                else:
                    z_surface[i, j] = np.nan
        
        fig_3d.add_trace(go.Surface(x=pa_unique, y=pb_unique, z=z_surface, colorscale='Viridis'))
        fig_3d.update_layout(
            title='Interactive Response Surface',
            scene=dict(
                xaxis_title='Parameter A',
                yaxis_title='Parameter B', 
                zaxis_title='Mean Response'
            ),
            width=800, height=600
        )
        
        fig_3d.write_html(self.output_dir / 'interactive_surface.html')
        
        logger.info("Visualizations created")
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report")
        
        report_content = f"""
# NetLogo Simulation Analysis Report

## Summary Statistics

- **Total Simulations**: {len(self.data):,}
- **Parameter Combinations**: {len(self.data.groupby(['pa', 'pb'])):,}
- **Parameter A Range**: {self.data['pa'].min():.3f} to {self.data['pa'].max():.3f}
- **Parameter B Range**: {self.data['pb'].min():.3f} to {self.data['pb'].max():.3f}
- **Response Range**: {self.data['mean'].min():.3f} to {self.data['mean'].max():.3f}

## Key Findings

### Parameter Effects
- **Parameter A ANOVA**: F = {self.results['statistics']['anova_pa'].statistic:.3f}, p = {self.results['statistics']['anova_pa'].pvalue:.4f}
- **Parameter B ANOVA**: F = {self.results['statistics']['anova_pb'].statistic:.3f}, p = {self.results['statistics']['anova_pb'].pvalue:.4f}

### Machine Learning Analysis
- **Random Forest CV Score**: {self.results['statistics']['rf_cv_score']:.3f}
- **Feature Importance**: 
  - Parameter A: {self.results['statistics']['rf_feature_importance']['pa']:.3f}
  - Parameter B: {self.results['statistics']['rf_feature_importance']['pb']:.3f}

### Optimal Parameter Settings
"""
        
        # Find optimal parameters
        max_idx = self.data['mean'].idxmax()
        optimal_pa = self.data.loc[max_idx, 'pa']
        optimal_pb = self.data.loc[max_idx, 'pb'] 
        optimal_response = self.data.loc[max_idx, 'mean']
        
        report_content += f"""
- **Optimal Parameter A**: {optimal_pa:.3f}
- **Optimal Parameter B**: {optimal_pb:.3f}
- **Maximum Response**: {optimal_response:.3f}

## Correlation Matrix

"""
        
        # Add correlation matrix
        corr_matrix = self.results['statistics']['correlations']
        for i, row_name in enumerate(corr_matrix.index):
            report_content += f"**{row_name}**: "
            for j, col_name in enumerate(corr_matrix.columns):
                if i <= j:  # Upper triangle only
                    report_content += f"{col_name}: {corr_matrix.iloc[i,j]:.3f}, "
            report_content += "\n"
        
        report_content += """

## Files Generated

- `analysis_overview.png`: Comprehensive analysis plots
- `interactive_surface.html`: 3D interactive response surface
- `simulation_data.csv`: Processed simulation data
- `analysis_results.json`: Statistical analysis results

## Methodology

- **Simulation Engine**: NetLogo Parallel Simulation System
- **Analysis**: Python with pandas, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
"""
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write(report_content)
            
        # Save processed data and results
        self.data.to_csv(self.output_dir / 'simulation_data.csv', index=False)
        
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for key, value in self.results['statistics'].items():
                if hasattr(value, 'tolist'):
                    serializable_results[key] = value.tolist()
                elif hasattr(value, 'to_dict'):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = str(value)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis report generated in {self.output_dir}")
        
    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline"""
        logger.info("Starting complete analysis pipeline")
        
        try:
            # Run simulation
            self.run_simulation()
            
            # Load and analyze data
            self.load_data()
            self.statistical_analysis()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            self.generate_report()
            
            logger.info("Pipeline completed successfully")
            return self.output_dir
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    # Create and run pipeline
    pipeline = NetLogoAnalysisPipeline("examples/03_advanced_high_resolution.yaml")
    output_directory = pipeline.run_complete_pipeline()
    print(f"Analysis complete! Results in: {output_directory}")
```

### 3. High-Performance Computing Integration
```bash
#!/bin/bash
# hpc_cluster_workflow.sh
# SLURM job script for running NetLogo simulations on HPC cluster

#SBATCH --job-name=netlogo_sweep
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=netlogo_%j.out
#SBATCH --error=netlogo_%j.err

# Environment setup
module load gcc/9.3.0
module load java/11
module load python/3.8

# Create working directory
WORK_DIR="/scratch/$USER/netlogo_$SLURM_JOB_ID"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy files
cp $SLURM_SUBMIT_DIR/nlproto_native .
cp $SLURM_SUBMIT_DIR/config.yaml .
cp -r $SLURM_SUBMIT_DIR/models .

# Split large parameter space across nodes
python3 << EOF
import yaml
import numpy as np

# Load original config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Calculate total parameter combinations
total_combinations = 1
for param in config['parameters']:
    if 'min' in param['spec']:
        n_values = int((param['spec']['max'] - param['spec']['min']) / param['spec']['step']) + 1
    elif 'values' in param['spec']:
        n_values = len(param['spec']['values'])
    else:  # logarithmic
        n_values = param['spec']['n']
    total_combinations *= n_values

print(f"Total combinations: {total_combinations}")

# Split across nodes
combinations_per_node = total_combinations // $SLURM_NNODES

# Create node-specific configs
for node in range($SLURM_NNODES):
    node_config = config.copy()
    # Implement splitting logic here based on your parameter structure
    # This is a simplified example
    
    with open(f'config_node_{node}.yaml', 'w') as f:
        yaml.dump(node_config, f)
EOF

# Run simulations in parallel across nodes
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    node_id=$SLURM_PROCID
    echo "Node $node_id starting simulation..."
    ./nlproto_native config_node_${node_id}.yaml
    mv results.csv.zst results_node_${node_id}.csv.zst
    mv results.json.zst results_node_${node_id}.json.zst
'

# Combine results
echo "Combining results from all nodes..."
python3 << EOF
import pandas as pd
import json
import zstandard as zstd
import io

# Combine CSV results
combined_data = []
for node in range($SLURM_NNODES):
    filename = f'results_node_{node}.csv.zst'
    with open(filename, 'rb') as f:
        decompressed = zstd.decompress(f.read())
        df = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))
        combined_data.append(df)

# Concatenate all data
final_data = pd.concat(combined_data, ignore_index=True)

# Save combined results
final_data.to_csv('combined_results.csv', index=False)

# Compress final results
import subprocess
subprocess.run(['zstd', 'combined_results.csv'])

print(f"Combined {len(final_data)} simulation results")
EOF

# Copy results back
cp combined_results.csv.zst $SLURM_SUBMIT_DIR/
cp *.out $SLURM_SUBMIT_DIR/
cp *.err $SLURM_SUBMIT_DIR/

echo "HPC workflow completed"
```

### 4. Docker Containerization
```dockerfile
# Dockerfile.research
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    libyaml-cpp-dev \
    libzstd-dev \
    libhwloc-dev \
    openjdk-11-jdk \
    python3 \
    python3-pip \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install pandas numpy matplotlib seaborn plotly scipy scikit-learn zstandard pyyaml

# Install R packages
RUN R -e "install.packages(c('tidyverse', 'ggplot2', 'viridis', 'plotly', 'knitr', 'rmarkdown'), repos='https://cran.rstudio.com/')"

# Create working directory
WORKDIR /netlogo

# Copy application
COPY nlproto_native .
COPY *.cpp *.hpp .
COPY examples/ examples/
COPY models/ models/

# Make executable
RUN chmod +x nlproto_native

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "simulate" ]; then\n\
    ./nlproto_native "$2"\n\
elif [ "$1" = "analyze" ]; then\n\
    python3 examples/python_analysis.py "$2"\n\
elif [ "$1" = "pipeline" ]; then\n\
    ./nlproto_native "$2" && python3 examples/python_analysis.py "$2"\n\
else\n\
    echo "Usage: docker run netlogo-research [simulate|analyze|pipeline] config.yaml"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

```bash
# docker_workflow.sh
# Build and run NetLogo research container

# Build container
docker build -f Dockerfile.research -t netlogo-research .

# Run simulation only
docker run --rm -v $(pwd)/examples:/netlogo/examples \
    netlogo-research simulate examples/03_advanced_high_resolution.yaml

# Run complete pipeline
docker run --rm -v $(pwd)/examples:/netlogo/examples \
    -v $(pwd)/output:/netlogo/output \
    netlogo-research pipeline examples/03_advanced_high_resolution.yaml

# Interactive analysis session
docker run -it --rm -v $(pwd):/netlogo \
    -p 8888:8888 \
    netlogo-research bash
```

### 5. Continuous Integration Pipeline
```yaml
# .github/workflows/netlogo-research.yml
name: NetLogo Research Pipeline

on:
  push:
    paths:
      - 'examples/*.yaml'
      - 'src/**'
  pull_request:
    paths:
      - 'examples/*.yaml'
      - 'src/**'

jobs:
  test-simulation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libyaml-cpp-dev libzstd-dev libhwloc-dev openjdk-11-jdk
    
    - name: Build simulation engine
      run: |
        g++ -std=c++20 -O3 -DNDEBUG \
            -I/usr/local/include -L/usr/local/lib \
            -ljni -lhwloc -lzstd -lyaml-cpp \
            main.cpp benchmark.cpp csv_json_writer.cpp config.cpp \
            -o nlproto_native
    
    - name: Test basic functionality
      run: |
        ./nlproto_native examples/01_beginner_basic_sweep.yaml
        test -f results.csv.zst
        test -f results.json.zst
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: simulation-results
        path: results.*

  analyze-results:
    needs: test-simulation
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/download-artifact@v3
      with:
        name: simulation-results
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Install Python dependencies
      run: |
        pip install pandas numpy matplotlib seaborn scipy scikit-learn zstandard
    
    - name: Run analysis
      run: |
        python examples/automated_analysis.py
    
    - name: Upload analysis results
      uses: actions/upload-artifact@v3
      with:
        name: analysis-results
        path: analysis_output/
```

This comprehensive integration guide demonstrates how to embed the NetLogo parallel simulation system into various research and development workflows, from individual analysis pipelines to large-scale HPC deployments and automated CI/CD systems.