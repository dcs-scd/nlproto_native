# Output Data Analysis Guide

## Understanding the Output Formats

The nlproto_native system produces results in two compressed formats optimized for different use cases:

### 1. CSV Format (results.csv.zst)
- **Best for**: Statistical analysis, spreadsheet import, R/Python data science
- **Structure**: Tabular data with parameter columns and metric columns
- **Compression**: Zstandard for fast decompression and small file size

### 2. JSON Format (results.json.zst)
- **Best for**: Web applications, metadata preservation, hierarchical analysis
- **Structure**: Nested objects with full experiment metadata
- **Compression**: Zstandard for efficient storage

## Basic Output Exploration

### Decompress and Examine Files
```bash
# Decompress CSV results
zstd -d results.csv.zst

# Quick look at structure
head -10 results.csv
tail -5 results.csv
wc -l results.csv

# Decompress JSON results
zstd -d results.json.zst

# Pretty-print JSON structure
jq '.' results.json | head -50

# Get summary information
jq '.model, .ticks, (.data | length)' results.json
```

### Sample Output Structure

#### CSV Format Example
```csv
run,pa,pb,mean,stdev,entropy
0,10.0,0.1,25.847,4.231,0.892
1,10.0,0.2,23.156,3.847,0.751
2,10.0,0.3,21.394,4.102,0.823
3,20.0,0.1,45.231,6.781,1.234
4,20.0,0.2,42.875,5.923,1.156
...
```

#### JSON Format Example
```json
{
  "model": "models/sample.nlogo",
  "ticks": 200,
  "data": [
    {
      "run": 0,
      "pa": 10.0,
      "pb": 0.1,
      "metrics": {
        "mean": 25.847,
        "stdev": 4.231,
        "entropy": 0.892
      }
    },
    {
      "run": 1,
      "pa": 10.0,
      "pb": 0.2,
      "metrics": {
        "mean": 23.156,
        "stdev": 3.847,
        "entropy": 0.751
      }
    }
  ]
}
```

## Analysis with R

### Basic R Analysis Script
```r
# load_and_analyze.R
library(tidyverse)
library(ggplot2)

# Load CSV data
data <- read_csv("results.csv")

# Basic summary statistics
summary(data)

# Parameter space coverage
cat("Parameter combinations tested:", nrow(data), "\n")
cat("Unique pa values:", length(unique(data$pa)), "\n")
cat("Unique pb values:", length(unique(data$pb)), "\n")

# Basic visualizations
# 1. Parameter relationship heatmap
ggplot(data, aes(x = pa, y = pb, fill = mean)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Mean Response Across Parameter Space",
       x = "Parameter A", y = "Parameter B") +
  theme_minimal()

ggsave("parameter_heatmap.png", width = 8, height = 6)

# 2. Response surface
ggplot(data, aes(x = pa, y = mean, color = factor(pb))) +
  geom_line() +
  geom_point() +
  labs(title = "Response Surface by Parameter B",
       x = "Parameter A", y = "Mean Response",
       color = "Parameter B") +
  theme_minimal()

ggsave("response_surface.png", width = 10, height = 6)

# 3. Variance analysis
ggplot(data, aes(x = mean, y = stdev)) +
  geom_point(aes(color = pa), alpha = 0.7) +
  geom_smooth(method = "lm") +
  labs(title = "Mean-Variance Relationship",
       x = "Mean Response", y = "Standard Deviation",
       color = "Parameter A") +
  theme_minimal()

ggsave("variance_analysis.png", width = 8, height = 6)

# Statistical analysis
# 1. Correlation matrix
cor_matrix <- cor(data[, c("pa", "pb", "mean", "stdev", "entropy")])
print("Correlation Matrix:")
print(round(cor_matrix, 3))

# 2. Linear model
model <- lm(mean ~ pa + pb + I(pa^2) + I(pb^2) + pa:pb, data = data)
summary(model)

# 3. ANOVA
anova_result <- aov(mean ~ factor(pa) * factor(pb), data = data)
summary(anova_result)
```

### Advanced R Analysis
```r
# advanced_analysis.R
library(tidyverse)
library(viridis)
library(plotly)
library(mgcv)

data <- read_csv("results.csv")

# 1. GAM (Generalized Additive Model) for smooth surfaces
gam_model <- gam(mean ~ s(pa) + s(pb) + s(pa, pb), data = data)
summary(gam_model)

# Create prediction grid
pa_seq <- seq(min(data$pa), max(data$pa), length.out = 50)
pb_seq <- seq(min(data$pb), max(data$pb), length.out = 50)
pred_grid <- expand_grid(pa = pa_seq, pb = pb_seq)

# Generate predictions
pred_grid$predicted <- predict(gam_model, pred_grid)

# 3D surface plot
p <- plot_ly(pred_grid, x = ~pa, y = ~pb, z = ~predicted,
             type = "surface",
             colorscale = "Viridis") %>%
  layout(title = "Predicted Response Surface",
         scene = list(
           xaxis = list(title = "Parameter A"),
           yaxis = list(title = "Parameter B"),
           zaxis = list(title = "Mean Response")
         ))

htmlwidgets::saveWidget(p, "response_surface_3d.html")

# 2. Sensitivity analysis
sensitivity_analysis <- function(data, param_col, response_col) {
  # Calculate local sensitivity (partial derivatives)
  data %>%
    arrange(!!sym(param_col)) %>%
    mutate(
      sensitivity = (lead(!!sym(response_col)) - lag(!!sym(response_col))) /
                   (lead(!!sym(param_col)) - lag(!!sym(param_col)))
    ) %>%
    filter(!is.na(sensitivity))
}

sens_pa <- sensitivity_analysis(data, "pa", "mean")
sens_pb <- sensitivity_analysis(data, "pb", "mean")

# Plot sensitivity
ggplot(sens_pa, aes(x = pa, y = sensitivity)) +
  geom_point() +
  geom_smooth() +
  labs(title = "Sensitivity to Parameter A",
       x = "Parameter A", y = "Local Sensitivity") +
  theme_minimal()

ggsave("sensitivity_pa.png", width = 8, height = 6)
```

## Analysis with Python

### Basic Python Analysis
```python
# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import zstandard as zstd

# Load compressed CSV data
def load_compressed_csv(filename):
    with open(filename, 'rb') as f:
        decompressed = zstd.decompress(f.read())
        return pd.read_csv(io.StringIO(decompressed.decode('utf-8')))

# Load compressed JSON data
def load_compressed_json(filename):
    with open(filename, 'rb') as f:
        decompressed = zstd.decompress(f.read())
        return json.loads(decompressed.decode('utf-8'))

# Load data
df = load_compressed_csv('results.csv.zst')
json_data = load_compressed_json('results.json.zst')

print(f"Loaded {len(df)} simulation results")
print(f"Model: {json_data['model']}")
print(f"Ticks: {json_data['ticks']}")

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Parameter space analysis
print(f"\nParameter Space:")
print(f"Parameter A range: {df['pa'].min()} to {df['pa'].max()}")
print(f"Parameter B range: {df['pb'].min()} to {df['pb'].max()}")
print(f"Unique combinations: {len(df)}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Heatmap
pivot_data = df.pivot(index='pb', columns='pa', values='mean')
sns.heatmap(pivot_data, ax=axes[0,0], cmap='viridis')
axes[0,0].set_title('Mean Response Heatmap')

# 2. Parameter effects
df.groupby('pa')['mean'].mean().plot(ax=axes[0,1])
axes[0,1].set_title('Effect of Parameter A')
axes[0,1].set_xlabel('Parameter A')
axes[0,1].set_ylabel('Mean Response')

# 3. Variance vs Mean
axes[1,0].scatter(df['mean'], df['stdev'], alpha=0.6)
axes[1,0].set_xlabel('Mean')
axes[1,0].set_ylabel('Standard Deviation')
axes[1,0].set_title('Mean-Variance Relationship')

# 4. Distribution of outcomes
df['mean'].hist(bins=30, ax=axes[1,1])
axes[1,1].set_title('Distribution of Mean Responses')
axes[1,1].set_xlabel('Mean Response')

plt.tight_layout()
plt.savefig('basic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis
correlation_matrix = df[['pa', 'pb', 'mean', 'stdev', 'entropy']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Statistical modeling
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Prepare features
X = df[['pa', 'pb']]
y = df['mean']

# Linear model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Polynomial model
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

print(f"\nModel Performance:")
print(f"Linear R²: {r2_score(y, y_pred_linear):.3f}")
print(f"Polynomial R²: {r2_score(y, y_pred_poly):.3f}")
```

### Advanced Python Analysis
```python
# advanced_python_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data (assuming functions from previous script)
df = load_compressed_csv('results.csv.zst')

# 1. Response surface modeling
def create_response_surface(df, param1, param2, response):
    """Create smooth response surface using interpolation"""
    from scipy.interpolate import griddata
    
    # Create fine grid
    p1_range = np.linspace(df[param1].min(), df[param1].max(), 100)
    p2_range = np.linspace(df[param2].min(), df[param2].max(), 100)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Interpolate
    points = df[[param1, param2]].values
    values = df[response].values
    
    surface = griddata(points, values, (P1, P2), method='cubic')
    
    return P1, P2, surface

# Create 3D surface plot
P1, P2, surface = create_response_surface(df, 'pa', 'pb', 'mean')

fig = go.Figure(data=[go.Surface(x=P1, y=P2, z=surface, colorscale='Viridis')])
fig.add_trace(go.Scatter3d(
    x=df['pa'], y=df['pb'], z=df['mean'],
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Actual Data'
))

fig.update_layout(
    title='Response Surface with Data Points',
    scene=dict(
        xaxis_title='Parameter A',
        yaxis_title='Parameter B',
        zaxis_title='Mean Response'
    )
)

fig.write_html('response_surface_3d.html')

# 2. Optimization analysis
def find_optima(df, response_col):
    """Find parameter combinations giving optimal responses"""
    max_idx = df[response_col].idxmax()
    min_idx = df[response_col].idxmin()
    
    print(f"Maximum {response_col}: {df.loc[max_idx, response_col]:.3f}")
    print(f"  at pa={df.loc[max_idx, 'pa']}, pb={df.loc[max_idx, 'pb']}")
    
    print(f"Minimum {response_col}: {df.loc[min_idx, response_col]:.3f}")
    print(f"  at pa={df.loc[min_idx, 'pa']}, pb={df.loc[min_idx, 'pb']}")
    
    return df.loc[max_idx], df.loc[min_idx]

max_point, min_point = find_optima(df, 'mean')

# 3. Uncertainty quantification
def uncertainty_analysis(df):
    """Analyze parameter uncertainty and confidence intervals"""
    
    # Group by parameter combinations and calculate statistics
    grouped = df.groupby(['pa', 'pb']).agg({
        'mean': ['mean', 'std', 'count'],
        'stdev': ['mean', 'std'],
        'entropy': ['mean', 'std']
    }).round(4)
    
    # Calculate confidence intervals
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    grouped_flat = grouped.copy()
    grouped_flat.columns = ['_'.join(col).strip() for col in grouped_flat.columns]
    
    # Add confidence intervals for mean
    t_critical = stats.t.ppf(1 - alpha/2, grouped_flat['mean_count'] - 1)
    margin_error = t_critical * (grouped_flat['mean_std'] / 
                                np.sqrt(grouped_flat['mean_count']))
    
    grouped_flat['mean_ci_lower'] = grouped_flat['mean_mean'] - margin_error
    grouped_flat['mean_ci_upper'] = grouped_flat['mean_mean'] + margin_error
    
    return grouped_flat

uncertainty_stats = uncertainty_analysis(df)
print("\nUncertainty Analysis (first 10 rows):")
print(uncertainty_stats.head(10))

# 4. Machine learning analysis
def ml_analysis(df):
    """Use machine learning to understand parameter relationships"""
    
    # Features and target
    X = df[['pa', 'pb']]
    y = df['mean']
    
    # Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    
    print(f"\nMachine Learning Analysis:")
    print(f"Random Forest R² (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Feature Importance:")
    for feature, importance in zip(X.columns, rf.feature_importances_):
        print(f"  {feature}: {importance:.3f}")
    
    return rf

ml_model = ml_analysis(df)

# 5. Export processed results
def export_analysis_results(df, uncertainty_stats, filename_prefix):
    """Export analysis results in multiple formats"""
    
    # Summary statistics
    summary = {
        'total_simulations': len(df),
        'parameter_combinations': len(df.groupby(['pa', 'pb'])),
        'mean_response': {
            'overall_mean': df['mean'].mean(),
            'overall_std': df['mean'].std(),
            'min': df['mean'].min(),
            'max': df['mean'].max()
        },
        'optimal_parameters': {
            'max_response': {
                'pa': float(max_point['pa']),
                'pb': float(max_point['pb']),
                'value': float(max_point['mean'])
            },
            'min_response': {
                'pa': float(min_point['pa']),
                'pb': float(min_point['pb']),
                'value': float(min_point['mean'])
            }
        }
    }
    
    # Save summary as JSON
    with open(f'{filename_prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed statistics
    uncertainty_stats.to_csv(f'{filename_prefix}_uncertainty.csv')
    
    print(f"Analysis results exported to {filename_prefix}_*.json/csv")

export_analysis_results(df, uncertainty_stats, 'analysis_results')
```

## Working with Large Result Sets

### Memory-Efficient Processing
```python
# large_data_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
import zstandard as zstd

def process_large_results(csv_file, chunk_size=10000):
    """Process large result files in chunks to manage memory"""
    
    def read_compressed_chunks(filename, chunk_size):
        with open(filename, 'rb') as f:
            decompressed = zstd.decompress(f.read())
            
        # Convert to StringIO and read in chunks
        import io
        string_data = io.StringIO(decompressed.decode('utf-8'))
        
        for chunk in pd.read_csv(string_data, chunksize=chunk_size):
            yield chunk
    
    # Process chunks and accumulate statistics
    total_rows = 0
    sum_stats = {}
    
    for chunk in read_compressed_chunks(csv_file, chunk_size):
        total_rows += len(chunk)
        
        # Accumulate statistics
        for col in chunk.select_dtypes(include=[np.number]).columns:
            if col not in sum_stats:
                sum_stats[col] = {'sum': 0, 'sum_sq': 0, 'min': float('inf'), 'max': float('-inf')}
            
            sum_stats[col]['sum'] += chunk[col].sum()
            sum_stats[col]['sum_sq'] += (chunk[col] ** 2).sum()
            sum_stats[col]['min'] = min(sum_stats[col]['min'], chunk[col].min())
            sum_stats[col]['max'] = max(sum_stats[col]['max'], chunk[col].max())
    
    # Calculate final statistics
    final_stats = {}
    for col, stats in sum_stats.items():
        mean = stats['sum'] / total_rows
        variance = (stats['sum_sq'] / total_rows) - (mean ** 2)
        std = np.sqrt(variance)
        
        final_stats[col] = {
            'mean': mean,
            'std': std,
            'min': stats['min'],
            'max': stats['max']
        }
    
    return final_stats, total_rows

# Usage for large files
if Path('results.csv.zst').stat().st_size > 100_000_000:  # > 100MB
    print("Processing large file in chunks...")
    stats, n_rows = process_large_results('results.csv.zst')
    print(f"Processed {n_rows} rows")
    for col, stat in stats.items():
        print(f"{col}: mean={stat['mean']:.3f}, std={stat['std']:.3f}")
```

## Interactive Analysis Dashboard

### Simple Streamlit Dashboard
```python
# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import zstandard as zstd
import io

# Load data function
@st.cache_data
def load_data():
    with open('results.csv.zst', 'rb') as f:
        decompressed = zstd.decompress(f.read())
        return pd.read_csv(io.StringIO(decompressed.decode('utf-8')))

# Streamlit app
st.title('NetLogo Simulation Results Dashboard')

# Load data
df = load_data()

# Sidebar controls
st.sidebar.header('Analysis Controls')
param_x = st.sidebar.selectbox('X Parameter', ['pa', 'pb'])
param_y = st.sidebar.selectbox('Y Parameter', ['pa', 'pb'] if param_x != 'pb' else ['pa'])
response = st.sidebar.selectbox('Response Variable', ['mean', 'stdev', 'entropy'])

# Filter data
min_val = st.sidebar.slider(f'Min {param_x}', 
                           float(df[param_x].min()), 
                           float(df[param_x].max()), 
                           float(df[param_x].min()))
max_val = st.sidebar.slider(f'Max {param_x}', 
                           float(df[param_x].min()), 
                           float(df[param_x].max()), 
                           float(df[param_x].max()))

filtered_df = df[(df[param_x] >= min_val) & (df[param_x] <= max_val)]

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader('Parameter Space Visualization')
    fig = px.scatter(filtered_df, x=param_x, y=param_y, color=response,
                     title=f'{response} across parameter space')
    st.plotly_chart(fig)

with col2:
    st.subheader('Response Distribution')
    fig = px.histogram(filtered_df, x=response, nbins=30)
    st.plotly_chart(fig)

# Summary statistics
st.subheader('Summary Statistics')
st.write(filtered_df[['pa', 'pb', 'mean', 'stdev', 'entropy']].describe())

# Correlation matrix
st.subheader('Correlation Matrix')
corr_matrix = filtered_df[['pa', 'pb', 'mean', 'stdev', 'entropy']].corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
st.plotly_chart(fig)
```

Run the dashboard with:
```bash
pip install streamlit plotly
streamlit run dashboard.py
```

This comprehensive output analysis guide provides multiple approaches for extracting insights from your NetLogo simulation results, from basic statistical analysis to advanced machine learning techniques and interactive dashboards.