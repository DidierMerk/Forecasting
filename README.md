<div align="center">

# 🔮 The Future of Forecasting: Foundation Models for Time Series

[![Website](https://img.shields.io/badge/🌐_Website-didiermerk.github.io-blue)](https://didiermerk.github.io)
[![Thesis](https://img.shields.io/badge/📄_Thesis_PDF-MSc_AI_2024-green)](https://didiermerk.github.io/files/msc_didier.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Chronos](https://img.shields.io/badge/🦙_Chronos-AWS-orange)](https://github.com/amazon-science/chronos-forecasting)
[![Nixtla](https://img.shields.io/badge/⚡_Nixtla-Forecasting-purple)](https://github.com/Nixtla)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DidierMerk/Forecasting/blob/main/demo.ipynb)

*Can AI predict the future? Exploring zero-shot forecasting with Large Language Model architectures on financial time series.*

[**Try the Demo**](#-quick-demo) • [**Read the Research**](#-the-research) • [**View Results**](#-key-findings) • [**Explore the Code**](#-repository-structure)

<img src="assets/hero_forecasting.gif" width="800" alt="Forecasting Visualization">

</div>

---

## 📚 Table of Contents

- [What is Time Series Forecasting?](#-what-is-time-series-forecasting)
- [A Brief History](#-a-brief-history-of-forecasting)
- [Quick Demo](#-quick-demo)
- [The Research](#-the-research)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Key Findings](#-key-findings)
- [Models Compared](#-models-compared)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🤔 What is Time Series Forecasting?

Imagine you're looking at your bank account balance over the past year. You see patterns - maybe it dips after rent day, spikes on payday, and fluctuates with your spending habits. **Time series forecasting** is the art and science of using these historical patterns to predict what comes next.

<div align="center">
<img src="assets/forecasting_explained.png" width="700" alt="What is Forecasting">
</div>

In essence, time series forecasting answers the question: *"Given what happened before, what will happen next?"*

### Real-World Applications

- 🌪️ **Weather Prediction**: Hurricane paths that save lives
- 💊 **Healthcare**: Detecting irregular heartbeats before they become dangerous
- 💰 **Finance**: Predicting market trends and managing risk
- ⚡ **Energy**: Forecasting electricity demand to prevent blackouts
- 🛒 **Retail**: Predicting sales to optimize inventory

---

## 📜 A Brief History of Forecasting

<div align="center">
<img src="assets/timeline.png" width="900" alt="Forecasting Timeline">
</div>

From Yule's autoregressive models in 1927 to today's foundation models, forecasting has evolved dramatically:

| Era | Key Development | Impact |
|-----|----------------|---------|
| **1927** | Yule's AR Model | First mathematical approach to forecasting |
| **1970** | Box-Jenkins ARIMA | Statistical forecasting goes mainstream |
| **1997** | LSTM Networks | Deep learning enters time series |
| **2017** | Transformer Architecture | Attention mechanisms revolutionize sequence modeling |
| **2024** | Foundation Models (Chronos, TimeGPT) | Zero-shot forecasting becomes reality |

---

## 🚀 Quick Demo

Experience the power of foundation models in just a few lines of code! 

### Run in Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DidierMerk/Forecasting/blob/main/demo.ipynb)

### Or Run Locally:
```python
# Install required libraries
!pip install git+https://github.com/amazon-science/chronos-forecasting.git
!pip install neuralforecast statsforecast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# Create sample financial time series data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
trend = np.linspace(100, 120, 365)
seasonal = 10 * np.sin(np.arange(365) * 2 * np.pi / 30)  # Monthly pattern
noise = np.random.normal(0, 5, 365)
values = trend + seasonal + noise

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Load Chronos (Foundation Model)
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Make predictions
forecast_horizon = 30
past_values = torch.tensor(df['value'].values[:-forecast_horizon])
predictions = pipeline.predict(
    past_values.unsqueeze(0),
    prediction_length=forecast_horizon,
    num_samples=100
)

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(df['date'][:-forecast_horizon], df['value'][:-forecast_horizon], 
         label='Historical Data', color='blue')
plt.plot(df['date'][-forecast_horizon:], df['value'][-forecast_horizon:], 
         label='Actual Future', color='green', linestyle='--')

# Plot predictions with confidence intervals
forecast_dates = df['date'][-forecast_horizon:]
median_forecast = predictions.median(dim=1).values.squeeze()
lower_bound = predictions.quantile(0.1, dim=1).values.squeeze()
upper_bound = predictions.quantile(0.9, dim=1).values.squeeze()

plt.plot(forecast_dates, median_forecast, label='Chronos Forecast', color='red')
plt.fill_between(forecast_dates, lower_bound, upper_bound, alpha=0.3, color='red')

plt.legend()
plt.title('Zero-Shot Forecasting with Chronos Foundation Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

### Expected Output:
<div align="center">
<img src="assets/demo_output.png" width="800" alt="Demo Output">
</div>

---

## 📊 The Research

This repository contains the complete implementation of my Master's thesis: **"Rethinking Models and Evaluations for Financial Time Series Forecasting"** (University of Amsterdam, 2024).

### Research Question
> *"To what extent can large language model architectures be applied to financial time series forecasting, in comparison to traditional statistical and deep learning models?"*

### Key Contributions

1. **First comprehensive evaluation** of Chronos (AWS's foundation model) on private financial transaction data
2. **Zero-shot performance analysis** comparing foundation models against 6 state-of-the-art baselines
3. **Confidence interval reliability study** revealing overconfidence in foundation model predictions
4. **Seasonality and entropy analysis** showing model performance across different time series characteristics

### Dataset
- **Source**: ING Bank transaction data (largest bank in the Netherlands)
- **Scale**: 278 aggregated account balance time series
- **Duration**: January 2022 - October 2024 (1,014 daily observations)
- **Privacy**: Aggregated and anonymized at ultimate parent level

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/DidierMerk/Forecasting.git
cd Forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended for Chronos)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Repository Structure
```
Forecasting/
│
├── 📊 notebooks/           # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_statistical_models.ipynb
│   ├── 03_deep_learning_models.ipynb
│   ├── 04_chronos_evaluation.ipynb
│   └── 05_results_visualization.ipynb
│
├── 📈 src/                 # Source code
│   ├── models/            # Model implementations
│   ├── evaluation/        # Evaluation metrics
│   └── utils/            # Helper functions
│
├── 🎨 assets/             # Visualizations and figures
│
├── 📄 thesis/             # LaTeX source files
│
├── 🚀 demo.ipynb          # Interactive demo notebook
│
└── 📋 requirements.txt    # Package dependencies
```

---

## 🎯 Key Findings

<div align="center">
<img src="assets/results_comparison.png" width="900" alt="Model Comparison">
</div>

### 1. Zero-Shot Excellence
- **Chronos-Large achieved 18.35% MAPE** on 30-day forecasts without any training on the financial data
- Outperformed deep learning models specifically trained on the dataset

### 2. The Confidence Paradox
- Foundation models showed **systematic overconfidence** in predictions
- Only ~50% of actual values fell within 90% confidence intervals (expected: 90%)

### 3. Seasonality Matters
- Models performed **40% better** on series with clear seasonal patterns
- Statistical models excelled on non-seasonal data

### 4. Model Performance by Horizon

| Model | 1-Day | 7-Day | 14-Day | 30-Day |
|-------|-------|-------|--------|--------|
| **Chronos-Large** | **3.23%** | **12.19%** | **17.32%** | **18.35%** |
| PatchTST | 5.09% | 13.14% | 17.78% | 18.92% |
| AutoARIMA | 9.21% | 18.25% | 23.95% | 24.88% |
| Naive | 3.58% | 13.46% | 19.86% | 21.64% |

*Values shown are MAPE (Mean Absolute Percentage Error) - lower is better*

---

## 🤖 Models Compared

<div align="center">
<img src="assets/model_architectures.png" width="900" alt="Model Architectures">
</div>

### Statistical Baselines
- **AutoARIMA** - Automatic parameter selection for ARIMA models
- **AutoETS** - Exponential smoothing with automatic model selection
- **Naive** - Last value persistence baseline

### Deep Learning Models
- **PatchTST** - Transformer-based model using time series patches
- **NHITS** - Neural Hierarchical Interpolation
- **DeepAR** - Probabilistic RNN-based forecaster
- **TimesNet** - CNN-based temporal modeling

### Foundation Models
- **Chronos-Small** (46M parameters) - Efficient variant
- **Chronos-Large** (710M parameters) - High-capacity variant
- **Chronos-FT** - Fine-tuned on financial data

---

## 🔮 Future Work

1. **Ensemble Methods**: Combining foundation models with traditional approaches
2. **Calibration Techniques**: Improving confidence interval reliability
3. **Multi-variate Extension**: Extending to multiple related time series
4. **Real-time Deployment**: Production-ready implementations
5. **Interpretability**: Understanding what patterns foundation models learn

---

## 📝 Citation

If you use this code or find our research helpful, please cite:
```bibtex
@mastersthesis{merk2024rethinking,
  title={Rethinking Models and Evaluations for Financial Time Series Forecasting},
  author={Merk, Didier},
  year={2024},
  school={University of Amsterdam},
  type={MSc Thesis in Artificial Intelligence}
}
```

---

## 🙏 Acknowledgments

Special thanks to:
- **[Nixtla](https://github.com/Nixtla)** for their excellent `neuralforecast` and `statsforecast` libraries
- **[AWS AI Labs](https://github.com/amazon-science/chronos-forecasting)** for open-sourcing Chronos
- **ING Bank** WBAA team for data access and computational resources
- **Yongtuo Liu** (Supervisor) and **Prof. Dr. Efstratios Gavves** (Examiner) at UvA

---

<div align="center">

**[⬆ Back to Top](#-the-future-of-forecasting-foundation-models-for-time-series)**

Made with ❤️ for the forecasting community

</div>
