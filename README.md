

# FeedForward Neural Network with SHAP Analysis

This repository contains the implementation of a Feed-Forward Neural Network with model interpretability using Shapley values, developed as part of my Master's thesis at Technische Universität Dortmund.

## 📂 Repository Structure

```
FeedForward-Neural-Network/
├── Data/                           # Data preprocessing scripts (R)
│   └── Getdata.R
├── Hyperparameter_Tuning/         # Parameter optimization
│   └── tuning.py
├── Models/                        # FFNN model definition and training
│   └── FFNN.py
├── Predict/                       # Model interpretation and predictions
│   └── ShapleyValues.py
├── Paper.pdf                      # Final thesis/report
└── README.md                      # Project overview and instructions
```

## 🧠 Project Overview

This project implements a Feed-Forward Neural Network (FFNN) to model and forecast short-term electricity load in France, focusing on the impact of temperature. It incorporates SHAP (SHapley Additive exPlanations) values for model interpretability, helping to explain individual predictions and the influence of input variables on forecasting outcomes.

## 🔧 Technologies Used

- Python 3.8+
- TensorFlow (Keras)
- R for data preprocessing
- SHAP for model interpretation
- Optuna for hyperparameter tuning
- Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- R 4.0+ (for data preprocessing)
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/FeedForward-Neural-Network.git
   cd FeedForward-Neural-Network
   ```

2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install required R packages:
   ```R
   install.packages(c("tidyverse", "data.table", "caret"))
   ```

## 📊 Data Processing

The script `Data/Getdata.R`:
- Collects electricity load data from ENTSOE (2019–2024)
- Collects temperature data from DWD for France’s major cities
- Handles daylight saving time changes, interpolates missing hours
- Scales data using RobustScaler to reduce outlier impact

## 🔍 Model Training

The FFNN model (`Models/FFNN.py`) includes:
- Input layer with 24+ engineered features
- Two hidden layers with layer normalization
- ReLU/ELU activation functions
- Dropout and L1 regularization
- MAE as loss function; Adam optimizer
- Forecasting horizon: 192 hours (8 days)

## ⚙️ Hyperparameter Tuning

Using `Hyperparameter_Tuning/tuning.py`:
- Optuna for Bayesian optimization
- Tuned: number of neurons, activation functions, dropout rate, regularization, learning rate, etc.

## 🔮 Model Interpretation

`Predict/ShapleyValues.py` uses SHAP to:
- Analyze feature contributions to individual forecasts
- Visualize feature importance (e.g. temperature, calendar variables, lagged load)
- Explore interactions over time (e.g. seasonal effects)

## 📝 Results

- Weather-dependent FFNN models outperform weather-independent ones
- Temperature and autoregressive features are critical for accuracy
- Average MAPE significantly reduced by incorporating temperature
- SHAP plots confirm feature relevance and seasonal impact

## 📄 Paper

See `Paper.pdf` for:
- Full methodology and theoretical background
- Detailed analysis and figures
- SHAP visualizations across months
- Conclusions and future research directions

## 👨‍💻 Author

Öcal Kaptan – Technische Universität Dortmund, Faculty of Statistics  
Supervised by: Prof. Dr. Florian Ziel, Prof. Dr. Christoph Hanck

## 📚 Citation

If you use this work, please cite it as:

```bibtex
@mastersthesis{Kaptan2024,
  author = {Öcal Kaptan},
  title = {Analyzing Weather Effects in Short-Term Load Forecasting Using Feed-Forward Neural Networks},
  school = {Technische Universität Dortmund},
  year = {2024},
  address = {Dortmund, Germany},
  month = {October}
}
```

