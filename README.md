MT24/
├── README.md              # Comprehensive project documentation
├── requirements.txt       # Dependencies list
├── setup.py               # Package installation script
├── data/
│   ├── raw/               # Original unprocessed data
│   └── processed/         # Preprocessed data ready for modeling
├── notebooks/             # Jupyter notebooks for exploration and results
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_shapley_analysis.ipynb
├── src/                   # Source code as a proper package
│   ├── __init__.py
│   ├── data/              # Data processing scripts
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── models/            # Model definition and training code
│   │   ├── __init__.py
│   │   ├── fnn.py
│   │   └── train_model.py
│   ├── visualization/     # Visualization utilities
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── explanation/       # Shapley values calculation and interpretation
│       ├── __init__.py
│       └── shapley.py
├── tests/                 # Unit tests for your code
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── models/                # Saved model files
│   └── fnn_model.pkl
└── reports/               # Generated analysis reports and figures
    └── figures/
