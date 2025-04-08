```markdown
# FeedForward Neural Network

This project implements a FeedForward Neural Network (FFNN) from scratch for regression and classification tasks, with a focus on interpretability and performance tuning. The workflow includes data preprocessing, model construction, hyperparameter tuning, and interpretability through Shapley values.

## 📁 Project Structure

```
FeedForward-Neural-Network/ ├── Data/ # Data preprocessing scripts (R) │ └── Getdata.R
├── Hyperparameter_Tuning/ # Parameter optimization │ └── tuning.py
├── Models/ # FFNN model definition and training │ └── FFNN.py
├── Predict/ # Model interpretation and predictions │ └── ShapleyValues.py
├── Paper.pdf # Final thesis/report └── README.md # Project overview and instructions
```

## 🚀 Features

- Custom FFNN implementation using PyTorch
- Data wrangling in R
- Hyperparameter tuning via grid/random search
- Shapley values for feature importance
- Clear separation of concerns across folders

## 🧠 Technologies Used

- Python (PyTorch, NumPy, Scikit-learn)
- R (data import/cleaning)
- SHAP (Shapley Additive Explanations)

## 📄 How to Use

1. **Prepare the data**  
   Run `Getdata.R` to generate the dataset required by the model.

2. **Tune the model**  
   Navigate to `Hyperparameter_Tuning/` and run:
   ```bash
   python tuning.py
   ```

3. **Train the model**  
   In `Models/`, train the neural network using:
   ```bash
   python FFNN.py
   ```

4. **Interpret results**  
   Run Shapley value computation:
   ```bash
   python ShapleyValues.py
   ```

## 📘 Paper

A detailed explanation of the methodology, results, and evaluation is included in [Paper.pdf](./Paper.pdf).

## 🔍 Future Improvements

- Add cross-validation functionality
- Automate the entire pipeline with a CLI
- Expand to multi-class classification

## 📬 Contact

For questions or collaborations, feel free to reach out via [GitHub profile](https://github.com/Ocalak).

