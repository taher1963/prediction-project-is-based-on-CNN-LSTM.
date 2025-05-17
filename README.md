# Gold Stock Price Prediction using CNN-LSTM

![Gold](gold.jpg)

## Overview

This Jupyter notebook demonstrates how to use a hybrid CNN–LSTM model in PyTorch to forecast gold prices. It includes data loading, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Dataset

- **Source:** [Gold Stock Prices (Kaggle)](https://www.kaggle.com/datasets/sahilwagh/gold-stock-prices)
- **File:** `Gold.csv`
- **Description:** Daily gold price data, with columns:
  - `Date` (YYYY-MM-DD)
  - `Price` (closing price in USD)

## Prerequisites

- Python 3.7+
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Installation

1. Clone this repository (or download the notebook file).
2. Install required Python packages:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
```

3.	Download the Gold.csv dataset from Kaggle and place it in ../input/gold-stock-prices/ (or update the path in the notebook).


## Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open gold_stock_cnn_lstm.ipynb.
3. Run each cell in order. Ensure the dataset path is correct.
4. Inspect the plots and model outputs.

## Notebook Structure

1. Import Libraries

Load essential Python packages for data handling, visualization, and model building.

2. Load and Inspect Data

Read the CSV into a DataFrame, set the Date as index, and preview basic statistics.

3. Exploratory Data Analysis (EDA)
    - Summary statistics of gold prices
    - Time-series plot of price movement

4. Feature Engineering
    - Create 10-day and 20-day moving averages (MA10, MA20)
    - Drop initial NaN values

5. Data Preprocessing
    - Scale features and target using MinMaxScaler
    - Define a function to create sliding-window sequences

6. Prepare DataLoaders
    - Split data into train/test sets
    - Wrap into PyTorch TensorDataset and DataLoader

7. Build CNN-LSTM Model
    - 1D convolution layer to extract local patterns
    - LSTM layer for temporal modeling
    - Fully connected output

8. Train the Model
    - Use MSE loss and Adam optimizer
    - Run for 50 epochs, printing loss every 10 epochs

9. Evaluate the Model
    - Generate predictions on test set
    - Inverse-transform to original price scale
    - Plot Actual vs. Predicted prices
    - Compute test MSE

10. Conclusions

Summary of findings and suggestions for further improvement.

## Results
- The plot of actual vs. predicted gold prices illustrates the model’s performance.
- Final Test MSE (on original scale) is displayed in the notebook.

## Next Steps
- Hyperparameter tuning (learning rate, network depth, sequence length)
- Incorporate additional technical indicators (e.g., RSI, MACD)
- Experiment with alternative architectures (e.g., Transformer-based models)
