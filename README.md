# Online Learning for Financial Time Series

This project benchmarks online machine learning algorithms for adaptive filtering in financial time-series prediction. It implements linear and kernel-based methods to predict stock prices in a sequential, real-time manner.

## Overview

The system predicts the "Mid-Price" of assets (derived from High/Low or Open/Close) using algorithms that learn incrementally. Unlike batch processing, these models update their weights instantly after every sample, making them suitable for non-stationary financial environments.

## Implemented Models (`models.py`)

The following algorithms are implemented adhering to the `river` library interface:

* **LMS (Least Mean Square):** A standard adaptive filter that minimizes the mean square error using stochastic gradient descent.
* **KLMS (Kernel Least Mean Square):** A non-linear extension of LMS utilizing Gaussian (RBF) kernels to map input vectors into a high-dimensional feature space.
* **KAPA (Kernel Adaptive Parallel Algorithm):** An advanced kernel method designed for efficient online learning with sparsification.

## Features

* **Data Integration:** Fetches historical stock data (e.g., `^NSEI`, `TITAN.NS`) via the `yfinance` API.
* **Feature Engineering:** Dynamic calculation of mid-prices and sliding window inputs.
* **Evaluation Metrics:**
    * **MSE / MAE:** Accuracy measurements.
    * **Directional Symmetry (DS):** Metric to evaluate the correctness of predicted price movement direction.
* **Visualization:** Automated plotting of convergence rates, prediction vs. actual values, and residual histograms.

## Dependencies

* Python 3.8+
* `river`
* `yfinance`
* `numpy`
* `pandas`
* `matplotlib`
* `jupyter`

## Usage

1.  Install the required packages:
    ```bash
    pip install river yfinance numpy pandas matplotlib jupyter
    ```

2.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook test.ipynb
    ```

3.  Execute the cells to download data, initialize models, and run the online learning simulation.

## File Structure

* `models.py`: Contains the `LMS`, `KLMS`, and `KAPA` class definitions.
* `test.ipynb`: Main entry point for data loading, training loops, and results visualization.

## Disclaimer

This code is provided for educational and research purposes only. It is not intended for use in actual financial trading and does not constitute investment advice.
