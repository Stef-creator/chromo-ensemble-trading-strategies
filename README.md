
# chromo-ensemble-trading-strategies - Ensemble Trading Strategy Pipeline

## Overview

This repository contains a full systematic trading pipeline integrating:

- Genetic Algorithm optimization for technical indicator parameter selection  
- Machine learning models: MLP, CNN, LSTM, Transformer  
- Ensemble integration methods: Weighted voting, probabilistic voting, Kelly sizing  
- Advanced backtesting with comprehensive trading performance metrics

The pipeline is designed for researchers, quantitative analysts, and algorithmic traders seeking to build robust data-driven trading strategies combining technical analysis and machine learning.

---

## Features

- Genetic Algorithm (GA)  
  - Optimizes RSI thresholds and intervals for buy/sell signals  
  - Evaluates fitness based on risk-adjusted returns (annualized return / max drawdown)

- Machine Learning Models  
  - MLP, CNN, LSTM, and Transformer architectures  
  - Hyperparameter tuning and final retraining pipelines

- Ensemble Methods  
  - Weighted majority voting  
  - Probabilistic voting (softmax averaging)  
  - Kelly criterion dynamic position sizing

- Backtesting & Evaluation  
  - Total return, annualized return, Sharpe Ratio, Sortino Ratio, Calmar Ratio  
  - Omega Ratio, Gain-to-Pain Ratio, Win Rate, Profit Factor, drawdown statistics

- Research Notebook  
  - Fully structured Jupyter notebook for replication and presentation  
  - Markdown explanations with mathematical derivations

---

## Project Structure


- data/                     # CSV datasets, tuning results, saved parameters
- models/                   # Model architecture scripts for MLP, CNN, LSTM, Transformer
- scripts/                  # Python scripts for training, evaluation, and backtesting
- optimized_models/         # Trained model weight files (.pth)
- plots/                    # Generated figures and result plots
- utils.py                  # Utility functions for backtesting, metrics, plotting
- notebook.ipynb            # Main project notebook
- requirements.txt          # Python package dependencies
- LICENSE                   # Project license
- .gitignore                # Git ignore file
- README.md                 # Project documentation


---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Stef-creator/chromo-ensemble-trading-strategies.git
cd chromo-ensemble-trading-strategies
pip install -r requirements.txt
```

## Usage

Run the main notebook for:

1. **Data preparation and technical indicator generation**
2. **Genetic Algorithm parameter optimization**
3. **Model training, tuning, and final retraining**
4. **Ensemble strategy backtesting and performance evaluation**
5. **Result visualization and comparative tables**

---

## Example Results

| Ensemble Method                 | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown |
|---------------------------------|--------------|-------------------|--------------|--------------|
| Weighted Voting                 | +257.6%      | 16.6%             | 0.58         | –58.3%       |
| Prob. Voting + Kelly            | –2.25%       | –0.27%            | –0.31        | –3.5%        |
| ...                             | ...          | ...               | ...          | ...          |

*(See notebook for full evaluation tables.)*

---

## License

This project is licensed under the **GNU Affero General Public License v3.0**.
See the [LICENSE](LICENSE) file for details.
