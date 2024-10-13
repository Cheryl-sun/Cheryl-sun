# Stock Price Prediction Using XGBoost

This project uses historical stock price data from Apple Inc. (AAPL) and XGBoost, a machine learning algorithm, to predict future stock movements. The prediction is based on whether the stock's closing price will increase or decrease the next day. The model is evaluated using common classification metrics, including precision, recall, F1-score, and AUC-ROC.

## Prerequisites

Before running the code, ensure that you have the following libraries installed:

- `yfinance` (for downloading stock data)
- `pandas` (for data manipulation)
- `numpy` (for numerical computations)
- `xgboost` (for machine learning modeling)
- `matplotlib` (for plotting)
- `scikit-learn` (for evaluation metrics and confusion matrix)

To install these dependencies, run:

```bash
pip install yfinance pandas numpy xgboost matplotlib scikit-learn
```

## Project Overview

This project downloads historical stock price data for Apple Inc. and processes it to create a dataset suitable for training a binary classification model. The target variable indicates whether the stock price will rise the next day.

The project workflow involves:
1. Downloading Apple stock data using `yfinance`.
2. Plotting the stock's closing price over time.
3. Creating a target variable for predicting whether the next day's price will be higher.
4. Splitting the data into training and testing sets.
5. Training an XGBoost classifier model on the training set.
6. Making predictions on the test set.
7. Evaluating the model using precision, recall, F1-score, confusion matrix, and AUC-ROC.

## Code Explanation

### Data Loading

```python
import yfinance as yf
df = yf.Ticker("AAPL").history(period="max")
```
This block uses the `yfinance` API to download historical stock price data for Apple. The data includes columns like `Open`, `High`, `Low`, `Close`, and `Volume`.

### Data Preprocessing

The code shifts the closing price by one day to create the target variable, which indicates whether the price will go up the next day:

```python
df["Tomorrow"] = df["Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
```

Only data from 1989 onwards is used:

```python
df = df.loc[df.Date >= "1989-12-31",].copy()
```

### Train-Test Split

The dataset is split into training and test sets, with the last 100 data points used for testing:

```python
train = df.iloc[:-100]
test = df.iloc[-100:]
```

### Model Training

An XGBoost classifier is trained using the `Close`, `Volume`, `Open`, `High`, and `Low` features:

```python
predictors = ["Close", "Volume", "Open", "High", "Low"]
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
model.fit(train[predictors], train["Target"])
```

### Model Evaluation

The model's performance is evaluated using precision, recall, and F1-score:

```python
precision = precision_score(test["Target"], preds)
recall = recall_score(test["Target"], preds)
f1 = f1_score(test["Target"], preds)
```

A confusion matrix and its visualization are also included:

```python
cm = confusion_matrix(test["Target"], preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
```
<img width="521" alt="截屏2024-10-13 下午3 51 51" src="https://github.com/user-attachments/assets/20f4138c-8241-4e16-9eff-71a9d4562c83">


### ROC-AUC Curve

The ROC curve is plotted to visualize the model's performance:

```python
fpr, tpr, thresholds = roc_curve(test["Target"], y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
```
<img width="593" alt="截屏2024-10-13 下午3 54 04" src="https://github.com/user-attachments/assets/78e541b1-24a0-42c8-839a-745d5f5e87d4">

## Running the Code

To run the code, you need to:

1. Download the stock data or provide your own CSV file.
2. Make sure the necessary libraries are installed.
3. Run the script in your environment.

## Output

The script will generate:
- Line plot of the stock's closing price over time.
- Confusion matrix for model predictions.
- Precision, recall, and F1-scores.
- ROC curve with AUC score.
