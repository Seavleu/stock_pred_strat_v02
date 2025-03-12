import numpy as np
import torch

def backtest_model(model, test_loader, initial_capital=1000000, threshold=0.0):
    """
    Simulate a simple trading strategy based on model predictions.
    For each time step, if the predicted return (or price change) exceeds a threshold,
    a 'buy' signal is generated. Computes cumulative return.
    """
    model.eval()
    predictions = []
    actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    # Compute daily returns from actual prices (simple percentage change)
    daily_returns = actuals[1:] / actuals[:-1] - 1
    # Generate signals: 1 if predicted next-day return > threshold, else 0.
    signals = (predictions[1:] > threshold).astype(float)
    # Strategy returns: apply signal to daily returns.
    strategy_returns = signals * daily_returns
    cumulative_return = np.prod(1 + strategy_returns) - 1
    print(f"Backtested cumulative return: {cumulative_return*100:.2f}%")
    return cumulative_return
