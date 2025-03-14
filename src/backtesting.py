import numpy as np
import torch

def backtest_model(model, test_loader, initial_capital=1000000, threshold=0.0):
    """
    simulate a trading strategy based on model predictions
    
    for each day in the test set, if the predicted future average return exceeds a threshold,
    a long position is taken at the current price.
    
    if actual prices rise for two consecutive days,
    the position is held for two days; otherwise, the trade is exited after one day.
    
    capital is updated based on the realized return for each trade, and cumulative return is computed.
    
    args:
        model: trained model that outputs the predicted future_avg_return.
        test_loader: dataloader for the test set.
        initial_capital (float): starting capital.
        threshold (float): minimum predicted return to trigger a trade.
        
    returns:
        cumulative_return (float): total profit (or loss) from the strategy.
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
    
    if np.isnan(predictions).any() or np.isnan(actuals).any():
        raise ValueError("backtesting data contains nan values.")
    
    # model's prediction is the expected avg return over day+1 and day+2
    capital = initial_capital
    trades = 0
    # ensure we have at least two days ahead for each trade
    for t in range(len(actuals) - 2):
        if predictions[t] > threshold:
            entry_price = actuals[t]
            # if actual prices rise for two consecutive days, hold for two days; else exit after one day
            if actuals[t+1] > entry_price and actuals[t+2] > actuals[t+1]:
                exit_price = actuals[t+2]
            else:
                exit_price = actuals[t+1]
            trade_return = (exit_price / entry_price) - 1
            capital *= (1 + trade_return)
            trades += 1
    cumulative_return = capital - initial_capital
    print(f"executed {trades} trades")
    print(f"backtested cumulative return: {cumulative_return/initial_capital*100:.2f}%")
    return cumulative_return
