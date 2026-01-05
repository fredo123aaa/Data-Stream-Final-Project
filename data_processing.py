import pandas as pd 
import numpy as np 
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler

def prepare_stock_data(df, mid_price_type='HL'):
    """
    Prepare stock data with mid-price calculation and normalization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data with OHLC prices
    mid_price_type : str
        'HL' for (High+Low)/2 or 'OC' for (Open+Close)/2
    """
    df = df.copy()
    
    # Calculate mid-price
    if mid_price_type == 'HL':
        df['mid_price'] = (df['High'] + df['Low']) / 2
    else:  # 'OC'
        df['mid_price'] = (df['Open'] + df['Close']) / 2
    
    # Calculate percentage change
    df['pct_change'] = df['mid_price'].pct_change()
    
    # Remove NaN
    df = df.dropna()
    
    # Min-Max normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['pct_change_norm'] = scaler.fit_transform(df[['pct_change']])
    
    return df


def create_stream(data, embedding_dim=5):
    """
    Create streaming data with autoregressive features
    
    Parameters:
    -----------
    data : pd.DataFrame
        Prepared data with 'pct_change_norm' column
    embedding_dim : int
        Number of lag features
        
    Yields:
    -------
    x : dict
        Features as dictionary
    y : float
        Target value
    """
    values = data['pct_change_norm'].values
    
    for i in range(embedding_dim, len(values)):
        # Features: past embedding_dim values
        x = {f'lag_{j}': values[i-j-1] for j in range(embedding_dim)}
        # Target: current value
        y = values[i]
        yield x, y