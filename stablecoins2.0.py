# pip install requests
# pip install pandas
# pip install matplotlib
# pip install plotly
# pip install openpyxl  # for Excel export

import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

def get_coin_price_history_batch(coin_id, from_date, to_date):
    """Fetch price data for a specific date range"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    
    # Convert dates to Unix timestamps
    from_timestamp = int(pd.to_datetime(from_date).timestamp())
    to_timestamp = int(pd.to_datetime(to_date).timestamp())
    
    params = {
        'vs_currency': 'usd',
        'from': from_timestamp,
        'to': to_timestamp
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'error' in data:
            print(f"Error for {coin_id} ({from_date} to {to_date}): {data['error']}")
            return pd.DataFrame()
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['price'].astype(float)
        df = df.set_index('date')
        return df[['price']]
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {str(e)}")
        return pd.DataFrame()

def get_coin_price_history(coin_id, start_date=None):
    """Fetch complete price history in batches, working backwards from current date"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(datetime.now())
    
    # Calculate total days and number of batches needed
    total_days = (end_dt - start_dt).days
    
    print(f"\nFetching data for {coin_id}")
    print(f"Total days requested: {total_days}")
    
    all_data = []
    
    # For historical coins, we can only get the last 365 days of data
    actual_start = max(start_dt, end_dt - timedelta(days=365))
    
    # Fetch the last 365 days
    df_batch = get_coin_price_history_batch(coin_id, actual_start, end_dt)
    if not df_batch.empty:
        all_data.append(df_batch)
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all batches and remove any duplicates
    combined_df = pd.concat(all_data)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()
    
    # Verify we got all the data we wanted
    if not combined_df.empty:
        actual_start = combined_df.index.min()
        actual_end = combined_df.index.max()
        print(f"\nData retrieved:")
        print(f"Requested start: {start_dt.strftime('%Y-%m-%d')}")
        print(f"Actual start: {actual_start.strftime('%Y-%m-%d')}")
        print(f"Actual end: {actual_end.strftime('%Y-%m-%d')}")
        print(f"Total days: {len(combined_df)}")
    
    return combined_df

def analyze_stablecoins(coin_id, coin_name, start_date=None, threshold=0.99):
    print(f"\nAnalyzing {coin_name} (from {start_date if start_date else 'available data'})")
    df = get_coin_price_history(coin_id, start_date)
    
    if df.empty:
        print(f'Failed to retrieve data for {coin_name}')
        return
    
    # Calculate statistics
    df['deviation'] = (1 - df['price']) * 100
    df['below_threshold'] = df['price'] < threshold
    
    max_depeg_idx = df['deviation'].idxmax()
    max_depeg_price = df.loc[max_depeg_idx, 'price']
    
    stats = {
        'min_price': df['price'].min(),
        'max_price': df['price'].max(),
        'mean_price': df['price'].mean(),
        'max_depeg': df['deviation'].max(),
        'max_depeg_date': max_depeg_idx,
        'max_depeg_price': max_depeg_price,
        'days_below_threshold': df['below_threshold'].sum(),
        'total_days': len(df),
        'percent_below_threshold': (df['below_threshold'].sum() / len(df)) * 100,
        'data_start': df.index.min(),
        'data_end': df.index.max()
    }
    
    print(f"\nResults for {coin_name}:")
    print(f"Analysis period: {stats['data_start'].strftime('%Y-%m-%d')} to {stats['data_end'].strftime('%Y-%m-%d')}")
    print(f"Max depeg from $1: {stats['max_depeg']:.2f}%")
    print(f"Date of max depeg: {stats['max_depeg_date'].strftime('%Y-%m-%d')}")
    print(f"Price at max depeg: ${stats['max_depeg_price']:.4f}")
    print(f"Price range: ${stats['min_price']:.4f} - ${stats['max_price']:.4f}")
    print(f"Mean price: ${stats['mean_price']:.4f}")
    print(f"Days below threshold: {stats['days_below_threshold']} out of {stats['total_days']}")
    print(f"Percent below threshold: {stats['percent_below_threshold']:.2f}%")

    # Create interactive plot with Plotly
    fig = go.Figure()

    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['price'],
            name='Price',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
        )
    )

    # Add target price line
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[1, 1],
            name='Target ($1)',
            line=dict(color='red', dash='dash')
        )
    )

    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[threshold, threshold],
            name=f'Threshold (${threshold})',
            line=dict(color='orange', dash='dot')
        )
    )

    # Add deviation markers for significant deviations (>0.5%)
    significant_deviations = df[abs(df['deviation']) > 0.5]
    if not significant_deviations.empty:
        fig.add_trace(
            go.Scatter(
                x=significant_deviations.index,
                y=significant_deviations['price'],
                mode='markers',
                name='Significant Deviations',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle'
                ),
                hovertemplate='Date: %{x}<br>Price: $%{y:.4f}<br>Deviation: %{text:.2f}%<extra></extra>',
                text=significant_deviations['deviation']
            )
        )

    # Update layout
    fig.update_layout(
        title=f'{coin_name} Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(
            range=[
                max(0, min(df['price'].min() * 0.99, 0.95)),
                min(df['price'].max() * 1.01, 1.05)
            ]
        )
    )

    fig.show()
    
    return df, stats

def export_data(results, output_dir='analysis_results'):
    """Export analysis results to CSV and Excel files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for export
    all_prices = pd.DataFrame()
    stats_data = []
    
    for coin_name, (df, stats) in results.items():
        # Add coin prices to the combined dataframe
        price_col = f'{coin_name}_price'
        deviation_col = f'{coin_name}_deviation'
        
        # Create a new DataFrame with the correct column names
        coin_df = pd.DataFrame({
            price_col: df['price'],
            deviation_col: df['deviation']
        }, index=df.index)
        
        all_prices = pd.concat([all_prices, coin_df], axis=1)
        
        # Collect statistics
        stats_data.append({
            'Coin': coin_name,
            'Min Price': stats['min_price'],
            'Max Price': stats['max_price'],
            'Mean Price': stats['mean_price'],
            'Max Depeg (%)': stats['max_depeg'],
            'Max Depeg Date': stats['max_depeg_date'],
            'Price at Max Depeg': stats['max_depeg_price'],
            'Days Below Threshold': stats['days_below_threshold'],
            'Total Days': stats['total_days'],
            'Percent Below Threshold': stats['percent_below_threshold']
        })
    
    # Export combined price data
    all_prices.to_csv(f'{output_dir}/price_history.csv')
    all_prices.to_excel(f'{output_dir}/price_history.xlsx')
    
    # Export statistics
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f'{output_dir}/statistics.csv', index=False)
    stats_df.to_excel(f'{output_dir}/statistics.xlsx', index=False)
    
    print(f"\nData exported to {output_dir}:")
    print("- price_history.csv/.xlsx: Complete price and deviation history")
    print("- statistics.csv/.xlsx: Summary statistics for each coin")

if __name__ == "__main__":
    coins = [
        ("anzen-usdz", "Anzen USDZ", "2024-05-22"),  # Launch date: May 22, 2024
        ("level-usd", "lvlUSD", "2023-12-28"),       # Launch date: Dec 28, 2023
        ("crvusd", "crvUSD", "2023-05-24"),          # Launch date: May 24, 2023
        ("elixir-deusd", "deUSD", "2024-08-02")      # Launch date: August 1, 2024
    ]
    
    results = {}
    threshold = float(input("Enter the threshold (e.g., 0.99): "))  # Convert input to float
    
    for coin_id, coin_name, start_date in coins:
        try:
            time.sleep(1)  # Add delay between requests to avoid rate limiting
            df, stats = analyze_stablecoins(coin_id, coin_name, start_date, threshold)
            if not df.empty:
                results[coin_name] = (df, stats)
        except Exception as e:
            print(f"Error processing {coin_name}: {str(e)}")
    
    # Export results to CSV and Excel
    if results:
        export_data(results)