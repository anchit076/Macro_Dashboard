import streamlit as st
import pandas as pd
import datetime

# Set page config for wider layout
st.set_page_config(layout="wide", page_title="Rolling Returns Dashboard")

def cleaner(df, date_col, close_price_col, instrument_col):
    """
    Calculate rolling returns for various time periods based on trading days.
    Works with multiple instruments using groupby.
    """
    cols = [date_col, close_price_col, instrument_col]
    
    clean_df = df.copy()
    clean_df = clean_df[cols]
    
    # Ensure date column is datetime
    clean_df[date_col] = pd.to_datetime(clean_df[date_col])
    
    # Sort by instrument and date
    clean_df = clean_df.sort_values([instrument_col, date_col]).reset_index(drop=True)
    
    # Define time periods in TRADING DAYS
    periods = {
        'Daily Returns %': 1,
        '1 Week Returns %': 5,
        '2 Week Returns %': 10,
        '3 Week Returns %': 15,
        '1 Month Returns %': 21,
        '2 Month Returns %': 42,
        '3 Month Returns %': 63,
        '4 Month Returns %': 84,
        '5 Month Returns %': 105,
        '6 Month Returns %': 126,
        '9 Month Returns %': 189,
        '1 Year- Annual Returns %': 252,
        '2 Year- Annual Returns %': 504,
        '3 Year- Annual Returns %': 756,
        '5 Year- Annual Returns %': 1260,
        '7 Year- Annual Returns %': 1764
    }
    
    # Years for CAGR calculation
    years_dict = {
        '1 Year- Annual Returns %': 1,
        '2 Year- Annual Returns %': 2,
        '3 Year- Annual Returns %': 3,
        '5 Year- Annual Returns %': 5,
        '7 Year- Annual Returns %': 7
    }
    
    # Calculate returns for each period using groupby
    for col_name, trading_days in periods.items():
        if col_name in years_dict:
            # CAGR calculation for periods >= 1 year
            years = years_dict[col_name]
            past_price = clean_df.groupby(instrument_col)[close_price_col].shift(trading_days)
            current_price = clean_df[close_price_col]
            clean_df[col_name] = ((current_price / past_price) ** (1 / years) - 1)
        else:
            # Absolute returns for periods < 1 year
            past_price = clean_df.groupby(instrument_col)[close_price_col].shift(trading_days)
            current_price = clean_df[close_price_col]
            clean_df[col_name] = ((current_price / past_price) - 1)
    
    return clean_df


def calculate_stats(df, return_columns, start_date=None, end_date=None):
    """
    Calculate descriptive statistics for the given dataframe and date range.
    Returns a dictionary with stats for each return column.
    """
    # Filter by date range if provided
    if start_date and end_date:
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    else:
        df_filtered = df.copy()
    
    stats_dict = {}
    
    for col in return_columns:
        col_data = df_filtered[col].dropna()
        
        if len(col_data) == 0:
            stats_dict[col] = {
                'Current': None,
                'Min': None,
                '25th Percentile': None,
                'Median': None,
                '75th Percentile': None,
                'Max': None,
                'Average': None,
                'Range': None
            }
        else:
            mean = col_data.mean()
            
            stats_dict[col] = {
                'Current': df_filtered[col].iloc[-1] if len(df_filtered) > 0 else None,
                'Min': col_data.min(),
                '25th Percentile': col_data.quantile(0.25),
                'Median': col_data.median(),
                '75th Percentile': col_data.quantile(0.75),
                'Max': col_data.max(),
                'Average': mean,
                'Range': col_data.max() - col_data.min()
            }
    
    return stats_dict


def calculate_percentile_ranks(df, return_columns, current_returns, end_date):
    """
    Calculate percentile ranks for current returns across different lookback periods.
    Percentile ranks are shown as percentages.
    Time periods are relative to the selected end_date:
    - Last 5 years
    - Last 10 years
    - Last 20 years
    - Since inception
    """
    import datetime
    from dateutil.relativedelta import relativedelta
    
    percentile_ranks = {}
    
    # Get the earliest date in the dataset (inception)
    earliest_date = df['Date'].min()
    earliest_year = earliest_date.year
    
    # Get end date year
    end_year = end_date.year
    
    # Define lookback periods relative to end_date
    lookback_periods = {
        '5 years': 5,
        '10 years': 10,
        '20 years': 20,
        'inception': None  # Special case: from earliest date to end_date
    }
    
    for period_name, years_back in lookback_periods.items():
        if years_back is None:
            # Inception: from earliest date to end_date
            start_period = earliest_date
            period_label = f'Since Inception ({earliest_year}-{end_year})'
        else:
            # Calculate start date by going back N years from end_date
            start_period = end_date - relativedelta(years=years_back)
            
            # Don't go before inception
            if start_period < earliest_date:
                start_period = earliest_date
            
            start_year = start_period.year
            period_label = f'Last {years_back} Years ({start_year}-{end_year})'
        
        percentile_ranks[period_label] = {}
        
        # Filter data for this time period
        df_period = df[(df['Date'] >= start_period) & (df['Date'] <= end_date)]
        
        for col in return_columns:
            col_data = df_period[col].dropna()
            
            if len(col_data) > 0 and current_returns.get(col) is not None:
                # Calculate percentile rank (as decimal, will be formatted as % later)
                rank = (col_data < current_returns[col]).sum() / len(col_data)
                percentile_ranks[period_label][col] = rank
            else:
                percentile_ranks[period_label][col] = None
    
    return percentile_ranks


def style_current_returns(display_df, stats, return_columns):
    """
    Apply color coding to the Current Prices row based on percentile thresholds.
    Green if below 25th percentile, Red if above 75th percentile.
    """
    def color_cell(val, col_name):
        """
        Determine cell color based on value and percentile thresholds.
        """
        # Skip if this is not a return column or if value is N/A
        if col_name not in return_columns or val == "N/A":
            return ''
        
        # Remove percentage sign and convert to float
        try:
            current_val = float(val.strip('%')) / 100
        except:
            return ''
        
        # Get the 25th and 75th percentile for this column
        percentile_25 = stats[col_name].get('25th Percentile')
        percentile_75 = stats[col_name].get('75th Percentile')
        
        if percentile_25 is None or percentile_75 is None:
            return ''
        
        # Apply color logic
        if current_val < percentile_25:
            return 'background-color: #90EE90'  # Light green
        elif current_val > percentile_75:
            return 'background-color: #FFB6C6'  # Light red/pink
        else:
            return ''  # No color (white)
    
    # Create a style DataFrame with same shape as display_df
    styles = pd.DataFrame('', index=display_df.index, columns=display_df.columns)
    
    # Apply styling only to the first row (Current Prices)
    for col in return_columns:
        if col in display_df.columns:
            styles.loc[0, col] = color_cell(display_df.loc[0, col], col)
    
    return styles


# Main Streamlit App
def main():
    st.title("Rolling Returns Dashboard")
    
    # Load data
    try:
        # Read your input file - UPDATE THIS PATH
        input_df = pd.read_excel("Quant Dataset.xlsx")  # Use your actual path here
        
        # Process data using the cleaner function
        clean_df = cleaner(input_df, 'Date', 'Close Price', 'Index Name')
        
        # Get list of instruments
        instruments = clean_df['Index Name'].unique().tolist()
        
        # Dropdown for instrument selection (positioned like "Index Name" in your Excel)
        selected_instrument = st.selectbox("Index Name", instruments)
        
        # Filter data for selected instrument
        df_instrument = clean_df[clean_df['Index Name'] == selected_instrument].copy()
        df_instrument = df_instrument.sort_values('Date', ascending=False).reset_index(drop=True)
        
        # Date range selector
        min_date = df_instrument['Date'].min()
        max_date = df_instrument['Date'].max()
        
        st.subheader("Select Date Range for Statistics Calculation")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Convert to datetime for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Define return columns
        return_columns = [
            'Daily Returns %',
            '1 Week Returns %',
            '2 Week Returns %',
            '3 Week Returns %',
            '1 Month Returns %',
            '2 Month Returns %',
            '3 Month Returns %',
            '4 Month Returns %',
            '5 Month Returns %',
            '6 Month Returns %',
            '9 Month Returns %',
            '1 Year- Annual Returns %',
            '2 Year- Annual Returns %',
            '3 Year- Annual Returns %',
            '5 Year- Annual Returns %',
            '7 Year- Annual Returns %'
        ]
        
         # Get current returns AS OF the selected end_date
        current_returns = {}
        if len(df_instrument) > 0:
            # Find the row that matches or is closest to the end_date
            df_instrument_sorted = df_instrument.sort_values('Date', ascending=True)
            
            # Find the index of the date closest to (but not after) end_date
            valid_dates = df_instrument_sorted[df_instrument_sorted['Date'] <= end_date]
            
            if len(valid_dates) > 0:
                # Get the last valid date (closest to end_date without exceeding it)
                end_date_row = valid_dates.iloc[-1]
                
                # Extract current returns from that specific date
                for col in return_columns:
                    current_returns[col] = end_date_row[col]
            else:
                # If no valid date found, use None
                for col in return_columns:
                    current_returns[col] = None
        
        # Calculate statistics based on selected date range
        stats = calculate_stats(df_instrument, return_columns, start_date, end_date)
        
        # Calculate percentile ranks (pass end_date as parameter)
        percentile_ranks = calculate_percentile_ranks(df_instrument, return_columns, current_returns, end_date)
        
        # Create the statistics table (Excel-like format)
        st.subheader("Statistics Table")
        
        # Build the table data
        table_data = []
        
        # Row 1: Current Returns
        current_row = ['Current Returns'] + [f"{current_returns.get(col, 0):.2%}" if current_returns.get(col) is not None else "N/A" for col in return_columns]
        table_data.append(current_row)
        
        # Empty row
        table_data.append([''] * (len(return_columns) + 1))
        
        # Stats rows
        stat_labels = ['Min', '25th Percentile (0.25)', 'Median (0.5)', '75th Percentile (0.75)', 'Max', 
                       '', 'Average', 'Range']
        
        stat_keys = ['Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 
                     None, 'Average', 'Range']
        
        for label, key in zip(stat_labels, stat_keys):
            if key is None:
                table_data.append([''] * (len(return_columns) + 1))
            else:
                row = [label]
                for col in return_columns:
                    val = stats[col].get(key)
                    if val is not None:
                        row.append(f"{val:.2%}")
                    else:
                        row.append("N/A")
                table_data.append(row)
        
        # Empty row
        table_data.append([''] * (len(return_columns) + 1))
        
        # Percentile rank rows
        for period_name in percentile_ranks.keys():
            row = [f'Percentile Rank {period_name}']
            for col in return_columns:
                val = percentile_ranks[period_name].get(col)
                if val is not None:
                    row.append(f"{val:.2%}")  # Format as percentage
                else:
                    row.append("N/A")
            table_data.append(row)
        
        # Convert to DataFrame for display
        display_df = pd.DataFrame(table_data, columns=['Index Name'] + return_columns)
        
        # Apply color styling to the table
        styled_df = display_df.style.apply(
            lambda _: style_current_returns(display_df, stats, return_columns), 
            axis=None
        )
        
        # Display the styled table
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ============ ADD LINE CHART HERE ============
        
        # LINE CHART - Add after the table, before Export Options
        st.subheader("Returns Across Time Horizons")
        
        # Prepare data for the chart (excluding Daily Returns %)
        chart_columns = [
            '1 Week Returns %',
            '2 Week Returns %',
            '3 Week Returns %',
            '1 Month Returns %',
            '2 Month Returns %',
            '3 Month Returns %',
            '4 Month Returns %',
            '5 Month Returns %',
            '6 Month Returns %',
            '9 Month Returns %',
            '1 Year- Annual Returns %',
            '2 Year- Annual Returns %',
            '3 Year- Annual Returns %',
            '5 Year- Annual Returns %',
            '7 Year- Annual Returns %'
        ]
        
        # Extract data for each line
        current_values = []
        percentile_25_values = []
        median_values = []
        percentile_75_values = []
        
        for col in chart_columns:
            current_values.append(current_returns.get(col, None))
            percentile_25_values.append(stats[col].get('25th Percentile', None))
            median_values.append(stats[col].get('Median', None))
            percentile_75_values.append(stats[col].get('75th Percentile', None))
        
        # Create chart data as DataFrame
        chart_data = pd.DataFrame({
            'Time Horizon': chart_columns,
            'Current Returns': current_values,
            '25th Percentile': percentile_25_values,
            'Median': median_values,
            '75th Percentile': percentile_75_values
        })
        
        # Create the line chart using Plotly
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add Current Returns line (Blue)
        fig.add_trace(go.Scatter(
            x=chart_data['Time Horizon'],
            y=chart_data['Current Returns'],
            mode='lines+markers',
            name='Current Returns',
            line=dict(color='blue', width=2),
            marker=dict(size=6, color='blue')
        ))
        
        # Add 25th Percentile line (Green)
        fig.add_trace(go.Scatter(
            x=chart_data['Time Horizon'],
            y=chart_data['25th Percentile'],
            mode='lines+markers',
            name='25th Percentile',
            line=dict(color='green', width=2),
            marker=dict(size=6, color='green')
        ))
        
        # Add Median line (Yellow/Gold)
        fig.add_trace(go.Scatter(
            x=chart_data['Time Horizon'],
            y=chart_data['Median'],
            mode='lines+markers',
            name='Median',
            line=dict(color='gold', width=2),
            marker=dict(size=6, color='gold')
        ))
        
        # Add 75th Percentile line (Red)
        fig.add_trace(go.Scatter(
            x=chart_data['Time Horizon'],
            y=chart_data['75th Percentile'],
            mode='lines+markers',
            name='75th Percentile',
            line=dict(color='red', width=2),
            marker=dict(size=6, color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{selected_instrument} - Returns Distribution Across Time Horizons",
            xaxis_title="Time Horizon",
            yaxis_title="Returns",
            yaxis_tickformat='.1%',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=-45)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # ============ END OF CHART ============

        # Option to download as Excel
        st.subheader("Export Options")
        if st.button("Export to Excel"):
            output_filename = f"{selected_instrument}_stats_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
            display_df.to_excel(output_filename, index=False)
            st.success(f"File exported as {output_filename}")

    except FileNotFoundError:
        st.error("Inputfile.csv not found. Please ensure the file exists in the same directory as this script.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Full error details:")
        st.exception(e)


if __name__ == "__main__":

    main()
