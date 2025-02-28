"""
Streamlit interface for Stock Sentiment Analysis Tool.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import traceback
import plotly.express as px

# Import from existing script
from sentiment_analysis import Config, SentimentAnalyser, NewsScraper, Results

# Set page configuration
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create sidebar for inputs
st.sidebar.title("Stock Sentiment Analyzer")
st.sidebar.write("Analyze sentiment from financial news")

selected_stock = ""
st.title(f"Stock Sentiment Analysis")
selected_stock = st.text_input("Enter a stock symbol:").upper()
if selected_stock:
    st.write(f"Selected stock: {selected_stock}")


st.sidebar.subheader("Settings")
use_cnbc = st.sidebar.checkbox("Use CNBC News", value=True)
use_yf = st.sidebar.checkbox("Use Yahoo Finance", value=True)
use_quarterly = st.sidebar.checkbox("Use Quarterly Reports (Apple only)", value=True) if selected_stock == 'AAPL' else False
show_stock_price = st.sidebar.checkbox("Show stock price chart", value=True)
show_raw_headlines = st.sidebar.checkbox("Show raw headlines")
# change the colour of the plots

with st.sidebar.expander("About"):
    st.write("""
    This app analyzes sentiment from financial news using the FinBERT model.
    It is a Python-based sentiment analysis tool that evaluates market sentiment through financial news headlines and quarterly reports using FinBERT.
    This program aggregates headlines from multiple sources to provide sentiment insights.
    
    
    If you have any suggestions, feel free to drop me a message at jay.1.shah@kcl.ac.uk
    """)

tab1, tab2 = st.tabs(["Analysis", "Raw Data"])

results = Results()

with tab1:
    def run_analysis():
        if not selected_stock:
            st.warning("Please enter a valid stock")
            return False, Results()
        with st.spinner(f"Analyzing sentiment for {selected_stock}..."):
            config = Config()
            config.stock_symbol = selected_stock
            config.USE_CNBC = use_cnbc
            config.USE_YF = use_yf
            config.USE_QUARTERLY_REVIEW = use_quarterly and selected_stock == 'AAPL'
            
            try:
                analyzer = SentimentAnalyser(selected_stock)
                
                local_results = Results()
                
                # Collect headlines
                progress_text = st.empty()
                progress_text.text("Collecting news headlines...")
                scraper = NewsScraper(selected_stock)
                headlines = scraper.get_all_headlines()
                
                progress_text.text(f"Analyzing {len(headlines)} headlines...")
                local_results.headlines = list(headlines)
                
                result = analyzer.analyse_sentiment(headlines)
                
                local_results.score = result
                local_results.total_positive = analyzer.total_positive
                local_results.total_neutral = analyzer.total_neutral
                local_results.total_negative = analyzer.total_negative
                local_results.headline_count = len(headlines)
                local_results.analysis_complete = True
                
                if len(headlines) > 0:
                    progress_text.text("Analysis complete!")
                else:
                    st.warning("An error occurred: check the stock ticker is valid")
                    
                return True, local_results
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.code(traceback.format_exc())
                return False, local_results
    
    success = False
    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    if analyze_button:
            success, results = run_analysis()
            if results.headline_count == 0:
                success = False
                results = Results()
    
    # Analysis section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if success:
            # Display sentiment gauge chart
            st.subheader("Sentiment Score")
            
            score = results.score
            sentiment_label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            sentiment_color = "#2ecc71" if score > 0 else "#e74c3c" if score < 0 else "#74B9FF"
            
            # Create a gauge chart with Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score * 100,  # Convert to percentage
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Sentiment: {sentiment_label}", "font": {"size": 24, "color": sentiment_color}},
                gauge={
                    "axis": {"range": [-100, 100], "tickwidth": 1},
                    "bar": {"color": sentiment_color},
                    "steps": [
                        {"range": [-100, -33], "color": "rgba(231, 76, 60, 0.3)"},
                        {"range": [-33, 33], "color": "rgba(116, 185, 255, 0.3)"},
                        {"range": [33, 100], "color": "rgba(46, 204, 113, 0.3)"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": score * 100
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if results.analysis_complete:
            st.subheader("Summary")
            
            # Create a dataframe for the summary
            data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Score': [
                    results.total_positive,
                    results.total_neutral,
                    results.total_negative
                ]
            })
            
            total = data['Score'].sum()
            data['Percentage'] = data['Score'] / total * 100
            
            # Show summary stats
            st.metric(
                label="Overall Score", 
                value=f"{results.score:.2f}",
                delta=f"{results.score * 100:.1f}%" 
            )
            st.metric(
                label="Headlines Analyzed", 
                value=results.headline_count
            )
            
            # Show a summary table
            st.dataframe(
                data.style.format({
                    'Score': '{:.2f}',
                    'Percentage': '{:.1f}%'
                }),
                use_container_width=True
            )
        else:
            st.info("Click 'Analyze Sentiment' to run the analysis")

    if results.analysis_complete:
        
    
        # bar chart
        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [
                results.total_positive,
                results.total_neutral,
                results.total_negative
            ]
        })
        
        total = data['Score'].sum()
        data['Percentage'] = data['Score'] / total * 100
        
        colors = ['#2ecc71', '#74B9FF', '#e74c3c']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data['Sentiment'],
            y=data['Score'],
            text=[f'{p:.1f}%' for p in data['Percentage']],
            textposition='auto',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=f'Sentiment Analysis Results for {selected_stock}',
            xaxis_title='Sentiment Category',
            yaxis_title='Sentiment Score',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # pie chart
        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [
                results.total_positive,
                results.total_neutral,
                results.total_negative
            ]
        })
        
        colors = ['#2ecc71', '#74B9FF', '#e74c3c']
        
        fig = go.Figure(data=[go.Pie(
            labels=data['Sentiment'],
            values=data['Score'],
            hole=.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title=f'Sentiment Distribution for {selected_stock}',
            annotations=[dict(text=f'{selected_stock}', x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


        # Fetch the stock data from Yahoo Finance using yfinance
        data = yf.download(selected_stock, period='1mo', interval='1d')

        # Check if data was fetched properly
        if not data.empty:
            # Reset the index to ensure 'Date' is a column
            data.reset_index(inplace=True)

            # Ensure 'Close' column is a Series, not a DataFrame
            if 'Close' in data.columns:
                y_values = data['Close'].squeeze()  # Convert to 1D if needed
            else:
                st.error("The 'Close' column is missing in the dataset.")
                st.stop()
            
            latest_close = float(data['Close'].iloc[-1])
            oldest_close = float(data['Close'].iloc[0])
            change_pct = (latest_close - oldest_close) / oldest_close * 100
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"${latest_close:.2f}")
            c2.metric("1-Month Change", f"{change_pct:.2f}%", delta=f"{change_pct:.1f}%")

            # Flatten column names if they are multi-indexed
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Plot using Plotly Express
            fig = px.line(data, x='Date', y=y_values, title=f'{selected_stock} Stock Closing Price Over the Last Month')

            # Customize the layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price in USD',
                template='plotly_dark'
            )

            # Display the interactive plot
            st.plotly_chart(fig)

        else:
            st.error("No data found for the given ticker. Please try a different one.")

if show_raw_headlines:
    if results.analysis_complete:
        if results.headline_count > 0:
            st.subheader("Raw Headlines Used for Analysis")
            
            # Create a dataframe for the headlines
            headline_df = pd.DataFrame({
                "Headline": results.headlines
            })
            
            # Add a filter
            filter_query = st.text_input("Filter headlines (contains text):")
            
            if filter_query:
                filtered_df = headline_df[headline_df["Headline"].str.contains(filter_query, case=False)]
                st.write(f"Showing {len(filtered_df)} of {len(headline_df)} headlines")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.write(f"Total headlines: {len(headline_df)}")
                st.dataframe(headline_df, use_container_width=True)
        else:
            st.info("Enable 'Show raw headlines' in the sidebar to view the raw data")
    else:
        st.info("Run the analysis first to see raw data")

# Create a footer
st.markdown("---")
st.caption("Financial Sentiment Analysis Tool using FinBERT | Not financial advice")
