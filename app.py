"""
Streamlit interface for Stock Sentiment Analysis Tool.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib
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
import plotly.io as pio
pio.renderers.default = "browser"
matplotlib.use("Agg")

# Import from existing script
from sentiment_analysis import Config, SentimentAnalyser, NewsScraper, Results

# Set page configuration
st.set_page_config(
    page_title="Stock Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create sidebar for inputs
st.sidebar.title("Stock Sentiment Analyser")
st.sidebar.write("Analyse sentiment from financial news")

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

tab1, tab2 = st.tabs(["Analysis", "About"])

results = Results()

with tab1:
    def run_analysis():
        if not selected_stock:
            st.warning("Please enter a valid stock")
            return False, Results()
        with st.spinner(f"Analysing sentiment for {selected_stock}..."):
            config = Config()
            config.stock_symbol = selected_stock
            config.USE_CNBC = use_cnbc
            config.USE_YF = use_yf
            config.USE_QUARTERLY_REVIEW = use_quarterly and selected_stock == 'AAPL'
            
            try:
                analyser = SentimentAnalyser(selected_stock)
                
                local_results = Results()
                
                # Collect headlines
                progress_text = st.empty()
                progress_text.text("Collecting news headlines...")
                scraper = NewsScraper(selected_stock)
                headlines = scraper.get_all_headlines()
                
                progress_text.text(f"Analysing {len(headlines)} headlines...")
                local_results.headlines = list(headlines)
                
                result = analyser.analyse_sentiment(headlines)
                
                local_results.score = result
                local_results.total_positive = analyser.total_positive
                local_results.total_neutral = analyser.total_neutral
                local_results.total_negative = analyser.total_negative
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
    analyse_button = st.button("Analyse Sentiment", type="primary", use_container_width=True)
    if analyse_button:
        success, results = run_analysis()
        if results.headline_count == 0:
            success = False
            results = Results()
    
    # Analysis section
    if success:
        # Display sentiment gauge chart
        st.subheader("Sentiment Score")
        
        score = results.score
        sentiment_label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        sentiment_color = "#28a745" if score > 0 else "#d9534f" if score < 0 else "#3498db"
        
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

    if results.analysis_complete:
        
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
        
        st.metric(
            label="Headlines Analysed", 
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
        st.info("Click 'Analyse Sentiment' to run the analysis")

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
        
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[results.total_positive, results.total_neutral, results.total_negative],
            text=[f'{p:.1f}%' for p in (np.array([results.total_positive, results.total_neutral, results.total_negative]) / sum([results.total_positive, results.total_neutral, results.total_negative]) * 100)],
            textposition='auto',
            marker_color=['#2ecc71', '#74B9FF', '#e74c3c']
        ))

        fig.update_layout(
            title=f'Sentiment Analysis Results for {selected_stock}',
            xaxis_title='Sentiment Category',
            yaxis_title='Sentiment Score',
            autosize=True,  # Forces a refresh
            template='plotly_white',
            height=500,
            newshape=dict(line=dict(color='black'))  # Ensures full refresh
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
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[results.total_positive, results.total_neutral, results.total_negative],
            hole=0.4,
            marker_colors=['#2ecc71', '#74B9FF', '#e74c3c']
        ))

        fig.update_layout(autosize=True, newshape=dict(line=dict(color='black')))  # Force re-render
        st.plotly_chart(fig, use_container_width=True)


        # # Fetch the stock data from Yahoo Finance using yfinance
        # data = yf.download(selected_stock, period='1mo', interval='1d')

        # # Check if data was fetched properly
        # if not data.empty:
        #     st.write(data.head())
        #     # Reset the index to ensure 'Date' is a column
        #     data.reset_index(inplace=True)

        #     # Ensure 'Close' column is present
        #     if 'Close' in data.columns:
        #         y_values = data['Close'].squeeze()  # Convert to 1D if needed
        #     else:
        #         st.error("The 'Close' column is missing in the dataset.")
        #         st.stop()
            
        #     # Compute price change
        #     latest_close = float(data['Close'].iloc[-1])
        #     oldest_close = float(data['Close'].iloc[0])
        #     change_pct = (latest_close - oldest_close) / oldest_close * 100

        #     # Display metrics
        #     c1, c2 = st.columns(2)
        #     c1.metric("Current Price", f"${latest_close:.2f}")
        #     c2.metric("1-Month Change", f"{change_pct:.2f}%", delta=f"{change_pct:.1f}%")

        #     # Interactive Stock Chart with Plotly
        #     fig = go.Figure()

        #     fig.add_trace(go.Scatter(
        #         x=data['Date'], 
        #         y=y_values, 
        #         mode='lines',
        #         name='Closing Price',
        #         line=dict(color='#1f77b4', width=2),
        #     ))

        #     fig.update_layout(
        #         title=f'{selected_stock} Stock Closing Price Over the Last Month',
        #         xaxis_title='Date',
        #         yaxis_title='Price in USD',
        #         template='plotly_white',
        #         autosize=True,  # Forces refresh
        #         height=500,
        #         hovermode="x",  # Interactive crosshair on hover
        #         xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor"),
        #         yaxis=dict(showspikes=True, spikemode="across"),
        #         margin=dict(l=40, r=40, t=50, b=40)  # Adjust margins
        #     )

        #     # Show interactive chart
        #     st.plotly_chart(fig, use_container_width=True)
        # else:
        #     st.error("No data found for the given ticker. Please try a different one.")

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

with tab2:
    st.write("""
    This app analyses sentiment from financial news using the FinBERT model.
    It is a Python-based sentiment analysis tool that evaluates market sentiment through financial news headlines and quarterly reports using FinBERT.
    This program aggregates headlines from multiple sources to provide sentiment insights.
    
    
    If you have any suggestions, feel free to drop me a message at jay.1.shah@kcl.ac.uk
    """)
    with st.expander(label="Limitations", expanded=False):
        st.markdown("""
    ### Key Limitations
    
    1. Limited news sources (currently CNBC and Yahoo Finance only)
    2. Headlines must explicitly mention company name/ticker symbol to be included
    3. Subject to limitations of the FinBERT model itself
    4. Headlines analysed in isolation without full article context
    
    *This tool is intended to provide a basic sentiment overview only and should be used accordingly*
    """)
# Create a footer
st.markdown("---")
st.caption("Financial Sentiment Analysis Tool using FinBERT | Not financial advice")
