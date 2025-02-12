import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import torch
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Set, List, Tuple

class Config:
    # Sentiment Analysis settings
    MIN_HEADLINE_LENGTH = 5

    SENTIMENT_WEIGHTS = {
        'positive': 1.0,
        'neutral': 0.0,
        'negative': -1.0
    }

    PLOT_COLOURS = {
        'positive': '#2ecc71',
        'neutral': '#74B9FF',
        'negative': '#e74c3c'
    }

class NewsScraper:
    def __init__(self, stock_symbol: str):
        self.stock_symbol = stock_symbol
        self.short_name = self.get_company_short_name(stock_symbol)

    def get_all_headlines(self) -> Set[str]:
        if self.short_name is None:
            st.error('Invalid input parameters, unable to execute')
            return set()

        headlines = []
        try:
            # CNBC Headlines
            cnbc_headlines = self.get_cnbc_headlines()

            # Yahoo Finance Headlines
            yf_headlines = self.get_yf_headlines()

            headlines = [
                headline for headline in (cnbc_headlines + yf_headlines) 
                if (self.stock_symbol.lower() in headline.lower() or 
                    self.short_name.lower() in headline.lower())
                ]

            return set(headlines)
        except Exception as e:
            st.error(f"Error retrieving headlines: {e}")
            return set()

    def get_company_short_name(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            return stock.info.get('shortName', '').split()[0].strip(',')
        except Exception:
            st.error('Error: Stock ticker invalid')
            return None

    def get_cnbc_headlines(self) -> List[str]:
        headlines = []
        url = f"https://www.cnbc.com/quotes/{self.stock_symbol}?tab=news"
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            to_analyse = soup.find(class_="QuotePageTabs")
            
            if to_analyse:
                headline_elements = to_analyse.find_all(['h3', 'a'], class_=lambda x: x and 'headline' in x.lower())
                
                for element in headline_elements:
                    headline = None
                    if element.get('title'):
                        headline = element['title']
                    elif element.text.strip():
                        headline = element.text.strip()
                    elif element.get('aria-label'):
                        headline = element['aria-label']
                    
                    if headline and any(substring.lower() in headline.lower() for substring in [*self.short_name, self.stock_symbol]) \
                       and len(headline) >= Config.MIN_HEADLINE_LENGTH:
                        headlines.append(headline)
            
            return list(dict.fromkeys(headlines))  # Remove duplicates
        
        except Exception as e:
            st.error(f"Error retrieving CNBC headlines: {e}")
            return []

    def get_yf_headlines(self) -> List[str]:
        headlines = []
        try:
            stock = yf.Ticker(self.stock_symbol)
            if stock.news:
                for entry in stock.news:
                    title = entry.get('title') or entry.get('headline') or entry.get('description')
                    
                    if title and any(name.lower() in title.lower() for name in [*self.short_name, self.stock_symbol]):
                        headlines.append(title)
            
            return headlines
        
        except Exception as e:
            st.error(f'Error retrieving Yahoo Finance headlines: {e}')
            return []

class SentimentAnalyser:
    def __init__(self, stock_symbol: str):
        self.stock_symbol = stock_symbol
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    def analyse_sentiment(self, headlines: Set[str]) -> Tuple[float, pd.DataFrame]:
        overall_score = 0
        num_headlines = len(headlines)

        total_positive, total_neutral, total_negative = 0, 0, 0
        progress_bar = st.progress(0)

        for i, headline in enumerate(headlines):
            progress_bar.progress(min(int((i+1)/num_headlines * 100), 100))
            
            if len(headline) < Config.MIN_HEADLINE_LENGTH:
                continue

            inputs = self.tokenizer(headline, return_tensors="pt")
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            
            positive_prob, neutral_prob, negative_prob = probabilities[0].item(), probabilities[1].item(), probabilities[2].item()

            total_positive += positive_prob
            total_neutral += neutral_prob
            total_negative += negative_prob

            overall_score += (
                positive_prob * Config.SENTIMENT_WEIGHTS['positive'] +
                neutral_prob * Config.SENTIMENT_WEIGHTS['neutral'] +
                negative_prob * Config.SENTIMENT_WEIGHTS['negative']
            )

        weighted_avg_score = overall_score / num_headlines if num_headlines > 0 else 0

        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [total_positive, total_neutral, total_negative],
        })

        total = data['Score'].sum()
        data['Percentage'] = data['Score'] / total * 100

        return weighted_avg_score, data

    def plot_results(self, data: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))

        colours = [Config.PLOT_COLOURS['positive'], Config.PLOT_COLOURS['neutral'], Config.PLOT_COLOURS['negative']]
        bars = ax.bar(data['Sentiment'], data['Score'], color=colours, alpha=0.7, width=0.6, edgecolor='white', linewidth=2)

        for bar in bars:
            x = bar.get_x()
            w = bar.get_width()
            h = bar.get_height()
            ax.add_patch(Rectangle((x, 0), w, h, facecolor='black', alpha=0.05, zorder=0))

        for i, (bar, percentage) in enumerate(zip(bars, data['Percentage'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}\n({percentage:.1f}%)',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        ax.set_title(f'Sentiment Analysis for {self.stock_symbol}')
        ax.set_xlabel('Sentiment Category')
        ax.set_ylabel('Sentiment Score')

        return fig

def main():
    st.title("Stock News Sentiment Analysis")
    
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    
    if st.button("Analyze Sentiment"):
        scraper = NewsScraper(stock_symbol)
        headlines = scraper.get_all_headlines()
        
        if not headlines:
            st.error("No headlines found. Please check the stock ticker.")
            return
        
        with st.expander("Retrieved Headlines"):
            for headline in headlines:
                st.write(headline)
        
        analyser = SentimentAnalyser(stock_symbol)
        overall_score, sentiment_data = analyser.analyse_sentiment(headlines)
        
        st.subheader("Overall Sentiment")
        sentiment_text = 'Positive' if overall_score > 0 else 'Negative' if overall_score < 0 else 'Neutral'
        st.metric("Sentiment", sentiment_text, f"{round(overall_score * 100, 2)}%")
        
        pie_fig = px.pie(
            sentiment_data, 
            values='Percentage', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': Config.PLOT_COLOURS['positive'],
                'Neutral': Config.PLOT_COLOURS['neutral'], 
                'Negative': Config.PLOT_COLOURS['negative']
            },
            title=f'Sentiment Distribution for {stock_symbol}'
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        
        fig = analyser.plot_results(sentiment_data)
        st.pyplot(fig)
        
        st.subheader("Sentiment Breakdown")
        for _, row in sentiment_data.iterrows():
            st.metric(row['Sentiment'], f"{row['Score']:.2f}", f"{row['Percentage']:.1f}%")

if __name__ == "__main__":
    main()
