# Sentiment Analysis

from typing import List, Optional, Tuple, Dict, Set
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class Results:
  total_positive = 0
  total_neutral = 0
  total_negative = 0
  score = 0
  headline_count = 0
  headlines = []
  analysis_complete = False

class Config:
  # Sentiment Analysis settings
  stock_symbol = 'AAPL'

  MIN_HEADLINE_LENGTH = 5

  SENTIMENT_WEIGHTS = {
      'positive': 1.0,
      'neutral': 0.0,
      'negative': -1.0
  }

  USE_CNBC = True
  USE_YF = True
  USE_QUARTERLY_REVIEW = True if stock_symbol == 'AAPL' else False # only currently working for Apple due to API limitations for the scope of this project

  # Visualization settings
  PLOT_COLOURS = {
      'positive': '#2ecc71',
      'neutral': '#74B9FF',
      'negative': '#e74c3c'
  }

  FIGURE_SIZE = (10, 6)

class SentimentAnalyser:
  def __init__(self, stock_symbol: str):
    self.config = Config()
    self.config.stock_symbol = stock_symbol  # Update the stock symbol in config
    self.stock_symbol = stock_symbol
    # Initialize sentiment values
    self.total_positive = 0
    self.total_neutral = 0
    self.total_negative = 0
    # Load FinBERT model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

  def execute(self):
    scraper = NewsScraper(self.stock_symbol)
    unique_headlines = scraper.get_all_headlines()
    if len(unique_headlines) == 0:
      return
   
    # Check if we have any headlines to analyze
    if not unique_headlines:
      print("No headlines found to analyze. Using default company info for analysis.")
      # Add a default headline about the company to ensure we have something to analyze
      unique_headlines = {f"Information about {self.stock_symbol} stock"}
    
    result = self.analyse_sentiment(unique_headlines)

    # Only plot results if we have data
    if hasattr(self, 'total_positive') and self.total_positive + self.total_neutral + self.total_negative > 0:
      self.plot_results()
    else:
      print("Not enough data to visualize results")

    print('Overall sentiment score')
    print('Positive' if result > 0 else 'Negative')
    print(f'{round(result * 100, 2)}%') # Output score to 2dp
    # 100% is a purely positive score
    # -100% is a purely negative score
    
    return result  # Add missing return statement


  # Analyses sentiment with FinBERT
  def analyse_sentiment(self, headlines: Set[str]) -> float:
    overall_score = 0
    if not headlines:
      return
    num_headlines = len(headlines)

    print('Starting analysis...')
    self.total_positive, self.total_neutral, self.total_negative = 0, 0, 0
    for headline in headlines:
      if len(headline) < self.config.MIN_HEADLINE_LENGTH: # ignores text that is likely to be invalid
        continue

      print(headline, end = " ")
      inputs = self.tokenizer(headline, return_tensors="pt")
      outputs = self.model(**inputs)
      probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
      positive_prob, neutral_prob, negative_prob = probabilities[0].item(), probabilities[1].item(), probabilities[2].item()

      self.total_positive += probabilities[0].item()
      self.total_neutral += probabilities[1].item()
      self.total_negative += probabilities[2].item()

      # Weighted score: positive = +1, neutral = 0, negative = -1
      overall_score += (
          positive_prob * self.config.SENTIMENT_WEIGHTS['positive'] +
          neutral_prob * self.config.SENTIMENT_WEIGHTS['neutral'] +
          negative_prob * self.config.SENTIMENT_WEIGHTS['negative']
      )
      print('| Score:', overall_score)

    print()
    # Calculate weighted average sentiment score
    weighted_avg_score = overall_score / num_headlines if num_headlines > 0 else 0

    return weighted_avg_score


  def plot_results(self):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)

    # Prepare data
    data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Score': [self.total_positive, self.total_neutral, self.total_negative],
    })

    total = data['Score'].sum()
    data['Percentage'] = data['Score'] / total * 100

    colours = [self.config.PLOT_COLOURS['positive'], self.config.PLOT_COLOURS['neutral'], self.config.PLOT_COLOURS['negative']]

    bars = ax.bar(data['Sentiment'], data['Score'], color=colours, alpha=0.7, width=0.6, edgecolor='white', linewidth=2)

    for bar in bars:
        x = bar.get_x()
        w = bar.get_width()
        h = bar.get_height()
        ax.add_patch(Rectangle((x, 0), w, h, facecolor='black', alpha=0.05, zorder=0))

    title = f'Sentiment Analysis Results for {self.stock_symbol}'
    subtitle = f'Based on {len(data)} sentiment categories'

    ax.text(0.5, 1.1, title,
            ha='center', va='bottom',
            transform=ax.transAxes,
            fontsize=16, fontweight='bold', fontfamily='sans-serif', color='#2D3436')

    ax.text(0.5, 1.05, subtitle,
            ha='center', va='bottom',
            transform=ax.transAxes,
            fontsize=12, fontweight='normal',
            fontfamily='sans-serif', color='#636E72')

    for i, (bar, percentage) in enumerate(zip(bars, data['Percentage'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}\n({percentage:.1f}%)',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2D3436')

        ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                f'{data["Sentiment"].iloc[i]}',
                ha='center', va='bottom',
                fontsize=10, color='#B2BEC3', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlabel('Sentiment Category', fontsize=12, labelpad=15, color='#636E72')
    ax.set_ylabel('Sentiment Score', fontsize=12, labelpad=15, color='#636E72')

    max_sentiment = data.loc[data['Score'].idxmax(), 'Sentiment']
    insight_text = (f"Dominant Sentiment: {max_sentiment}\n"
                    f"Total Score: {total:.2f}")
    plt.figtext(0.95, 0.15, insight_text,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=1'),
                ha='right', va='bottom', fontsize=10, color='#636E72')
    plt.figtext(0.95, 0.05, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d")}',
                ha='right', va='bottom', fontsize=8, alpha=0.5)

    plt.tight_layout()

    plt.show()

    # Print summary
    print(f'\nSentiment Analysis Summary for {self.stock_symbol}:')
    print(f'Dominant Sentiment: {max_sentiment}')
    for sentiment, score, percentage in zip(data['Sentiment'], data['Score'], data['Percentage']):
        print(f'{sentiment}: {score:.2f} ({percentage:.1f}%)')


class NewsScraper:
  def __init__(self, stock_symbol: str):
    self.config = Config()
    self.config.stock_symbol = stock_symbol  # Update the stock symbol in config
    self.stock_symbol = stock_symbol
    
    # Set default company name values in case API fails
    self.short_name = stock_symbol
    self.short_name_alt = stock_symbol
    
    # Try to get better company name but handle failure gracefully
    try:
      result = self.get_company_short_name(stock_symbol)
      if result:
        self.short_name, self.short_name_alt = result
    except Exception as e:
      print(f"Warning: Could not get company name details: {e}")
      print(f"Using stock symbol {stock_symbol} as company name")

  def get_all_headlines(self) -> Set[str]:
    # Even if short_name is None, we'll still try to get headlines
    # using just the stock symbol
    headlines = []
    
    if self.config.USE_CNBC:
      try:
        cnbc_headlines = self.get_cnbc_headlines()
        headlines += cnbc_headlines
        print(f"CNBC headlines found: {len(cnbc_headlines)}")
      except Exception as e:
        print(f"Error getting CNBC headlines: {e}")
    
    if self.config.USE_YF:
      try:
        yf_headlines = self.get_yf_headlines()
        headlines += yf_headlines
        print(f"Yahoo Finance headlines found: {len(yf_headlines)}")
      except Exception as e:
        print(f"Error getting Yahoo Finance headlines: {e}")

    if self.config.USE_QUARTERLY_REVIEW and self.stock_symbol == 'AAPL':
      try:
        text = self.get_quarterly_reivews()
        if text:  # Add check to ensure text is not None
          statements = text.split('. ')
          headlines += statements
          print(f"Quarterly review statements found: {len(statements)}")
      except Exception as e:
        print(f"Error getting quarterly review: {e}")

    # If no headlines were found, add a default one to prevent errors
    if not headlines:
      print("No headlines found from any source.")
      return []

    unique_headlines = set(headlines)  # Removes the duplicate headlines
    return unique_headlines


  def get_company_short_name(self, symbol: str) -> Optional[Tuple[str, str]]:
    """Get company name from stock symbol with retry logic."""
    MAX_RETRIES = 2
    RETRY_DELAY = 2  # seconds
    
    for attempt in range(MAX_RETRIES):
      try:
        # Try a simpler, more reliable approach first
        import time
        if attempt > 0:
          print(f"Retrying company name lookup (attempt {attempt+1}/{MAX_RETRIES})...")
          time.sleep(RETRY_DELAY)
        
        # Create a fresh ticker instance each time
        stock = yf.Ticker(symbol)
        
        # Try to get company name from different possible attributes
        if hasattr(stock, 'info') and stock.info and 'shortName' in stock.info:
          short_name = stock.info['shortName'].split()[0].strip(',')
          return short_name, short_name.strip(".")
        elif hasattr(stock, 'info') and stock.info and 'longName' in stock.info:
          long_name = stock.info['longName'].split()[0].strip(',')
          return long_name, long_name.strip(".")
          
      except Exception as e:
        print(f"Attempt {attempt+1} failed: {str(e)}")
    
    # If all retries fail, use a hardcoded mapping for common symbols
    common_companies = {
      'AAPL': ('Apple', 'Apple'),
      'MSFT': ('Microsoft', 'Microsoft'),
      'GOOGL': ('Google', 'Google'),
      'AMZN': ('Amazon', 'Amazon'),
      'META': ('Meta', 'Meta'),
      'TSLA': ('Tesla', 'Tesla'),
      'NVDA': ('NVIDIA', 'NVIDIA'),
    }
    
    if symbol in common_companies:
      print(f"Using pre-defined company name for {symbol}")
      return common_companies[symbol]
      
    # Fallback: just use the symbol itself
    print(f"Using symbol {symbol} as company name")
    return symbol, symbol


  def get_cnbc_headlines(self) -> List[str]:
    headlines = []
    url = f"https://www.cnbc.com/quotes/{self.stock_symbol}?tab=news"

    try:
      print(f"Attempting to fetch news from: {url}")
      response = requests.get(url, timeout=10)
      soup = BeautifulSoup(response.text, 'html.parser')
      to_analyse = soup.find(class_="QuotePageTabs")
      print(f"\nFound QuotePageTabs section: {bool(to_analyse)}")

      if to_analyse:
        headline_elements = []
        article_headlines = to_analyse.find_all(['h3', 'a'], class_=lambda x: x and 'headline' in x.lower())
        headline_elements.extend(article_headlines)
        print(f"\nFound {len(headline_elements)} potential headline elements")

        # Process found elements
        for element in headline_elements:
          # Try multiple ways to get the headline text
          headline = None
          if element.get('title'):
            headline = element['title']
          elif element.text.strip():
            headline = element.text.strip()
          elif element.get('aria-label'):
            headline = element['aria-label']

          if headline:
            print(f"\nFound potential headline: {headline}")
            # Check if headline contains stock symbol or company name
            if any(substring.lower() in headline.lower() 
                  for substring in [self.short_name, self.short_name_alt, self.stock_symbol] 
                  if substring):  # Check that substring is not None
              if len(headline) >= self.config.MIN_HEADLINE_LENGTH:
                headlines.append(headline)
                print(f"Added headline: {headline}")

      headlines = list(dict.fromkeys(headlines)) # Remove duplicates

      # Output results
      if headlines:
        print('\nSuccessfully retrieved headlines from CNBC:')
        for idx, headline in enumerate(headlines, 1):
          print(f"{idx}. {headline}")
      else:
        print('\nNo headlines found matching the search criteria')

    except requests.ConnectionError:
      print("Error: Failed to connect to CNBC website")
    except requests.Timeout:
      print("Error: Request timed out")
    except requests.RequestException as e:
      print(f"Error during request: {str(e)}")
    except Exception as e:
      print(f"Unexpected error occurred: {str(e)}")

    print()
    return headlines


  def get_yf_headlines(self) -> List[str]:
    headlines = []
    try:
      stock = yf.Ticker(self.stock_symbol)
      if stock.news:
        for entry in stock.news:
          # Try to extract title from different possible keys
          title = None
          
          if 'title' in entry:
            title = entry['title']
          elif 'headline' in entry:
            title = entry['headline']
          elif 'description' in entry:
            title = entry['description']
          elif 'content' in entry and isinstance(entry['content'], dict) and 'title' in entry['content']:
            title = entry['content']['title']

          if title and self.short_name:
            # Check if the title contains the stock symbol or company short name
            if any(name.lower() in title.lower() 
                  for name in [self.short_name, self.short_name_alt, self.stock_symbol] 
                  if name):  # Check that name is not None
              headlines.append(title)

      if len(headlines) > 0:
        print('Successfully retrieved headlines from Yahoo Finance')
        for headline in headlines:
          print(headline)
      else:
        print('No relevant headlines found from Yahoo Finance')
    except Exception as e:
      print(f'Error retrieving Yahoo Finance headlines: {str(e)}')

    print()
    return headlines


  def get_quarterly_reivews(self) -> Optional[str]:  # Fix return type
    # Quarterly review analysis for Apple
    if self.stock_symbol != 'AAPL':
      return None
      
    url = "https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/"
    try:
      response = requests.get(url, timeout=15)  # Add timeout

      if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        content_blocks = soup.find_all('div', class_='component-content')
        article_text = "\n".join([block.get_text(separator="\n").strip() for block in content_blocks])
        
        # Safe extraction of relevant part
        start_idx = article_text.find('CALIFORNIA')
        if start_idx > -1:
          article_text = article_text[start_idx+11:]
        
        end_idx = article_text.find('Apple will provide live streaming of its Q4')
        if end_idx > -1:
          article_text = article_text[:end_idx]
          
        cleaned_text = article_text.replace("\n", " ")
        return cleaned_text
      else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None
    except Exception as e:
      print(f"Error fetching quarterly review: {str(e)}")
      return None


def main():
  try:
    config = Config()
    
    # Allow user to specify stock symbol
    import sys
    if len(sys.argv) > 1:
      stock_symbol = sys.argv[1].upper()
      print(f"Analyzing sentiment for: {stock_symbol}")
    else:
      stock_symbol = config.stock_symbol
      print(f"No stock symbol provided, using default: {stock_symbol}")
      
    analyser = SentimentAnalyser(stock_symbol)
    result = analyser.execute()  # Capture the result
    
    if result is not None:
      print(f"Analysis complete. Overall sentiment score: {result:.4f}")
    else:
      print("Analysis could not be completed.")
      
  except Exception as e:
    print(f"An error occurred during execution: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
  main()
