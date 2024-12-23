# Analyzing Engagement, Misinformation, and Political Bias in News

This repository contains the code and data used in our project analyzing the relationship between user engagement, misinformation, and political bias in news articles. The project investigates how misinformation influences engagement, how political bias affects news consumption, and how the interplay between these elements shapes public perception. Our analysis leverages a comprehensive pipeline that integrates data extraction, sentiment analysis, bias scoring, and misinformation classification.


## How to Run the Code

### 1. Setup Environment

Ensure you have Python 3.11 installed. Use the following steps to set up the environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration

To properly run the code, create a `.env` file in the root directory and configure the following API keys and endpoints:

```
LAMBDA_API_KEY=your_lambda_api_key
LAMBDA_API_BASE=https://api.lambda.com/v1
LAMBDA_API_MODEL=your_model_name
OPENAI_API_KEY=your_openai_api_key
```

- **LAMBDA_API_KEY** – API key for accessing Lambda models.
- **LAMBDA_API_BASE** – Base URL for the Lambda API.
- **LAMBDA_API_MODEL** – Model version to be used (e.g., `gpt-3.5-turbo`).
- **OPENAI_API_KEY** – API key for OpenAI’s GPT models, used as a fallback or alternative model.

### 3. Data Collection and Crawling

The `src/crawling` folder contains scripts for extracting article content:
- **crawler.py** – Scrapes article content from URLs using Newspaper3k.
- **crawl_outliers.py** – Re-scrapes articles that could not be retrieved initially using Selenium.
- **crawler_caching.py** – Uses a headless Selenium driver to extract blocked articles and caches the results.
- **combine.py** – Merges scraped data into a consolidated CSV.
- **dataloader.py** – Loads and preprocesses article data.

Run:
```bash
python src/crawling/crawler.py
python src/crawling/crawl_outliers.py
python src/crawling/crawler_caching.py
```

### 4. Fake News Detection and Classification

- **classifier.py** – Applies a large language model (LLM) to classify articles as fake or not fake.
- **detector.py** – Retrieves relevant Wikipedia passages to assist the LLM in verification.
- **util.py** – Initializes the language model (Llama or OpenAI GPT) and manages API calls.

Run the classifier:
```bash
python classifier.py
```

### 5. Analysis and Visualization

- **analysis.ipynb** – Jupyter notebook for sentiment analysis, engagement comparison, and visualization.

Launch the notebooks with:
```bash
jupyter notebook
```

## Key Components

- **Sentiment Analysis**: Conducted using VADER and TextBlob to evaluate article tone and polarity.
- **Political Bias Scoring**: Bias metrics are derived from AllSides media bias ratings.
- **Engagement Metrics**: Engagement data is normalized using historical engagement from Social Blade.
- **Fake News Classification**: Llama3.1-70b-instruct is used with Wikipedia passage retrieval to verify article claims. The classification pipeline leverages the ColBERT model to fetch relevant Wikipedia abstracts to improve accuracy.

## Contributions
Contributors: Amaan Ansari, Christopher Rau, Ria Doshi