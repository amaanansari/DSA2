from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os

# Initialize error_dict
error_dict = {"errors": 0, "total": 0}

def get_fresh_driver():
    """
    Get a fresh Chrome driver with a Google Bot User Agent.
    :return: selenium.webdriver.Chrome
    """
    user_agent = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    options = Options()
    options.add_argument('--headless')
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions')
    driver = webdriver.Chrome(options=options)
    return driver

# Initialize the Selenium driver
driver = get_fresh_driver()

def get_article(url, error_dict=error_dict):
    try:
        # Use Selenium to get the fully rendered page
        driver.get(url)
        html = driver.page_source

        # Use newspaper3k to parse the article from the HTML
        article = Article(url)
        article.set_html(html)
        article.parse()
        error_dict["total"] += 1
        print(article.text)
        return article.text

    except Exception as e:
        error_dict["total"]  += 1
        error_dict["errors"] += 1
        print(f"<ERROR> {error_dict['errors']} / {error_dict['total']} ({error_dict['errors']/error_dict['total']:.2f}): {str(e)}")
        return f"<ERROR: {type(e).__name__}>"

# Ensure the /articles directory exists
os.makedirs('articles', exist_ok=True)

# Function to get the next file index
def get_next_file_index():
    files = [f for f in os.listdir('articles') if f.startswith("articles_") and f.endswith(".csv")]
    if not files:
        return 1
    indices = [int(f.split('_')[1].split('.')[0]) for f in files]
    return max(indices) + 1

# Function to load existing articles
def load_existing_articles():
    combined_df = pd.DataFrame()
    files = sorted([f for f in os.listdir('articles') if f.startswith("articles_") and f.endswith(".csv")])
    for file in files:
        df = pd.read_csv(f'articles/{file}')
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# Load the main articles data file
df = pd.read_csv("articles_data.csv")

# Load already processed articles to avoid reprocessing
existing_articles_df = load_existing_articles()
if not existing_articles_df.empty:
    df = df[~df['url'].isin(existing_articles_df['url'])]

batch_size = 100
file_index = get_next_file_index()

# Process articles in batches of 100
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i + batch_size].copy()
    if len(batch_df) == 0:
        break
    batch_df["full_article"] = batch_df["url"].apply(get_article)
    batch_df.to_csv(f'articles/articles_{file_index:04d}.csv', index=False)
    print(f"Saved batch {file_index} with {len(batch_df)} articles.")
    file_index += 1

# Close the Selenium driver after processing
driver.quit()
