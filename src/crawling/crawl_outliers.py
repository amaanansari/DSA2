import pandas as pd
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_fresh_driver():
    user_agent = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    options = Options()
    options.add_argument('--headless')
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions')
    driver = webdriver.Chrome(options=options)
    return driver

driver = get_fresh_driver()
error_dict = {"errors": 0, "total": 0}

def get_article(url, error_dict=error_dict):
    try:
        driver.get(url)
        html = driver.page_source
        article = Article(url)
        article.set_html(html)
        article.parse()
        error_dict["total"] += 1
        print("Article: ", article.text)
        return article.text
    except Exception as e:
        error_dict["total"]  += 1
        error_dict["errors"] += 1
        print(f"<ERROR> {error_dict['errors']} / {error_dict['total']} ({error_dict['errors']/error_dict['total']:.2f}): {str(e)}")
        return f"<ERROR: {type(e).__name__}>"

# Read the combined articles CSV
df = pd.read_csv("combined_articles.csv")

# Use na=False to handle NaN values in 'full_article'
error_rows = df[df['full_article'].str.contains('<ERROR:', na=True)].copy()
print(error_rows)

# Reprocess the articles that previously had errors
error_rows['full_article'] = error_rows['url'].apply(get_article)

# Save the reprocessed articles to 'rest_articles.csv'
error_rows.to_csv('rest_articles.csv', index=False)

# Close the Selenium driver
driver.quit()
