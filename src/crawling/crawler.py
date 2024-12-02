from newspaper import Article
import pandas as pd

error_dict = {}
error_dict["errors"] = 0
error_dict["total"] = 0

def get_article(url, error_dict=error_dict):
    try:
        article = Article(url)
        article.download()
        article.parse()
        error_dict["total"] += 1
        print(article.text)
        return article.text

    except:
        error_dict["total"]  += 1
        error_dict["errors"] += 1
        print(f"<ERROR> {error_dict["errors"]} / {error_dict["total"]} ({error_dict["errors"]/error_dict["total"]})")
        return "<ERROR>"
        
    
df = pd.read_csv("articles_data.csv").head(100)

df["full_article"] = df["url"].apply(get_article)

df.to_csv('articles_data_enriched.csv', index=False)
