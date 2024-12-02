import os
import pandas as pd
from natsort import natsorted

# Directory where all article files are saved
directory = "articles"

# Get all files starting with "articles_" and ending with ".csv", in natural sorted order
files = natsorted([f for f in os.listdir(directory) if f.startswith("articles_") and f.endswith(".csv")])

# Combine all files into a single DataFrame
combined_df = pd.concat([pd.read_csv(os.path.join(directory, file)) for file in files], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_articles.csv", index=False)
print(f"Combined {len(files)} files into 'combined_articles.csv'")
