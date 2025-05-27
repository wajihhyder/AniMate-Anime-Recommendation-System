import pandas as pd
import os
import requests
from tqdm import tqdm

# Load image data
anime =  pd.read_csv('data/anime-dataset-2023.csv')

# Create a folder for local images
image_dir = 'static/images/anime'
os.makedirs(image_dir, exist_ok=True)

# Download images
for _, row in tqdm(anime.iterrows(), total=len(anime)):
    anime_id = row['anime_id']
    image_url = row['Image URL']
    image_path = os.path.join(image_dir, f"{anime_id}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(image_path):
        continue
    
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: {anime_id}")
    except Exception as e:
        print(f"Error downloading {anime_id}: {e}")
