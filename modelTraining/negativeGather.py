import time
import requests
from ddgs import DDGS
from pathlib import Path

# Define the output directory for negative dataset
root = Path('./dataset/NotBaifern')

def gather_negatives(category_name, query, max_results=20):
    folder = root / category_name
    folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Scraping {category_name}...")
    
    # Disguise the script as a normal Google Chrome browser to bypass basic bot-blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
            
            for i, result in enumerate(results):
                image_url = result.get('image')
                if not image_url: 
                    continue
                
                try:
                    # Download the image using the same disguised headers
                    img_data = requests.get(image_url, timeout=5, headers=headers).content
                    file_path = folder / f"{i:03d}.jpg"
                    
                    with open(file_path, 'wb') as handler:
                        handler.write(img_data)
                except Exception:
                    # Skip silently if a specific image server times out
                    pass
    except Exception as e:
        print(f"Failed to scrape {category_name} via DuckDuckGo: {e}")
        
    # Pause for 3 seconds between categories to avoid triggering Ratelimit
    time.sleep(3)

# ---- TIER 1: THE ULTIMATE LOOK-ALIKES (60 images) ----
gather_negatives('Papillon', 'papillon dog adult', max_results=20)
gather_negatives('Pomeranian', 'brown pomeranian dog', max_results=20)
gather_negatives('OtherChihuahua', 'long haired chihuahua tri color', max_results=20)


# # ---- TIER 2: OTHER SMALL TOY BREEDS (40 images) ----
gather_negatives('ShihTzu', 'shih tzu dog', max_results=20)
gather_negatives('Pug', 'fawn pug dog', max_results=20)


# ---- TIER 3: DOMESTIC DISTRACTORS (40 images) ----
gather_negatives('Cat', 'domestic shorthair cat sitting', max_results=20)
gather_negatives('Cat2', 'fluffy tabby cat', max_results=20)

print("Scraping complete!")