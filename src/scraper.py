import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import time
import random

#### Constants ####

USER_AGENTS = [
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36",

    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
]

#### Helper Functions ####

def construct_images_size_url(photo_id: str, size: str = "l") -> str:
    return f"https://www.flickr.com/photos/spartabilleder/{photo_id}/sizes/{size}/"

def get_download_url(size_url: str) -> str | None:
    response = requests.get(size_url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {size_url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    download_links = [
        a["href"] for a in soup.find_all("a", href=True)
        if "photo_download.gne" in a["href"]
    ]

    if len(download_links) > 1:
        print(f" Multiple download links for {size_url}")
    return download_links[0] if download_links else None

def save_image(photo_id: str, url: str, folder="data/output"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": random.choice(USER_AGENTS),  
        "Referer": "https://www.flickr.com/"      
    }

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        filepath = folder / f"{photo_id}.jpg"
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        # polite pause
        time.sleep(0.5)

        return None

    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return True
    
#### Workflow Functions ####

def build_download_dict(photo_ids, size="l"):
    download_dict = {}
    for pid in tqdm(photo_ids, desc="Fetching download links"):
        size_url = construct_images_size_url(pid, size)
        dl_url = get_download_url(size_url)
        if dl_url:
            download_dict[pid] = dl_url
    return download_dict

def save_download_dict(download_dict, path="data/webpages/all_download_urls.json"):
    with open(path, "w") as f:
        json.dump(download_dict, f, indent=2)

def load_download_df(path="data/webpages/all_download_urls.json"):
    with open(path, "r") as f:
        download_dict = json.load(f)
    return pd.DataFrame(
        list(download_dict.items()), 
        columns=["id", "download_url"]
    )

#### Main pipeline ####

if __name__ == "__main__":
    # Load collected photoIDs
    with open("data/webpages/all_pages.json", "r") as f:
        photoIDs = json.load(f)

    # Step 1: build {photoID: download_url} dict
    download_dict = build_download_dict(photoIDs)

    # Step 2: save dictionary
    save_download_dict(download_dict, "data/webpages/all_download_urls.json")

    # Step 3: load into dataframe
    df = load_download_df()

    # Pick a slice, e.g., start at row 400
    subset = df.iloc[420+526:]  

    # Step 4: download from dataframe
    count = 0
    for pid, url in tqdm(zip(subset["id"], subset["download_url"]), 
                        total=len(subset), 
                        desc="Downloading images"):
        count +=1
        img = save_image(pid, url, folder="data/output")
        if img == True:
            print(count)
            break
