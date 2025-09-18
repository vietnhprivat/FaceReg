import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def get_all_links(url):
    """ Function written by chat"""
    options = Options()
    options.add_argument("--headless")   # run without showing browser
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")  # big viewport helps lazy-loading

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # keep scrolling until no more content loads
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # wait for lazy-loaded content
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # collect all <a> tags
        links = [a.get_attribute("href") for a in driver.find_elements(By.TAG_NAME, "a")]
        links = [link for link in links if link]

        return links

    finally:
        driver.quit()

def extract_photo_ids(urls):
    pattern = re.compile(
        r"^https://www\.flickr\.com/photos/spartabilleder/(\d+)/in/album-72177720328960708$"
    )

    photo_ids = []
    for url in urls:
        match = pattern.match(url)
        if match:
            photo_ids.append(match.group(1))

    return photo_ids

if __name__ == "__main__":
    # Get all links for each picture on all pages 
    photoIDs = []
    for n in range(1,20):
        url = f"https://www.flickr.com/photos/spartabilleder/albums/72177720328960708/page{n}"
        page = get_all_links(url)
        links = list(set(extract_photo_ids(page)))
        photoIDs.extend(links)
        
        print(f'links: {len(links)}', f"page: {n}")
    print(len(photoIDs))
        
    # Save all ids
    with open("./data/webpages/all_pages.json", "w") as f:
        json.dump(photoIDs, f)
    

