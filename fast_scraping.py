import csv
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os

<<<<<<< HEAD:fast_scraping.py
API_BASE = "https://api.gbgb.org.uk/api/results/dog"

# CSV fields exactly as in API 'items'
CSV_FIELDS = [
    "dogId",  # Add this first - we'll inject it manually
    "dogName",  # Add missing dog name
    "SP",
    "resultPosition",
    "resultBtnDistance",
    "resultSectionalTime",
    "resultComment",
    "resultRunTime",
    "resultDogWeight",
    "winnerOr2ndName",
    "winnerOr2ndId",
    "resultAdjustedTime",
    "trapNumber",
    "raceTime",
    "raceDate",
    "raceId",
    "raceNumber",  # Keep if exists in API
    "raceType",
    "raceClass",
    "raceDistance",
    "raceGoing",
    "raceWinTime",
    "meetingId",
    "trackName",
    "trainerName",  # Keep if exists in API
    "ownerName"    # Keep if exists in API
]


def fetch_items(dog_id, per_page=1000):
    """Fetch up to 'per_page' items in one request."""
    url = f"{API_BASE}/{dog_id}"
    params = {"page": 1, "itemsPerPage": per_page}
    resp = requests.get(url, params=params)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])


def normalize_item(item, dog_id):
    """Extract fields from API item and add dog_id"""
    record = {field: item.get(field, "") for field in CSV_FIELDS}
    # Override dogId since it's not in the API response
    record["dogId"] = dog_id
    return record


def save_to_csv(records, filename="dogs5.csv", header=False):
    if not records:
        return 0  # Return 0 instead of None
    df = pd.DataFrame(records, columns=CSV_FIELDS)
    df.to_csv(filename, mode="a", index=False, header=header)
    return len(records)  # Return the count of records saved


def get_existing_dog_ids(filename):
    """Retrieve existing dog IDs from the CSV file."""
    if not os.path.exists(filename):
        return set()
    df = pd.read_csv(filename, usecols=["dogId"])
    return set(df["dogId"].astype(str))


def main(start_id=600000, end_id=6000100, output_file="dogs5.csv"):
    """Smart append mode - only scrapes new dogs"""
    print(f"🚀 SMART APPEND MODE: Scraping dogs {start_id} to {end_id}")
    print(f"📂 Output file: {output_file}")
=======
def create_fast_driver():
    """Create optimized Chrome driver for speed"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
>>>>>>> 693537a0d7ed315ff49e848e7adfe5ef2e6d38e1:scraping/fast_scraping.py
    
    service = Service()
    return webdriver.Chrome(service=service, options=chrome_options)

def fast_scrape_multiple_dogs(dog_ids, output_file="dogs3.csv", batch_size=50):
    """Optimized scraping with minimal delays and batch processing"""
    
    # Load existing data
    processed_races = set()
    all_race_data = []
    
    if os.path.exists(output_file):
        print(f"Loading existing data from {output_file}...")
        with open(output_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_race_data.append(row)
                race_key = f"{row.get('meeting_id', '')}_{row.get('race_id', '')}"
                processed_races.add(race_key)
        print(f"Loaded {len(all_race_data)} existing records, {len(processed_races)} unique races")

    driver = create_fast_driver()
    wait = WebDriverWait(driver, 5)  # Shorter timeout
    
    try:
        # Collect all race URLs first
        print("Collecting race URLs from all dogs...")
        all_race_urls = set()
        
        for i, dog_id in enumerate(dog_ids, 1):
            print(f"Processing dog {i}/{len(dog_ids)}: {dog_id}")
            race_urls = fast_get_dog_race_urls(driver, dog_id, wait)
            for race_info in race_urls:
                race_tuple = (
                    race_info['race_url'],
                    race_info['meeting_id'], 
                    race_info['race_id'],
                    race_info['race_date']
                )
                all_race_urls.add(race_tuple)
        
        print(f"Found {len(all_race_urls)} unique races to process")
        
        # Process races in batches
        new_races = []
        for race_tuple in all_race_urls:
            race_info = {
                'race_url': race_tuple[0],
                'meeting_id': race_tuple[1],
                'race_id': race_tuple[2],
                'race_date': race_tuple[3]
            }
            
            race_key = f"{race_info['meeting_id']}_{race_info['race_id']}"
            if race_key not in processed_races:
                new_races.append(race_info)
        
        print(f"Need to scrape {len(new_races)} new races")
        
        # Process new races
        for i, race_info in enumerate(new_races, 1):
            if i % 10 == 0:  # Progress every 10 races
                print(f"Processed {i}/{len(new_races)} new races")
            
            try:
                driver.get(race_info['race_url'])
                # Minimal wait for page load
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".MeetingRaceTrap")))
                
                race_data = extract_complete_race_data(driver, race_info)
                if race_data:
                    all_race_data.extend(race_data)
                    
                    # Save in batches to avoid data loss
                    if i % batch_size == 0:
                        save_comprehensive_data(all_race_data, output_file)
                        print(f"Saved batch at race {i}")
                        
            except Exception as e:
                print(f"Error processing race {race_info['race_id']}: {str(e)[:50]}")
                continue
        
        # Final save
        save_comprehensive_data(all_race_data, output_file)
        print(f"Completed! Total records: {len(all_race_data)}")
        
    finally:
        driver.quit()
    
    return len(all_race_data)

def fast_get_dog_race_urls(driver, dog_id, wait):
    """Optimized race URL collection with minimal delays - FIXED VERSION"""
    race_urls = []
    
    try:
        profile_url = f"https://www.gbgb.org.uk/greyhound-profile/?greyhoundId={dog_id}"
        driver.get(profile_url)
        
        # Wait for page to load
        time.sleep(2)
        
        # Check if dog exists
        if "No greyhound found" in driver.page_source or "No results found" in driver.page_source:
            return race_urls
        
        # Quick cookie handling
        try:
            cookie_buttons = driver.find_elements(By.CSS_SELECTOR, "button.consent-btn, button.accept-cookies, .cookie-consent-btn")
            if cookie_buttons:
                driver.execute_script("arguments[0].click();", cookie_buttons[0])
                time.sleep(0.5)
        except:
            pass
        
        # Try to set page size to maximum
        try:
            from selenium.webdriver.support.ui import Select
            select_elems = driver.find_elements(By.CSS_SELECTOR, "select")
            for select_elem in select_elems:
                try:
                    select = Select(select_elem)
                    options = [option.text for option in select.options]
                    for size in ["All", "100", "50"]:
                        if size in options:
                            select.select_by_visible_text(size)
                            time.sleep(2)
                            break
                    break
                except:
                    continue
        except:
            pass
        
        def extract_races_from_page():
            """Extract races from current page"""
            page_races = []
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Method 1: Look for standard race rows
            race_rows = soup.select(".GreyhoundRow, tr")
            for row in race_rows:
                race_link = row.select_one("a[href*='meeting'], a[href*='race']")
                if race_link and race_link.get('href'):
                    href = race_link['href']
                    if not href.startswith('http'):
                        href = "https://www.gbgb.org.uk" + href
                    
                    meeting_match = re.search(r'meetingId=(\d+)', href)
                    race_match = re.search(r'raceId=(\d+)', href)
                    
                    if meeting_match and race_match:
                        meeting_id = meeting_match.group(1)
                        race_id = race_match.group(1)
                        
                        # Extract date
                        race_date = ""
                        date_elem = row.select_one(".GreyhoundRow__date, .date, td:first-child")
                        if date_elem:
                            race_date = date_elem.get_text(strip=True)
                        
                        race_info = {
                            'race_url': href,
                            'meeting_id': meeting_id,
                            'race_id': race_id,
                            'race_date': race_date
                        }
                        
                        # Avoid duplicates
                        race_key = f"{meeting_id}_{race_id}"
                        if not any(f"{r['meeting_id']}_{r['race_id']}" == race_key for r in page_races):
                            page_races.append(race_info)
            
            # Method 2: If no races found, try all links
            if not page_races:
                all_links = soup.select("a[href*='meeting'], a[href*='race']")
                for link in all_links:
                    href = link.get('href', '')
                    if href and ('meetingId' in href or 'raceId' in href):
                        if not href.startswith('http'):
                            href = "https://www.gbgb.org.uk" + href
                        
                        meeting_match = re.search(r'meetingId=(\d+)', href)
                        race_match = re.search(r'raceId=(\d+)', href)
                        
                        if meeting_match and race_match:
                            race_info = {
                                'race_url': href,
                                'meeting_id': meeting_match.group(1),
                                'race_id': race_match.group(1),
                                'race_date': ""
                            }
                            
                            race_key = f"{race_info['meeting_id']}_{race_info['race_id']}"
                            if not any(f"{r['meeting_id']}_{r['race_id']}" == race_key for r in page_races):
                                page_races.append(race_info)
            
            return page_races
        
        # Extract races from first page
        race_urls.extend(extract_races_from_page())
        
        # Check for pagination and process additional pages
        max_pages = 10  # Reasonable limit
        current_page = 1
        
        while current_page < max_pages:
            # Look for next page button or page numbers
            next_buttons = driver.find_elements(By.CSS_SELECTOR, 
                ".next-page, .pagination-next, a.next, [aria-label='Next page']")
            
            page_buttons = driver.find_elements(By.CSS_SELECTOR, 
                ".pagination li, .pagination-item, .page-link")
            
            next_page_found = False
            
            # Try clicking next button
            for btn in next_buttons:
                try:
                    if btn.is_enabled() and btn.is_displayed():
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(2)
                        next_page_found = True
                        break
                except:
                    continue
            
            # If no next button, try clicking page number
            if not next_page_found:
                target_page = current_page + 1
                for btn in page_buttons:
                    try:
                        if btn.text.strip() == str(target_page):
                            driver.execute_script("arguments[0].click();", btn)
                            time.sleep(2)
                            next_page_found = True
                            break
                    except:
                        continue
            
            if not next_page_found:
                break
            
            # Extract races from this page
            new_races = extract_races_from_page()
            if new_races:
                race_urls.extend(new_races)
                current_page += 1
            else:
                break
        
    except Exception as e:
        print(f"Error processing dog {dog_id}: {str(e)}")
    
    return race_urls

def extract_complete_race_data(driver, race_info):
    """Extract all dogs' data from a race page"""
    try:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        race_date_elem = soup.select_one(".Meeting__header__date")
        race_date = race_date_elem.text.strip() if race_date_elem else race_info.get('race_date', '')
        
        track_elem = soup.select_one(".Meeting__header__title__meta")
        track = track_elem.text.strip() if track_elem else ""
        
        race_time = ""
        race_class = ""
        race_distance = ""
        race_prizes = ""
        
        race_header = soup.select_one(".MeetingRace__header")
        if race_header:
            time_elem = race_header.select_one(".MeetingRace__time")
            if time_elem:
                race_time = time_elem.text.strip()
            
            class_elem = race_header.select_one(".MeetingRace__class")
            if class_elem:
                race_class = class_elem.text.strip().replace("|", "").strip()
            
            distance_elem = race_header.select_one(".MeetingRace__distance")
            if distance_elem:
                race_distance = distance_elem.text.strip()
                
            prizes_elem = race_header.select_one(".MeetingRace__prizes")
            if prizes_elem:
                race_prizes = prizes_elem.text.strip()
        
        dogs_data = []
        dog_rows = soup.select(".MeetingRaceTrap")
        
        for dog_row in dog_rows:
            dog_data = extract_single_dog_data(dog_row)
            if dog_data:
                dog_data.update({
                    'meeting_id': race_info['meeting_id'],
                    'race_id': race_info['race_id'],
                    'race_url': race_info['race_url'],
                    'race_date': race_date,
                    'track': track,
                    'race_time': race_time,
                    'race_class': race_class,
                    'race_distance': race_distance,
                    'race_prizes': race_prizes
                })
                dogs_data.append(dog_data)
        
        return dogs_data
        
    except Exception:
        return []

def extract_single_dog_data(dog_row):
    """Extract data for a single dog from its row"""
    try:
        data = {}
        
        pos_elem = dog_row.select_one(".MeetingRaceTrap__pos")
        data['position'] = pos_elem.text.strip() if pos_elem else ""
        
        greyhound_elem = dog_row.select_one(".MeetingRaceTrap__greyhound")
        data['dog_name'] = greyhound_elem.text.strip() if greyhound_elem else ""
        
        if greyhound_elem and greyhound_elem.get('href'):
            href = greyhound_elem['href']
            dog_id_match = re.search(r'greyhoundId=(\d+)', href)
            data['dog_id'] = dog_id_match.group(1) if dog_id_match else ""
        else:
            data['dog_id'] = ""
        
        trainer_elem = dog_row.select_one(".MeetingRaceTrap__trainer")
        data['trainer'] = trainer_elem.text.strip() if trainer_elem else ""
        
        comment_elem = dog_row.select_one(".MeetingRaceTrap__comment")
        data['comments'] = comment_elem.text.strip() if comment_elem else ""
        
        sp_elem = dog_row.select_one(".MeetingRaceTrap__sp")
        data['starting_price'] = sp_elem.text.strip() if sp_elem else ""
        
        time_s_elem = dog_row.select_one(".MeetingRaceTrap__timeS")
        data['time_s'] = time_s_elem.text.strip() if time_s_elem else ""
        
        time_dist_elem = dog_row.select_one(".MeetingRaceTrap__timeDistance")
        data['time_distance'] = time_dist_elem.text.strip() if time_dist_elem else ""
        
        trap_elem = dog_row.select_one(".MeetingRaceTrap__trap img")
        if trap_elem and trap_elem.get('src'):
            trap_match = re.search(r'icn-(\d+)', trap_elem['src'])
            data['trap'] = trap_match.group(1) if trap_match else ""
        else:
            data['trap'] = ""
        
        profile_elem = dog_row.select_one(".MeetingRaceTrap__houndProfile")
        if profile_elem:
            profile_text = profile_elem.text.strip()
            data['breeding_info'] = profile_text
            
            parts = [p.strip() for p in profile_text.split('|')]
            
            if len(parts) >= 1:
                data['birth_date'] = parts[0]
            if len(parts) >= 2:
                data['weight'] = parts[1]
            if len(parts) >= 3:
                data['color'] = parts[2]
            if len(parts) >= 4:
                parents = parts[3]
                if ' - ' in parents:
                    sire, dam = parents.split(' - ', 1)
                    data['sire'] = sire.strip()
                    data['dam'] = dam.strip()
                else:
                    data['sire'] = ""
                    data['dam'] = ""
        else:
            data['breeding_info'] = ""
            data['birth_date'] = ""
            data['weight'] = ""
            data['color'] = ""
            data['sire'] = ""
            data['dam'] = ""
        
        return data
        
    except Exception:
        return {}

def save_comprehensive_data(all_race_data, filename="dogs3.csv"):
    """Save comprehensive race data to CSV sorted by race_id"""
    if not all_race_data:
        return
    
    all_race_data.sort(key=lambda x: int(x.get('race_id', 0)) if x.get('race_id', '').isdigit() else 0)
    
    fieldnames = [
        'meeting_id', 'race_id', 'race_date', 'track', 'race_time', 'race_class', 
        'race_distance', 'race_prizes', 'position', 'trap', 'dog_id', 'dog_name', 
        'trainer', 'comments', 'starting_price', 'time_s', 'time_distance',
        'birth_date', 'weight', 'color', 'sire', 'dam', 'breeding_info', 'race_url'
    ]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for race in all_race_data:
            writer.writerow(race)

if __name__ == "__main__":
    # Test with a small range first
    print("Testing fast scraping...")
    dog_ids = ["637322", "637323", "637324"]  # Start with known IDs
    start_time = time.time()
    
    total_records = fast_scrape_multiple_dogs(dog_ids, "dogs3.csv", batch_size=10)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Average: {elapsed/len(dog_ids):.1f} seconds per dog")
    print(f"Total records scraped: {total_records}")
