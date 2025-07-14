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
    """Optimized race URL collection with minimal delays"""
    race_urls = []
    
    try:
        profile_url = f"https://www.gbgb.org.uk/greyhound-profile/?greyhoundId={dog_id}"
        driver.get(profile_url)
        
        # Quick cookie handling
        try:
            cookie_buttons = driver.find_elements(By.CSS_SELECTOR, "button.consent-btn, button.accept-cookies")
            if cookie_buttons:
                driver.execute_script("arguments[0].click();", cookie_buttons[0])
                time.sleep(0.5)
        except:
            pass
        
        # Set page size quickly
        try:
            from selenium.webdriver.support.ui import Select
            select = Select(driver.find_element(By.CSS_SELECTOR, ".GreyhoundProfile__pageSize select"))
            select.select_by_visible_text("100")
            time.sleep(1)
        except:
            pass
        
        def get_total_pages():
            soup = BeautifulSoup(driver.page_source, "html.parser")
            pages = soup.select(".LiveResultsPagination__page")
            page_numbers = [int(p.text.strip()) for p in pages if p.text.strip().isdigit()]
            return max(page_numbers) if page_numbers else 1
        
        total_pages = get_total_pages()
        
        for page in range(1, total_pages + 1):
            if page > 1:
                page_buttons = driver.find_elements(By.CSS_SELECTOR, ".LiveResultsPagination__page")
                for btn in page_buttons:
                    if btn.text.strip() == str(page):
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(1)
                        break
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            race_rows = soup.select(".GreyhoundRow")
            
            for row in race_rows:
                race_link = row.select_one("a[href*='meeting']")
                if race_link and race_link.get('href'):
                    href = race_link['href']
                    if href.startswith('http'):
                        full_url = href
                    else:
                        full_url = "https://www.gbgb.org.uk" + href
                    
                    meeting_match = re.search(r'meetingId=(\d+)', full_url)
                    race_match = re.search(r'raceId=(\d+)', full_url)
                    meeting_id = meeting_match.group(1) if meeting_match else ""
                    race_id = race_match.group(1) if race_match else ""
                    
                    race_date = row.select_one(".GreyhoundRow__date")
                    race_info = {
                        'race_url': full_url,
                        'meeting_id': meeting_id,
                        'race_id': race_id,
                        'race_date': race_date.text.strip() if race_date else ""
                    }
                    
                    race_urls.append(race_info)
        
    except Exception:
        pass
    
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
    # Test with single dog first
    dog_ids = [str(i) for i in range(600000, 600010)]
    start_time = time.time()
    
    total_records = fast_scrape_multiple_dogs(dog_ids, "dogs3.csv", batch_size=25)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Average: {elapsed/len(dog_ids):.1f} seconds per dog")
