import csv
import time
import re
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup

def create_driver(headless=True):
    """Create optimized Chrome driver"""
    chrome_options = Options()
    if headless:
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

def extract_dog_races(driver, dog_id, verbose=False):
    """Extract all race URLs for a dog with proper pagination support"""
    race_urls = []
    debug_mode = verbose or (dog_id == "607694")
    
    try:
        profile_url = f"https://www.gbgb.org.uk/greyhound-profile/?greyhoundId={dog_id}"
        if debug_mode:
            print(f"Accessing dog profile: {profile_url}")
        
        driver.get(profile_url)
        time.sleep(2)  # Allow initial page load
        
        # Check if dog exists
        if "No greyhound found" in driver.page_source or "No results found" in driver.page_source:
            if debug_mode:
                print(f"Dog ID {dog_id} not found")
            return race_urls
        
        # Get dog name
        dog_name = "Unknown"
        dog_name_elems = driver.find_elements(By.CSS_SELECTOR, ".GreyhoundProfile__name, h1, .greyhound-name")
        if dog_name_elems:
            dog_name = dog_name_elems[0].text.strip()
            if debug_mode:
                print(f"Processing dog: {dog_name} (ID: {dog_id})")
        
        # Handle cookie consent if present
        try:
            cookie_buttons = driver.find_elements(By.CSS_SELECTOR, "button.consent-btn, button.accept-cookies, .cookie-consent-btn")
            if cookie_buttons:
                driver.execute_script("arguments[0].click();", cookie_buttons[0])
                time.sleep(0.5)
        except:
            pass
        
        # Set page size to maximum to reduce pagination
        try:
            # Find all select elements that might control page size
            select_elems = driver.find_elements(By.CSS_SELECTOR, "select")
            page_size_set = False
            
            for select_elem in select_elems:
                try:
                    select = Select(select_elem)
                    options = [option.text for option in select.options]
                    
                    # Look for typical page size options (100, 50, 25, All)
                    if any(size in options for size in ["100", "50", "25", "All"]):
                        # Try to select the largest option
                        for size in ["All", "100", "50", "25"]:
                            if size in options:
                                select.select_by_visible_text(size)
                                if debug_mode:
                                    print(f"Set page size to {size}")
                                page_size_set = True
                                time.sleep(2)  # Wait for page refresh
                                break
                        
                        if page_size_set:
                            break
                except:
                    continue
        except:
            if debug_mode:
                print("Could not set page size")
        
        # Function to extract races from current page
        def extract_races_from_current_page():
            page_race_urls = []
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            # Extract all links that might be race links
            links = soup.select("a[href*='meeting'], a[href*='race']")
            
            for link in links:
                href = link.get('href', '')
                if href and ('meetingId' in href or 'raceId' in href):
                    if href.startswith('http'):
                        full_url = href
                    else:
                        full_url = "https://www.gbgb.org.uk" + href
                    
                    meeting_match = re.search(r'meetingId=(\d+)', full_url)
                    race_match = re.search(r'raceId=(\d+)', full_url)
                    meeting_id = meeting_match.group(1) if meeting_match else ""
                    race_id = race_match.group(1) if race_match else ""
                    
                    # Find date info nearby
                    race_date = ""
                    row = link.find_parent("tr") or link.find_parent("div")
                    if row:
                        date_elem = row.select_one(".date, .race-date, td:first-child, .race-date-cell")
                        if date_elem:
                            race_date = date_elem.text.strip()
                    
                    # Only add if we have both meeting and race IDs
                    if meeting_id and race_id:
                        race_info = {
                            'race_url': full_url,
                            'meeting_id': meeting_id,
                            'race_id': race_id,
                            'race_date': race_date,
                            'dog_id': dog_id,
                            'dog_name': dog_name
                        }
                        # Check if this race is already in our list (avoid duplicates)
                        race_key = f"{meeting_id}_{race_id}"
                        if not any(f"{r['meeting_id']}_{r['race_id']}" == race_key for r in page_race_urls):
                            page_race_urls.append(race_info)
            
            return page_race_urls
        
        # Extract races from first page
        race_urls.extend(extract_races_from_current_page())
        if debug_mode:
            print(f"Found {len(race_urls)} races on page 1")
        
        # Check if there are pagination links
        pagination_elements = driver.find_elements(By.CSS_SELECTOR, 
            ".pagination li, .pagination-item, a[data-page], .LiveResultsPagination__page")
        
        # Get maximum page number
        max_page = 1
        for elem in pagination_elements:
            try:
                page_num = int(elem.text.strip())
                max_page = max(max_page, page_num)
            except:
                pass
        
        # Alternative method: check for "Page X of Y" text
        if max_page == 1:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            page_patterns = [
                r"Page\s+\d+\s+of\s+(\d+)",
                r"page\s+\d+\s+of\s+(\d+)",
                r"Showing.*?of\s+(\d+)\s+pages"
            ]
            
            for pattern in page_patterns:
                matches = re.search(pattern, page_text)
                if matches:
                    max_page = int(matches.group(1))
                    break
        
        if debug_mode:
            print(f"Detected {max_page} total pages")
        
        # Navigate through each page
        for page in range(2, max_page + 1):
            if debug_mode:
                print(f"Processing page {page}/{max_page}")
            
            # Try to find the page button
            page_found = False
            
            # Method 1: Try clicking page number buttons
            for elem in pagination_elements:
                if elem.text.strip() == str(page):
                    try:
                        driver.execute_script("arguments[0].click();", elem)
                        page_found = True
                        time.sleep(2)  # Wait for page to load
                        break
                    except:
                        continue
            
            # Method 2: Try clicking 'Next' button if page number not found
            if not page_found:
                try:
                    next_buttons = driver.find_elements(By.CSS_SELECTOR, 
                        ".next-page, .pagination-next, a.next, [aria-label='Next page']")
                    if next_buttons:
                        driver.execute_script("arguments[0].click();", next_buttons[0])
                        page_found = True
                        time.sleep(2)  # Wait for page to load
                except:
                    pass
            
            # Method 3: Try direct URL manipulation if needed
            if not page_found:
                try:
                    current_url = driver.current_url
                    # Modify URL to include page parameter
                    if "page=" in current_url:
                        new_url = re.sub(r'page=\d+', f'page={page}', current_url)
                    else:
                        separator = "&" if "?" in current_url else "?"
                        new_url = f"{current_url}{separator}page={page}"
                    
                    driver.get(new_url)
                    page_found = True
                    time.sleep(2)  # Wait for page to load
                except:
                    if debug_mode:
                        print(f"Could not navigate to page {page}")
                    continue
            
            # Extract races from this page
            new_races = extract_races_from_current_page()
            if debug_mode:
                print(f"Found {len(new_races)} races on page {page}")
            
            # Add new races to our list
            race_urls.extend(new_races)
        
        if debug_mode:
            print(f"Total races found for {dog_name}: {len(race_urls)}")
        
        return race_urls
        
    except Exception as e:
        if debug_mode:
            print(f"Error extracting races for dog {dog_id}: {str(e)}")
        return race_urls

def extract_race_details(driver, race_info, verbose=False):
    """Extract detailed information from a single race"""
    try:
        # Navigate to race page
        driver.get(race_info['race_url'])
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".MeetingRaceTrap")))
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Extract race metadata
        race_date_elem = soup.select_one(".Meeting__header__date")
        race_date = race_date_elem.text.strip() if race_date_elem else race_info.get('race_date', '')
        
        track_elem = soup.select_one(".Meeting__header__title__meta")
        track = track_elem.text.strip() if track_elem else ""
        
        # Extract race details
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
        
        # Extract all dogs from race
        dogs_data = []
        dog_rows = soup.select(".MeetingRaceTrap")
        
        for dog_row in dog_rows:
            # Position
            pos_elem = dog_row.select_one(".MeetingRaceTrap__pos")
            position = pos_elem.text.strip() if pos_elem else ""
            
            # Dog name and ID
            greyhound_elem = dog_row.select_one(".MeetingRaceTrap__greyhound")
            dog_name = greyhound_elem.text.strip() if greyhound_elem else ""
            
            dog_id = ""
            if greyhound_elem and greyhound_elem.get('href'):
                href = greyhound_elem['href']
                dog_id_match = re.search(r'greyhoundId=(\d+)', href)
                dog_id = dog_id_match.group(1) if dog_id_match else ""
            
            # Is this the dog we're tracking?
            is_target_dog = (dog_id == race_info['dog_id'])
            
            # Only extract detailed data for our target dog
            if is_target_dog:
                # Trainer
                trainer_elem = dog_row.select_one(".MeetingRaceTrap__trainer")
                trainer = trainer_elem.text.strip() if trainer_elem else ""
                
                # Comments/Remarks
                comment_elem = dog_row.select_one(".MeetingRaceTrap__comment")
                comments = comment_elem.text.strip() if comment_elem else ""
                
                # Starting Price
                sp_elem = dog_row.select_one(".MeetingRaceTrap__sp")
                starting_price = sp_elem.text.strip() if sp_elem else ""
                
                # Time (S)
                time_s_elem = dog_row.select_one(".MeetingRaceTrap__timeS")
                time_s = time_s_elem.text.strip() if time_s_elem else ""
                
                # Time (Distance)
                time_dist_elem = dog_row.select_one(".MeetingRaceTrap__timeDistance")
                time_distance = time_dist_elem.text.strip() if time_dist_elem else ""
                
                # Extract trap number from trap image
                trap = ""
                trap_elem = dog_row.select_one(".MeetingRaceTrap__trap img")
                if trap_elem and trap_elem.get('src'):
                    trap_match = re.search(r'icn-(\d+)', trap_elem['src'])
                    trap = trap_match.group(1) if trap_match else ""
                
                # Extract breeding info
                breeding_info = ""
                birth_date = ""
                weight = ""
                color = ""
                sire = ""
                dam = ""
                
                profile_elem = dog_row.select_one(".MeetingRaceTrap__houndProfile")
                if profile_elem:
                    profile_text = profile_elem.text.strip()
                    breeding_info = profile_text
                    
                    # Parse breeding info: "Oct-2020 | 34.4 | d - bd | Ballymac Best - Ballykett Beauty"
                    parts = [p.strip() for p in profile_text.split('|')]
                    
                    if len(parts) >= 1:
                        birth_date = parts[0]
                    if len(parts) >= 2:
                        weight = parts[1]
                    if len(parts) >= 3:
                        color = parts[2]
                    if len(parts) >= 4:
                        # Extract sire and dam from "Sire - Dam" format
                        parents = parts[3]
                        if ' - ' in parents:
                            sire, dam = parents.split(' - ', 1)
                            sire = sire.strip()
                            dam = dam.strip()
                
                # Create dog data record
                dog_data = {
                    'meeting_id': race_info['meeting_id'],
                    'race_id': race_info['race_id'],
                    'race_url': race_info['race_url'],
                    'race_date': race_date,
                    'track': track,
                    'race_time': race_time,
                    'race_class': race_class,
                    'race_distance': race_distance,
                    'race_prizes': race_prizes,
                    'position': position,
                    'trap': trap,
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'trainer': trainer,
                    'comments': comments,
                    'starting_price': starting_price,
                    'time_s': time_s,
                    'time_distance': time_distance,
                    'birth_date': birth_date,
                    'weight': weight,
                    'color': color,
                    'sire': sire,
                    'dam': dam,
                    'breeding_info': breeding_info
                }
                
                dogs_data.append(dog_data)
                break  # We only need data for our target dog
        
        return dogs_data
        
    except Exception as e:
        if verbose:
            print(f"Error extracting race details: {str(e)}")
        return []

def scrape_dogs(dog_ids, output_file="dogs_data.csv", verbose=False):
    """Scrape race data for multiple dogs"""
    all_dog_data = []
    
    # Load existing data if file exists
    existing_races = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_dog_data.append(row)
                    race_key = f"{row.get('meeting_id', '')}_{row.get('race_id', '')}_{row.get('dog_id', '')}"
                    existing_races.add(race_key)
            if verbose:
                print(f"Loaded {len(all_dog_data)} existing records")
        except Exception as e:
            all_dog_data = []
            existing_races = set()
            if verbose:
                print(f"Error loading existing data: {str(e)}")
    
    # Initialize driver
    driver = create_driver(headless=True)
    
    try:
        for i, dog_id in enumerate(dog_ids):
            print(f"Processing dog {i+1}/{len(dog_ids)}: {dog_id}")
            
            # Extract all race URLs for this dog
            race_urls = extract_dog_races(driver, dog_id, verbose)
            print(f"Found {len(race_urls)} races for dog {dog_id}")
            
            # Process each race
            for j, race_info in enumerate(race_urls):
                # Check if we already have this race data
                race_key = f"{race_info['meeting_id']}_{race_info['race_id']}_{dog_id}"
                if race_key in existing_races:
                    if verbose:
                        print(f"  Skipping already processed race: {race_info['race_id']}")
                    continue
                
                # Progress update
                if j % 5 == 0:
                    print(f"  Processing race {j+1}/{len(race_urls)}")
                
                # Extract detailed data for this race
                race_data = extract_race_details(driver, race_info, verbose)
                
                if race_data:
                    all_dog_data.extend(race_data)
                    existing_races.add(race_key)
                    
                    # Save progress every 10 races
                    if j % 10 == 0:
                        save_data(all_dog_data, output_file)
                
                # Be nice to the server
                time.sleep(0.5)
            
            # Save after each dog
            save_data(all_dog_data, output_file)
            
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    finally:
        driver.quit()
        
    # Final save
    save_data(all_dog_data, output_file)
    
    print(f"Scraping completed! Total records: {len(all_dog_data)}")
    return all_dog_data

def save_data(data, filename):
    """Save data to CSV file"""
    if not data:
        return
    
    fieldnames = [
        'meeting_id', 'race_id', 'race_date', 'track', 'race_time', 'race_class', 
        'race_distance', 'race_prizes', 'position', 'trap', 'dog_id', 'dog_name', 
        'trainer', 'comments', 'starting_price', 'time_s', 'time_distance',
        'birth_date', 'weight', 'color', 'sire', 'dam', 'breeding_info', 'race_url'
    ]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def process_data(input_file="dogs_data.csv", output_file="dogs_processed.csv"):
    """Process raw dog data into features for ML"""
    import pandas as pd
    
    try:
        # Load the raw data
        df = pd.read_csv(input_file)
        
        # Make a copy for processing
        processed_df = df.copy()
        
        # Fill missing values
        processed_df = processed_df.fillna({
            'birth_date': 'Unknown',
            'weight': '0.0',
            'time_s': '0.00',
            'time_distance': '0.00 (0)',
            'trap': '0',
            'position': '0',
            'comments': '',
            'race_distance': '0m'
        })
        
        # Extract distance in meters
        processed_df['distance_meters'] = processed_df['race_distance'].apply(
            lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if isinstance(x, str) and re.search(r'(\d+)', str(x)) else 0
        )
        
        # Convert position to numeric
        processed_df['position_numeric'] = processed_df['position'].apply(
            lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if isinstance(x, str) and re.search(r'(\d+)', str(x)) else 99
        )
        
        # Extract weight as float
        processed_df['weight_kg'] = processed_df['weight'].apply(
            lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) and re.search(r'^\d+\.?\d*$', str(x).replace(',', '.')) else 0.0
        )
        
        # Convert trap to numeric
        processed_df['trap_numeric'] = processed_df['trap'].apply(
            lambda x: int(x) if str(x).isdigit() else 0
        )
        
        # Create "won_race" target variable
        processed_df['won_race'] = (processed_df['position_numeric'] == 1).astype(int)
        
        # Save the processed data
        processed_df.to_csv(output_file, index=False)
        print(f"Data processing completed! Saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

# Run the scraper with the problematic dog ID
if __name__ == "__main__":
    print("Dog Race Scraper")
    print("----------------")
    
    # You can add more dog IDs to this list
    dog_ids = ["607694"]
    
    # Main scraping function
    scrape_dogs(dog_ids, output_file="dogs_data.csv", verbose=True)
    
    # Process the data
    process_data("dogs_data.csv", "dogs_processed.csv")
