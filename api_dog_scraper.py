import requests
import csv
import json
import time
import os
import argparse
import pandas as pd
from datetime import datetime

def fetch_dog_data(dog_id, page=1, items_per_page=100):
    """Fetch data for a single dog from the GBGB API with pagination support"""
    url = f"https://api.gbgb.org.uk/api/results/dog/{dog_id}"
    
    # Include pagination parameters
    params = {
        "page": page,
        "itemsPerPage": items_per_page
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Dog doesn't exist - not an error, just return None
            return None
        else:
            print(f"Error fetching dog {dog_id}: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception when fetching dog {dog_id}: {str(e)}")
        return None

def fetch_all_race_data(dog_id, debug=False):
    """Fetch all race data for a dog with pagination support"""
    page = 1
    items_per_page = 100
    all_race_data = []
    dog_info = None
    
    while True:
        if debug and page == 1:
            print(f"Fetching page {page} for dog {dog_id}...")
        
        # Fetch data for current page
        data = fetch_dog_data(dog_id, page, items_per_page)
        
        if not data:
            if page == 1:
                # Dog doesn't exist or error on first page
                return None, []
            else:
                # No more pages
                break
        
        # Store dog info from first page
        if page == 1:
            dog_info = data
            
            # Print debug info for first page
            if debug:
                print(f"Dog {dog_id} name: {data.get('name', 'Unknown')}")
                # Display available keys in response
                print(f"API response keys: {list(data.keys())}")
        
        # Extract race data (formLines)
        form_lines = data.get('formLines', [])
        
        if debug and page == 1:
            print(f"Found {len(form_lines)} race records on page {page}")
            if form_lines:
                print(f"Sample race data: {json.dumps(form_lines[0], indent=2)[:500]}...")
        
        # Add to our collection
        all_race_data.extend(form_lines)
        
        # Check if we've reached the end of data
        if len(form_lines) < items_per_page:
            break
            
        # Move to next page
        page += 1
        
        # Be nice to the API - small delay between pages
        time.sleep(0.1)
    
    if debug:
        print(f"Total race records found for dog {dog_id}: {len(all_race_data)}")
    
    return dog_info, all_race_data

def process_dog_data(dog_info, race_data, dog_id):
    """Process the JSON response into a list of race records"""
    records = []
    
    # If API didn't return valid data
    if not dog_info or not race_data:
        return records
    
    # Extract dog info from the response
    dog_name = dog_info.get('name', 'Unknown')
    
    # Process each race
    for race in race_data:
        # Create base record with dog info
        record = {
            'dog_id': dog_id,
            'dog_name': dog_name,
            'date_of_birth': dog_info.get('dateOfBirth', ''),
            'sex': dog_info.get('sex', ''),
            'color': dog_info.get('color', ''),
            'sire': dog_info.get('sire', {}).get('name', '') if isinstance(dog_info.get('sire'), dict) else dog_info.get('sire', ''),
            'dam': dog_info.get('dam', {}).get('name', '') if isinstance(dog_info.get('dam'), dict) else dog_info.get('dam', ''),
        }
        
        # Handle track data which might be a string or a dict
        track_value = race.get('track', '')
        if isinstance(track_value, dict):
            track_name = track_value.get('name', '')
        else:
            track_name = str(track_value)
        
        # Handle other fields that might be objects or strings
        def extract_name(obj_or_str):
            if isinstance(obj_or_str, dict):
                return obj_or_str.get('name', '')
            return str(obj_or_str) if obj_or_str else ''
        
        # Add race details
        record.update({
            'meeting_id': race.get('meetingId', ''),
            'race_id': race.get('raceId', ''),
            'race_date': race.get('date', ''),
            'track': track_name,
            'race_time': race.get('time', ''),
            'race_distance': race.get('distance', ''),
            'race_grade': race.get('grade', ''),
            'position': race.get('position', ''),
            'trap': race.get('trap', ''),
            'starting_price': race.get('sp', ''),
            'remarks': race.get('remarks', ''),
            'prize_money': race.get('prizeMoney', ''),
            'time_sec': race.get('calculatedTime', ''),
            'forecasted_time': race.get('forecastedTime', ''),
            'sectional_time': race.get('sectionalTime', ''),
            'winning_time': race.get('winningTime', ''),
            'going': race.get('going', ''),
            'weight': race.get('weight', ''),
            'winner_name': extract_name(race.get('winner', '')),
            'winning_trainer': extract_name(race.get('winningTrainer', '')),
            'trainer': extract_name(race.get('trainer', '')),
        })
        
        records.append(record)
    
    return records

def save_to_csv(records, output_file="dogs4.csv", append=True):
    """Save records to CSV file, with option to append"""
    if not records:
        return 0
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    mode = 'a' if append and os.path.exists(output_file) else 'w'
    header = not (append and os.path.exists(output_file))
    
    df.to_csv(output_file, mode=mode, header=header, index=False)
    
    return len(df)

def scrape_dogs_via_api(start_id, end_id, output_file="dogs4.csv", batch_size=50, verbose=False):
    """Scrape multiple dogs via the API and save to CSV with pagination support"""
    total_records = 0
    processed_dogs = 0
    successful_dogs = 0
    missing_dogs = 0
    empty_dogs = 0
    batch_records = []
    
    start_time = time.time()
    
    # Test one dog ID to verify API works properly
    if verbose:
        test_id = "600001"  # Use a known ID that should have races
        print(f"Testing API with dog ID {test_id}...")
        
        dog_info, race_data = fetch_all_race_data(test_id, debug=True)
        
        if dog_info:
            print(f"✅ Found dog {test_id}: {dog_info.get('name', 'Unknown')} with {len(race_data)} races")
        else:
            print(f"❌ Couldn't find test dog {test_id}")
    
    for dog_id in range(start_id, end_id + 1):
        try:
            str_dog_id = str(dog_id)
            
            # Progress indicator
            if processed_dogs % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_dogs / elapsed if elapsed > 0 else 0
                eta = (end_id - dog_id) / rate if rate > 0 else 0
                print(f"Processing dog {dog_id} ({processed_dogs}/{end_id-start_id+1} - {rate:.1f} dogs/sec, ETA: {eta/60:.1f} min)")
                print(f"Success: {successful_dogs}, Empty: {empty_dogs}, Missing: {missing_dogs}, Records: {total_records}")
            
            # Fetch data from API with pagination support
            dog_info, race_data = fetch_all_race_data(str_dog_id, debug=(dog_id == start_id and verbose))
            
            if dog_info:
                if race_data:
                    # Process into records
                    records = process_dog_data(dog_info, race_data, str_dog_id)
                    
                    if records:
                        # Add to batch
                        batch_records.extend(records)
                        total_records += len(records)
                        successful_dogs += 1
                        
                        if processed_dogs % 10 == 0 or verbose:
                            print(f"  ✓ Dog {dog_id} ({dog_info.get('name', 'Unknown')}): Added {len(records)} race records")
                    else:
                        empty_dogs += 1
                        if processed_dogs % 10 == 0 or verbose:
                            print(f"  ⚠️ Dog {dog_id} ({dog_info.get('name', 'Unknown')}): Processed but no valid records created")
                else:
                    empty_dogs += 1
                    if processed_dogs % 10 == 0 or verbose:
                        print(f"  ⚠️ Dog {dog_id} ({dog_info.get('name', 'Unknown')}): Exists but has no race records")
            else:
                # Dog doesn't exist (404) or other error
                missing_dogs += 1
                if processed_dogs % 20 == 0 and verbose:
                    print(f"  ⏭️ Dog {dog_id}: Missing or invalid (skipped)")
            
            processed_dogs += 1
            
            # Save batch when it reaches the batch size
            if len(batch_records) >= batch_size:
                save_to_csv(batch_records, output_file=output_file)
                print(f"💾 Saved batch of {len(batch_records)} records (Total: {total_records})")
                batch_records = []
            
            # Be nice to the API - small delay between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing dog {dog_id}: {str(e)}")
    
    # Save any remaining records
    if batch_records:
        save_to_csv(batch_records, output_file=output_file)
        print(f"💾 Saved final batch of {len(batch_records)} records")
    
    elapsed_time = time.time() - start_time
    dogs_per_second = processed_dogs / elapsed_time if elapsed_time > 0 else 0
    
    print("\nScraping Completed!")
    print(f"Time: {elapsed_time:.1f} seconds")
    print(f"Dogs processed: {processed_dogs}")
    print(f"Successful dogs: {successful_dogs}")
    print(f"Dogs with no races: {empty_dogs}")
    print(f"Missing dogs: {missing_dogs}")
    print(f"Total race records: {total_records}")
    print(f"Performance: {dogs_per_second:.2f} dogs/second")
    
    # Warning if no records found
    if total_records == 0:
        print(f"\n⚠️ WARNING: No records were found in the selected ID range ({start_id}-{end_id})")
        print(f"Try using the --find option to locate dogs with race data")
    
    return total_records

def find_dogs_with_races(start_id=600000, end_id=601000, sample_size=10, verbose=True):
    """Find dogs that exist and have race records within a range"""
    found_dogs = []
    total_tested = 0
    exist_count = 0
    with_races_count = 0
    
    # Try to find sample_size dogs with races
    while len(found_dogs) < sample_size and total_tested < min(100, (end_id - start_id)):
        # Pick dogs at equal intervals across the range
        interval = max(1, (end_id - start_id) // sample_size)
        
        for i in range(sample_size):
            dog_id = start_id + (i * interval)
            if dog_id > end_id:
                break
                
            if verbose:
                print(f"Testing dog ID {dog_id}...")
            
            total_tested += 1
            dog_info, race_data = fetch_all_race_data(str(dog_id))
            
            if dog_info:
                exist_count += 1
                
                if race_data:
                    with_races_count += 1
                    found_dogs.append({
                        'dog_id': dog_id,
                        'name': dog_info.get('name', 'Unknown'),
                        'races': len(race_data)
                    })
                    
                    if verbose:
                        print(f"✓ Dog {dog_id} ({dog_info.get('name', 'Unknown')}) has {len(race_data)} races")
                else:
                    if verbose:
                        print(f"✗ Dog {dog_id} exists but has no races")
            else:
                if verbose:
                    print(f"✗ Dog {dog_id} does not exist")
    
    print(f"\nDog Search Results:")
    print(f"- Tested: {total_tested} dog IDs")
    print(f"- Dogs that exist: {exist_count}")
    print(f"- Dogs with races: {with_races_count}")
    
    if found_dogs:
        print("\nRecommended dogs to scrape:")
        for dog in found_dogs:
            print(f"Dog ID: {dog['dog_id']} - {dog['name']} ({dog['races']} races)")
    else:
        print("\nNo dogs with races found in the tested range.")
    
    return found_dogs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape dog racing data from GBGB API')
    parser.add_argument('--start', type=int, default=600000, help='Starting dog ID')
    parser.add_argument('--end', type=int, default=600100, help='Ending dog ID')
    parser.add_argument('--output', type=str, default="dogs4.csv", help='Output CSV file')
    parser.add_argument('--batch', type=int, default=50, help='Batch size for saving')
    parser.add_argument('--find', action='store_true', help='Find dogs with race data in the range')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    print(f"Starting API scraper for dogs {args.start} to {args.end}")
    print(f"Output file: {args.output}")
    print("=" * 50)
    
    if args.find:
        print("Finding dogs with race data in the specified range...")
        good_dogs = find_dogs_with_races(args.start, args.end, sample_size=5, verbose=True)
        
        if good_dogs and input("\nScrape these dogs? (y/n): ").lower() == 'y':
            # Calculate a range that includes all found dogs
            min_id = min(dog['dog_id'] for dog in good_dogs)
            max_id = max(dog['dog_id'] for dog in good_dogs)
            
            # Add a small buffer
            buffer = 5
            range_start = max(min_id - buffer, args.start)
            range_end = min(max_id + buffer, args.end)
            
            print(f"\nScraping range {range_start} to {range_end}...")
            scrape_dogs_via_api(range_start, range_end, output_file=args.output, batch_size=args.batch, verbose=args.verbose)
    else:
        # Run the scraper directly
        scrape_dogs_via_api(args.start, args.end, output_file=args.output, batch_size=args.batch, verbose=args.verbose)
