import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import os
from functools import lru_cache
import asyncio
#import aiohttp
import multiprocessing as mp

# OPTIMIZED CONFIGURATION
USE_TEST_SAMPLE = False
TEST_SAMPLE_SIZE = 100
SAVE_PROGRESS_EVERY = 100  # Save less frequently for better performance
MAX_WORKERS = 20  # Concurrent threads for API calls
BATCH_SIZE = 50  # Process dogs in batches
API_TIMEOUT = 5  # Reduced timeout
MAX_RETRIES = 2  # Reduced retries for speed
RATE_LIMIT_DELAY = 0.05  # Minimal delay between requests


# Global session for connection pooling
session = None
session_lock = threading.Lock()

def get_session():
    """Get or create a global requests session for connection pooling"""
    global session
    with session_lock:
        if session is None:
            session = requests.Session()
            # Configure session for high performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=50,
                pool_maxsize=50,
                max_retries=1
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            })
        return session

@lru_cache(maxsize=10000)
def fetch_meeting_data_cached(meeting_id):
    """Cached version of fetch_meeting_data to avoid duplicate API calls"""
    return fetch_meeting_data_raw(meeting_id)

def fetch_meeting_data_raw(meeting_id):
    """Fetch meeting data from GBGB API with optimized performance"""
    url = f"https://api.gbgb.org.uk/api/results/meeting/{meeting_id}"
    
    try:
        response = get_session().get(url, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def extract_dog_info_from_meeting_fast(meeting_data, target_dog_id):
    """Optimized version of extract_dog_info_from_meeting"""
    if not meeting_data or len(meeting_data) == 0:
        return None
    
    meeting = meeting_data[0]
    races = meeting.get('races', [])
    
    # Use list comprehension for faster iteration
    for race in races:
        for trap in race.get('traps', []):
            if trap.get('dogId') == target_dog_id:
                return {
                    'dogId': target_dog_id,
                    'dogName': trap.get('dogName', ''),
                    'dogSire': trap.get('dogSire', ''),
                    'dogDam': trap.get('dogDam', ''),
                    'dogBorn': trap.get('dogBorn', ''),
                    'dogColour': trap.get('dogColour', ''),
                    'dogSex': trap.get('dogSex', ''),
                    'trainerName': trap.get('trainerName', ''),
                    'ownerName': trap.get('ownerName', ''),
                    'meetingId': meeting.get('meetingId'),
                    'trackName': meeting.get('trackName', '')
                }
    return None

def process_single_dog(dog_data):
    """Process a single dog with all its meetings - optimized for threading"""
    dog_id, meeting_ids = dog_data
    
    # Try only the first meeting for speed - if name not found, try up to 2 more
    for i, meeting_id in enumerate(meeting_ids[:3]):
        if i > 0:
            time.sleep(RATE_LIMIT_DELAY)  # Very small delay for additional requests
        
        meeting_data = fetch_meeting_data_cached(meeting_id)
        if meeting_data:
            dog_info = extract_dog_info_from_meeting_fast(meeting_data, dog_id)
            if dog_info and dog_info.get('dogName', ''):
                return dog_info
    
    # Return empty record if no info found
    return {
        'dogId': dog_id,
        'dogName': '',
        'dogSire': '',
        'dogDam': '',
        'dogBorn': '',
        'dogColour': '',
        'dogSex': '',
        'trainerName': '',
        'ownerName': '',
        'meetingId': '',
        'trackName': ''
    }

def create_dog_meeting_map_optimized(df):
    """Create optimized mapping of dog_id to meeting_ids"""
    print("🔧 Creating optimized dog-meeting mapping...")
    
    # Use vectorized operations for speed
    dog_meetings = df.groupby('dogId')['meetingId'].apply(list).to_dict()
    
    print(f"✅ Created mapping for {len(dog_meetings)} dogs")
    return dog_meetings

def scrape_dog_names_and_info_optimized(scraped_csv="test_sample.csv", output_csv="dog_info_test.csv"):
    """Ultra-fast version of the dog name scraper using concurrent processing"""
    print("🚀 Starting OPTIMIZED dog name and info scraper...")
    print(f"⚡ Max workers: {MAX_WORKERS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"💾 Auto-save every {SAVE_PROGRESS_EVERY} dogs")
    
    # Load data with optimizations
    print("📊 Loading data with optimizations...")
    df = load_scraped_data(scraped_csv)
    if df is None:
        return None
    
    # Create optimized dog-meeting mapping
    dog_meetings = create_dog_meeting_map_optimized(df)
    total_dogs = len(dog_meetings)
    
    print(f"📊 Processing {total_dogs} unique dogs")
    
    # Load existing dog info
    existing_dog_info = {}
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_dog_info = existing_df.set_index('dogId').to_dict('index')
            print(f"📊 Loaded {len(existing_dog_info)} existing dog records")
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
    
    # Filter out already processed dogs
    dogs_to_process = [
        (dog_id, meetings) for dog_id, meetings in dog_meetings.items()
        if dog_id not in existing_dog_info
    ]
    
    print(f"🎯 Need to process {len(dogs_to_process)} new dogs")
    
    if not dogs_to_process:
        print("✅ All dogs already processed!")
        return existing_dog_info
    
    # Initialize results storage
    all_results = existing_dog_info.copy()
    completed_count = 0
    successful_count = sum(1 for info in existing_dog_info.values() if info.get('dogName', ''))
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("🔍 Starting CONCURRENT dog processing...")
    print("=" * 60)
    
    # Process dogs in batches with concurrent execution
    batch_number = 0
    
    for i in range(0, len(dogs_to_process), BATCH_SIZE):
        batch = dogs_to_process[i:i + BATCH_SIZE]
        batch_number += 1
        
        print(f"\n📦 Processing batch {batch_number}/{(len(dogs_to_process) + BATCH_SIZE - 1) // BATCH_SIZE}")
        print(f"🐕 Dogs in batch: {len(batch)}")
        
        batch_start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all dogs in batch for concurrent processing
            future_to_dog = {
                executor.submit(process_single_dog, dog_data): dog_data[0] 
                for dog_data in batch
            }
            
            # Collect results as they complete
            batch_results = {}
            for future in as_completed(future_to_dog):
                dog_id = future_to_dog[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per dog
                    batch_results[dog_id] = result
                    completed_count += 1
                    
                    if result.get('dogName', ''):
                        successful_count += 1
                        print(f"  ✅ {dog_id}: {result['dogName']}")
                    else:
                        print(f"  ❌ {dog_id}: No name found")
                        
                except Exception as e:
                    print(f"  💥 {dog_id}: Error - {str(e)[:50]}")
                    # Create empty record for failed dogs
                    batch_results[dog_id] = {
                        'dogId': dog_id, 'dogName': '', 'dogSire': '', 'dogDam': '',
                        'dogBorn': '', 'dogColour': '', 'dogSex': '', 'trainerName': '',
                        'ownerName': '', 'meetingId': '', 'trackName': ''
                    }
                    completed_count += 1
        
        # Add batch results to main results
        all_results.update(batch_results)
        
        # Calculate and display batch performance
        batch_time = time.time() - batch_start_time
        batch_speed = len(batch) / batch_time if batch_time > 0 else 0
        total_elapsed = time.time() - start_time
        overall_speed = completed_count / total_elapsed if total_elapsed > 0 else 0
        
        print(f"⚡ Batch completed in {batch_time:.1f}s ({batch_speed:.1f} dogs/sec)")
        print(f"📊 Overall progress: {completed_count}/{len(dogs_to_process)} ({(completed_count/len(dogs_to_process)*100):.1f}%)")
        print(f"✅ Success rate: {(successful_count/completed_count*100):.1f}% ({successful_count} dogs)")
        print(f"🏃 Overall speed: {overall_speed:.1f} dogs/sec")
        
        # Estimate remaining time
        if completed_count > 0:
            remaining_dogs = len(dogs_to_process) - completed_count
            eta_seconds = remaining_dogs / overall_speed if overall_speed > 0 else 0
            eta_minutes = eta_seconds / 60
            print(f"⏰ ETA: {eta_minutes:.1f} minutes remaining")
        
        # Save progress periodically
        if batch_number % (SAVE_PROGRESS_EVERY // BATCH_SIZE + 1) == 0:
            save_results_optimized(all_results, output_csv)
            print(f"💾 Progress saved: {len(all_results)} total dogs")
        
        # Small delay between batches to be nice to the API
        time.sleep(0.1)
    
    # Final save
    save_results_optimized(all_results, output_csv)
    
    total_time = time.time() - start_time
    final_speed = len(dogs_to_process) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 OPTIMIZED SCRAPING COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🐕 Dogs processed: {len(dogs_to_process)}")
    print(f"✅ Dogs with names: {successful_count}")
    print(f"❌ Dogs without names: {len(dogs_to_process) - successful_count}")
    print(f"📈 Success rate: {(successful_count/len(dogs_to_process)*100):.1f}%")
    print(f"⚡ Average speed: {final_speed:.1f} dogs/second")
    print(f"🚀 Performance improvement: ~{final_speed * 5:.0f}x faster than original!")
    print(f"📁 Results saved to: {output_csv}")
    
    return all_results

def save_results_optimized(results_dict, output_csv):
    """Optimized saving of results"""
    if not results_dict:
        return
    
    # Convert to DataFrame efficiently
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # Ensure required columns
    required_columns = ['dogId', 'dogName', 'dogSire', 'dogDam', 'dogBorn', 
                       'dogColour', 'dogSex', 'trainerName', 'ownerName', 
                       'meetingId', 'trackName']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Save with optimizations
    df[required_columns].to_csv(output_csv, index=False)

# Replace the main scraping function call
def scrape_dog_names_and_info(scraped_csv="test_sample.csv", output_csv="dog_info_test.csv"):
    """Wrapper to use the optimized version"""
    return scrape_dog_names_and_info_optimized(scraped_csv, output_csv)

def calculate_estimated_runtime(total_dogs, time_per_dog=5):
    """Calculate estimated runtime based on test performance"""
    total_seconds = total_dogs * time_per_dog
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    return {
        'total_dogs': total_dogs,
        'time_per_dog': time_per_dog,
        'total_seconds': total_seconds,
        'total_minutes': total_minutes,
        'total_hours': total_hours
    }

def display_runtime_estimate(total_dogs):
    """Display runtime estimation before starting"""
    print("⏱️  RUNTIME ESTIMATION")
    print("=" * 50)
    
    # Based on test results: ~5 seconds per dog (including API calls)
    estimate = calculate_estimated_runtime(total_dogs, time_per_dog=5)
    
    print(f"📊 Dogs to process: {estimate['total_dogs']:,}")
    print(f"⚡ Estimated time per dog: {estimate['time_per_dog']} seconds")
    print()
    print("🕐 Estimated Total Runtime:")
    print(f"  ⏱️  {estimate['total_seconds']:.0f} seconds")
    print(f"  ⏱️  {estimate['total_minutes']:.1f} minutes")
    print(f"  ⏱️  {estimate['total_hours']:.2f} hours")
    print()
    
    # Show range estimates
    best_case = calculate_estimated_runtime(total_dogs, time_per_dog=3)
    worst_case = calculate_estimated_runtime(total_dogs, time_per_dog=8)
    
    print("📈 Range Estimates:")
    print(f"  🟢 Best case: {best_case['total_hours']:.2f} hours")
    print(f"  🔴 Worst case: {worst_case['total_hours']:.2f} hours")
    print()
    
    if estimate['total_hours'] <= 1:
        print("✅ Short runtime - perfect for testing!")
    elif estimate['total_hours'] <= 4:
        print("⚠️  Medium runtime - consider running during free time")
    else:
        print("🔴 Long runtime - consider running overnight")
    
    return estimate

def get_progress_tracker(total_dogs, start_time):
    """Create a progress tracking function"""
    def update_progress(completed_dogs, successful_dogs):
        elapsed = time.time() - start_time
        
        if completed_dogs > 0:
            avg_time_per_dog = elapsed / completed_dogs
            remaining_dogs = total_dogs - completed_dogs
            eta_seconds = remaining_dogs * avg_time_per_dog
            eta_minutes = eta_seconds / 60
            
            completion_rate = (completed_dogs / total_dogs) * 100
            success_rate = (successful_dogs / completed_dogs) * 100 if completed_dogs > 0 else 0
            
            print(f"📊 Progress: {completed_dogs}/{total_dogs} dogs ({completion_rate:.1f}%)")
            print(f"✅ Success rate: {success_rate:.1f}% ({successful_dogs} dogs found names)")
            ##print(f"⏱️  Speed: {60/avg_time_per_dog:.1f} dogs/minute")
            print(f"⏰ ETA: {eta_minutes:.1f} minutes remaining")
    
    return update_progress

def load_scraped_data(csv_file="dogs5.csv"):
    """Load the scraped data CSV file with robust error handling"""
    try:
        print(f"🔍 Attempting to load {csv_file}...")
        
        # First, try reading normally
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} records from {csv_file}")
            return df
        except Exception as e:
            print(f"⚠️ Normal read failed: {e}")
        
        # Try with error handling for bad lines (pandas >= 1.3.0)
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
            print(f"✅ Loaded {len(df)} records from {csv_file} (skipped bad lines)")
            return df
        except Exception as e:
            print(f"⚠️ Skip bad lines failed: {e}")
        
        # Try with quoting handling
        try:
            df = pd.read_csv(csv_file, quoting=1, error_bad_lines=False, warn_bad_lines=True)
            print(f"✅ Loaded {len(df)} records from {csv_file} (with quoting)")
            return df
        except Exception as e:
            print(f"⚠️ Quoting read failed: {e}")
        
        # Manual line-by-line parsing for problematic files
        print(f"🔧 Attempting manual repair and read...")
        return manual_csv_repair_and_read(csv_file)
        
    except Exception as e:
        print(f"❌ All read attempts failed for {csv_file}: {e}")
        return None

def manual_csv_repair_and_read(csv_file):
    """Manually read and repair CSV file line by line"""
    try:
        import csv
        
        print(f"🔧 Reading {csv_file} line by line...")
        
        # Read the file manually
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Get header
            header = f.readline().strip().split(',')
            expected_fields = len(header)
            print(f"📋 Expected {expected_fields} fields: {header[:5]}...")
            
            # Process data lines
            good_lines = []
            bad_lines = []
            line_number = 1
            
            for line in f:
                line_number += 1
                line = line.strip()
                
                if not line:
                    continue
                
                # Split line into fields
                fields = line.split(',')
                
                if len(fields) == expected_fields:
                    good_lines.append(fields)
                else:
                    # Try to fix common issues
                    fixed_line = fix_csv_line(line, expected_fields)
                    if fixed_line and len(fixed_line) == expected_fields:
                        good_lines.append(fixed_line)
                    else:
                        bad_lines.append((line_number, len(fields), line[:100]))
            
            print(f"✅ Processed {len(good_lines)} good lines")
            print(f"❌ Found {len(bad_lines)} problematic lines")
            
            if bad_lines:
                print(f"🔍 Sample problematic lines:")
                for line_num, field_count, preview in bad_lines[:3]:
                    print(f"  Line {line_num}: {field_count} fields - {preview}...")
            
            # Create DataFrame from good lines
            if good_lines:
                df = pd.DataFrame(good_lines, columns=header)
                print(f"✅ Created DataFrame with {len(df)} records")
                return df
            else:
                print(f"❌ No valid lines found")
                return None
                
    except Exception as e:
        print(f"❌ Manual repair failed: {e}")
        return None

def fix_csv_line(line, expected_fields):
    """Attempt to fix a malformed CSV line"""
    try:
        # Method 1: Handle quoted fields with commas
        import csv
        import io
        
        # Try parsing with CSV module
        reader = csv.reader(io.StringIO(line))
        try:
            fields = next(reader)
            if len(fields) == expected_fields:
                return fields
        except:
            pass
        
        # Method 2: Handle extra commas by merging last fields
        fields = line.split(',')
        if len(fields) > expected_fields:
            # Merge extra fields into the last field
            fixed_fields = fields[:expected_fields-1]
            fixed_fields.append(','.join(fields[expected_fields-1:]))
            return fixed_fields
        
        # Method 3: Handle missing fields by adding empty ones
        elif len(fields) < expected_fields:
            while len(fields) < expected_fields:
                fields.append('')
            return fields
        
        return None
        
    except Exception:
        return None

def get_unique_dog_ids(df):
    """Get unique dog IDs from the dataframe"""
    unique_dogs = df['dogId'].unique()
    print(f"📊 Found {len(unique_dogs)} unique dogs")
    return unique_dogs

def get_meeting_ids_for_dog(df, dog_id):
    """Get all meeting IDs where this dog participated"""
    dog_races = df[df['dogId'] == dog_id]
    meeting_ids = dog_races['meetingId'].unique()
    return meeting_ids

def extract_dog_info_from_meeting(meeting_data, target_dog_id):
    """Extract dog information from meeting data"""
    if not meeting_data or len(meeting_data) == 0:
        return None
    
    meeting = meeting_data[0]  # API returns array with one meeting
    races = meeting.get('races', [])
    
    for race in races:
        traps = race.get('traps', [])
        for trap in traps:
            dog_id = trap.get('dogId')
            if dog_id == target_dog_id:
                return {
                    'dogId': dog_id,
                    'dogName': trap.get('dogName', ''),
                    'dogSire': trap.get('dogSire', ''),
                    'dogDam': trap.get('dogDam', ''),
                    'dogBorn': trap.get('dogBorn', ''),
                    'dogColour': trap.get('dogColour', ''),
                    'dogSex': trap.get('dogSex', ''),
                    'trainerName': trap.get('trainerName', ''),
                    'ownerName': trap.get('ownerName', ''),
                    'meetingId': meeting.get('meetingId'),
                    'trackName': meeting.get('trackName', '')
                }
    
    return None

def create_test_sample(input_csv="data/scraped/scraped_data.csv", output_csv="test_sample.csv", sample_size=None):
    """Create a small test sample from the large scraped data file"""
    if sample_size is None:
        sample_size = TEST_SAMPLE_SIZE
    
    print(f"🧪 Creating test sample of {sample_size} unique dogs...")
    
    try:
        # Read a larger chunk to get diverse dogs
        chunk_size = sample_size * 50  # Read more records to find unique dogs
        df = pd.read_csv(input_csv, nrows=chunk_size)
        
        # Get unique dog IDs and sample from them
        unique_dogs = df['dogId'].unique()
        sample_dogs = unique_dogs[:min(sample_size, len(unique_dogs))]
        
        # Filter data for these dogs
        sample_df = df[df['dogId'].isin(sample_dogs)]
        
        # Save test sample
        sample_df.to_csv(output_csv, index=False)
        
        print(f"✅ Test sample created: {len(sample_df)} records from {len(sample_dogs)} dogs")
        print(f"📁 Saved to: {output_csv}")
        
        # Show sample statistics
        print(f"\n📊 Sample statistics:")
        print(f"  - Records: {len(sample_df)}")
        print(f"  - Unique dogs: {sample_df['dogId'].nunique()}")
        print(f"  - Unique meetings: {sample_df['meetingId'].nunique()}")
        print(f"  - Date range: {sample_df['raceDate'].min()} to {sample_df['raceDate'].max()}")
        
        return sample_df
        
    except Exception as e:
        print(f"❌ Error creating test sample: {e}")
        return None

def create_first_n_dogs_sample(input_csv="dogs5.csv", output_csv="first_1000_dogs.csv", sample_size=1000):
    """Create a sample of the first N unique dogs from the dataset with robust CSV handling"""
    print(f"🎯 Creating sample of first {sample_size} unique dogs...")
    
    try:
        # Read the dataset with robust error handling
        print(f"📊 Loading data from {input_csv}...")
        df = load_scraped_data(input_csv)
        
        if df is None:
            print(f"❌ Could not load data from {input_csv}")
            return None
        
        print(f"📊 Loaded {len(df)} total records")
        
        # Check if dogId column exists
        if 'dogId' not in df.columns:
            print(f"❌ Column 'dogId' not found in {input_csv}")
            print(f"💡 Available columns: {list(df.columns)}")
            return None
        
        # Clean the dogId column - remove any non-numeric values
        print(f"🧹 Cleaning dogId column...")
        original_count = len(df)
        
        # Convert dogId to string first, then filter numeric
        df['dogId'] = df['dogId'].astype(str)
        df = df[df['dogId'].str.isdigit()]
        df['dogId'] = df['dogId'].astype(int)
        
        cleaned_count = len(df)
        if original_count != cleaned_count:
            print(f"🧹 Removed {original_count - cleaned_count} records with invalid dogId")
        
        # Get unique dog IDs in order they appear
        print(f"🔍 Finding first {sample_size} unique dogs...")
        unique_dogs = df['dogId'].drop_duplicates().head(sample_size)
        sample_dog_ids = unique_dogs.tolist()
        
        if len(sample_dog_ids) < sample_size:
            print(f"⚠️ Only found {len(sample_dog_ids)} unique dogs (requested {sample_size})")
        
        print(f"🔍 Selected dog ID range: {min(sample_dog_ids)} to {max(sample_dog_ids)}")
        
        # Filter data for these dogs
        sample_df = df[df['dogId'].isin(sample_dog_ids)]
        
        # Save sample
        sample_df.to_csv(output_csv, index=False)
        
        print(f"✅ Sample created: {len(sample_df)} records from {len(sample_dog_ids)} dogs")
        print(f"📁 Saved to: {output_csv}")
        
        # Show sample statistics
        print(f"\n📊 Sample statistics:")
        print(f"  - Records: {len(sample_df)}")
        print(f"  - Unique dogs: {sample_df['dogId'].nunique()}")
        
        # Only show meeting info if column exists
        if 'meetingId' in sample_df.columns:
            print(f"  - Unique meetings: {sample_df['meetingId'].nunique()}")
        
        # Only show date info if column exists
        if 'raceDate' in sample_df.columns:
            try:
                print(f"  - Date range: {sample_df['raceDate'].min()} to {sample_df['raceDate'].max()}")
            except:
                print(f"  - Date range: Unable to determine")
        
        return sample_df
        
    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        import traceback
        print(f"🔍 Full error trace:")
        traceback.print_exc()
        return None

def scrape_dog_names_and_info(scraped_csv="test_sample.csv", output_csv="dog_info_test.csv"):
    """Main function to scrape dog names and information with runtime tracking"""
    print("🚀 Starting dog name and info scraper...")
    print(f"🧪 Test mode: {USE_TEST_SAMPLE}")
    print(f"💾 Auto-save every {SAVE_PROGRESS_EVERY} dogs")
    
    # Check if input file exists
    if not os.path.exists(scraped_csv):
        print(f"❌ Input file {scraped_csv} not found!")
        print("💡 Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return None
    
    # Load existing scraped data
    df = load_scraped_data(scraped_csv)
    if df is None:
        return
    
    # Get unique dog IDs
    unique_dog_ids = get_unique_dog_ids(df)
    total_dogs = len(unique_dog_ids)
    
    # Display runtime estimation
    estimate = display_runtime_estimate(total_dogs)
    
    # Ask for confirmation if not in test mode
    if not USE_TEST_SAMPLE and total_dogs > 100:
        response = input(f"\n⚠️  This will process {total_dogs} dogs (estimated {estimate['total_hours']:.1f} hours). Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled by user")
            return None
    
    # Load existing dog info if file exists
    existing_dog_info = {}
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            for _, row in existing_df.iterrows():
                existing_dog_info[row['dogId']] = row.to_dict()
            print(f"📊 Loaded {len(existing_dog_info)} existing dog records")
        except Exception as e:
            print(f"⚠️ Error loading existing dog info: {e}")
    
    # Dictionary to store dog information
    dog_info_dict = existing_dog_info.copy()
    
    # Initialize progress tracking
    start_time = time.time()
    update_progress = get_progress_tracker(total_dogs, start_time)
    completed_dogs = 0
    successful_dogs = 0
    
    print("\n" + "=" * 60)
    print("🔍 Starting dog processing...")
    print("=" * 60)
    
    # Process each unique dog ID
    for i, dog_id in enumerate(unique_dog_ids):
        completed_dogs += 1
        
        # Skip if we already have info for this dog
        if dog_id in dog_info_dict:
            if dog_info_dict[dog_id].get('dogName', ''):  # Only count as successful if has name
                successful_dogs += 1
            print(f"⏭️ Dog {dog_id} already processed")
            
            # Show progress every 10 dogs
            if completed_dogs % 10 == 0:
                update_progress(completed_dogs, successful_dogs)
            continue
        
        print(f"\n🐕 Processing dog {dog_id} ({completed_dogs}/{total_dogs})")
        
        # Get meeting IDs for this dog
        meeting_ids = get_meeting_ids_for_dog(df, dog_id)
        print(f"  📅 Found {len(meeting_ids)} meetings")
        
        # Try to get dog info from meetings (limit to first 3 for efficiency)
        dog_info = None
        meetings_tried = 0
        max_meetings_to_try = 3
        
        for meeting_id in meeting_ids[:max_meetings_to_try]:
            meetings_tried += 1
            print(f"  🔍 Checking meeting {meeting_id} ({meetings_tried}/{max_meetings_to_try})...")
            meeting_data = fetch_meeting_data_raw(meeting_id)
            
            if meeting_data:
                dog_info = extract_dog_info_from_meeting(meeting_data, dog_id)
                if dog_info and dog_info.get('dogName', ''):
                    print(f"  ✅ Found: {dog_info['dogName']}")
                    successful_dogs += 1
                    break
                else:
                    print(f"  ⚠️ Dog not found in meeting {meeting_id}")
            else:
                print(f"  ❌ Could not fetch meeting {meeting_id}")
            
            # Small delay to be nice to the API
            time.sleep(0.1)
        
        if dog_info:
            dog_info_dict[dog_id] = dog_info
        else:
            print(f"  ❌ No info found after trying {meetings_tried} meetings")
            # Create empty record so we don't try again
            dog_info_dict[dog_id] = {
                'dogId': dog_id,
                'dogName': '',
                'dogSire': '',
                'dogDam': '',
                'dogBorn': '',
                'dogColour': '',
                'dogSex': '',
                'trainerName': '',
                'ownerName': '',
                'meetingId': '',
                'trackName': ''
            }
        
        # Save progress periodically
        if completed_dogs % SAVE_PROGRESS_EVERY == 0:
            save_dog_info(dog_info_dict, output_csv)
            print(f"💾 Progress saved: {len(dog_info_dict)} dogs")
            update_progress(completed_dogs, successful_dogs)
        
        # Small delay between dogs
        time.sleep(0.1)
    
    # Final save and summary
    save_dog_info(dog_info_dict, output_csv)
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ SCRAPING COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"🐕 Total dogs processed: {completed_dogs}")
    print(f"✅ Dogs with names found: {successful_dogs}")
    print(f"❌ Dogs without names: {completed_dogs - successful_dogs}")
    print(f"📈 Success rate: {(successful_dogs/completed_dogs)*100:.1f}%")
    print(f"⚡ Average speed: {completed_dogs/elapsed_time:.2f} dogs/second")
    print(f"📁 Results saved to: {output_csv}")
    
    return dog_info_dict

def save_dog_info(dog_info_dict, output_csv):
    """Save dog information to CSV"""
    if not dog_info_dict:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(dog_info_dict, orient='index')
    
    # Ensure all required columns exist
    required_columns = ['dogId', 'dogName', 'dogSire', 'dogDam', 'dogBorn', 
                       'dogColour', 'dogSex', 'trainerName', 'ownerName', 
                       'meetingId', 'trackName']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[required_columns]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)

def merge_with_original_data(scraped_csv="test_sample.csv", dog_info_csv="dog_info_test.csv", output_csv="dogs_enhanced_test.csv"):
    """Merge the original scraped data with dog information"""
    print("🔗 Merging original data with dog information...")
    
    try:
        # Check if input files exist
        if not os.path.exists(scraped_csv):
            print(f"❌ Original data file not found: {scraped_csv}")
            return None
            
        if not os.path.exists(dog_info_csv):
            print(f"❌ Dog info file not found: {dog_info_csv}")
            return None
        
        # Load original data with robust error handling
        print(f"📊 Loading original data from {scraped_csv}...")
        original_df = load_scraped_data(scraped_csv)
        if original_df is None:
            print(f"❌ Could not load original data")
            return None
            
        print(f"📊 Loaded {len(original_df)} original records")
        
        # Load dog info
        print(f"📊 Loading dog info from {dog_info_csv}...")
        dog_info_df = pd.read_csv(dog_info_csv)
        print(f"📊 Loaded {len(dog_info_df)} dog info records")
        
        # Ensure dogId columns are the same type
        original_df['dogId'] = original_df['dogId'].astype(str)
        dog_info_df['dogId'] = dog_info_df['dogId'].astype(str)
        
        print(f"🔗 Merging datasets on dogId...")
        
        # Merge on dogId - use left join to keep all original records
        merged_df = original_df.merge(dog_info_df, on='dogId', how='left', suffixes=('', '_new'))
        
        print(f"✅ Merge completed: {len(merged_df)} records")
        
        # Update existing columns with new information where original is empty
        columns_to_update = ['dogName', 'trainerName', 'ownerName']
        
        for col in columns_to_update:
            if col + '_new' in merged_df.columns:
                print(f"🔄 Updating {col} column...")
                
                # Create mask for empty/missing values in original column
                if col in merged_df.columns:
                    # Fill empty values in original column with new values
                    mask = (merged_df[col].isna()) | (merged_df[col] == '') | (merged_df[col] == 'nan')
                    before_count = mask.sum()
                    merged_df.loc[mask, col] = merged_df.loc[mask, col + '_new']
                    print(f"  ✅ Updated {before_count} empty {col} values")
                else:
                    # If original column doesn't exist, create it from new column
                    merged_df[col] = merged_df[col + '_new']
                    print(f"  ✅ Created new {col} column")
                
                # Drop the temporary column
                merged_df.drop(col + '_new', axis=1, inplace=True)
        
        # Add completely new columns that weren't in original data
        new_columns = ['dogSire', 'dogDam', 'dogBorn', 'dogColour', 'dogSex']
        
        for col in new_columns:
            if col in dog_info_df.columns:
                print(f"➕ Adding new column: {col}")
                merged_df[col] = merged_df['dogId'].map(dog_info_df.set_index('dogId')[col])
        
        # Clean up any remaining temporary columns
        temp_columns = [col for col in merged_df.columns if col.endswith('_new')]
        if temp_columns:
            merged_df.drop(temp_columns, axis=1, inplace=True)
            print(f"🧹 Removed {len(temp_columns)} temporary columns")
        
        # Save merged data
        print(f"💾 Saving enhanced data to {output_csv}...")
        merged_df.to_csv(output_csv, index=False)
        
        # Verify the file was created
        if os.path.exists(output_csv):
            file_size = os.path.getsize(output_csv) / 1024  # KB
            print(f"✅ Enhanced data saved successfully!")
            print(f"📊 File: {output_csv}")
            print(f"📊 Records: {len(merged_df)}")
            print(f"📊 Columns: {len(merged_df.columns)}")
            print(f"📊 File size: {file_size:.1f} KB")
            
            # Show enhancement statistics
            dogs_with_names = len(merged_df[merged_df['dogName'].notna() & (merged_df['dogName'] != '')])
            dogs_with_trainers = len(merged_df[merged_df['trainerName'].notna() & (merged_df['trainerName'] != '')])
            dogs_with_sires = len(merged_df[merged_df['dogSire'].notna() & (merged_df['dogSire'] != '')]) if 'dogSire' in merged_df.columns else 0
            
            print(f"\n📈 Enhancement Statistics:")
            print(f"  - Dogs with names: {dogs_with_names}/{len(merged_df)} ({(dogs_with_names/len(merged_df)*100):.1f}%)")
            print(f"  - Dogs with trainers: {dogs_with_trainers}/{len(merged_df)} ({(dogs_with_trainers/len(merged_df)*100):.1f}%)")
            if dogs_with_sires > 0:
                print(f"  - Dogs with sire info: {dogs_with_sires}/{len(merged_df)} ({(dogs_with_sires/len(merged_df)*100):.1f}%)")
            
            # Show sample of enhanced data
            print(f"\n📋 Sample enhanced records:")
            sample_cols = ['dogId', 'dogName', 'trainerName', 'ownerName']
            if 'dogSire' in merged_df.columns:
                sample_cols.append('dogSire')
            available_cols = [col for col in sample_cols if col in merged_df.columns]
            print(merged_df[available_cols].head(3).to_string(index=False))
            
        else:
            print(f"❌ Error: Enhanced file was not created at {output_csv}")
            return None
        
        return merged_df
        
    except Exception as e:
        print(f"❌ Error merging data: {e}")
        import traceback
        print(f"🔍 Full error trace:")
        traceback.print_exc()
        return None



if __name__ == "__main__":
    print("🐕 Dog Name and Info Scraper - 1000 DOG RUN")
    print("=" * 50)
    
    # Configuration display
    print("⚙️ Configuration for 1000 Dog Run:")
    print(f"  - Processing first {TEST_SAMPLE_SIZE} unique dogs")
    print(f"  - Auto-save frequency: every {SAVE_PROGRESS_EVERY} dogs")
    print(f"  - Expected runtime: ~{(TEST_SAMPLE_SIZE * 5) / 3600:.1f} hours")
    print()
    
    # Check if we have the main data file - prioritize dogs5.csv
    possible_files = [
        "dogs5.csv",
        "../dogs5.csv",
        "../data/dogs5.csv",
        "scraped_data.csv",
        "../scraped_data.csv",
        "data/scraped/scraped_data.csv",
        "../data/scraped/scraped_data.csv"
    ]
    
    main_data_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            main_data_file = file_path
            print(f"✅ Found main data file: {main_data_file}")
            
            # Show file info
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"📊 File size: {file_size:.1f} MB")
            except:
                pass
            
            break
    
    if main_data_file:
        # Create sample of first 1000 dogs
        sample_file = "first_1000_dogs.csv"
        first_1000_sample = create_first_n_dogs_sample(main_data_file, sample_file, TEST_SAMPLE_SIZE)
        
        if first_1000_sample is not None:
            print("\n" + "="*60)
            print("🚀 STARTING 1000 DOG NAME SCRAPING...")
            print("="*60)
            
            # Run the name scraper on first 1000 dogs
            dog_info = scrape_dog_names_and_info(sample_file, "dog_info_1000.csv")
            
            if dog_info:
                print("\n" + "="*60)
                print("🔗 MERGING DATA...")
                print("="*60)
                
                # Ensure both input files exist before merging
                if os.path.exists(sample_file) and os.path.exists("dog_info_1000.csv"):
                    enhanced_data = merge_with_original_data(sample_file, "dog_info_1000.csv", "dogs_enhanced_1000.csv")
                    
                    if enhanced_data is not None:
                        print("\n🎯 1000 DOG RUN COMPLETED SUCCESSFULLY!")
                        print("="*60)
                        print("📁 Files created:")
                        print(f"  - {sample_file}: Sample of first 1000 unique dogs")
                        print("  - dog_info_1000.csv: Scraped dog names and info")
                        print("  - dogs_enhanced_1000.csv: Original data enhanced with names")
                        print()
                        
                        # Verify all files exist
                        print("🔍 File verification:")
                        for filename in [sample_file, "dog_info_1000.csv", "dogs_enhanced_1000.csv"]:
                            if os.path.exists(filename):
                                size_kb = os.path.getsize(filename) / 1024
                                print(f"  ✅ {filename} ({size_kb:.1f} KB)")
                            else:
                                print(f"  ❌ {filename} - NOT FOUND")
                        
                        # Final statistics
                        if os.path.exists("dog_info_1000.csv"):
                            try:
                                final_df = pd.read_csv("dog_info_1000.csv")
                                dogs_with_names = len(final_df[final_df['dogName'] != ''])
                                print(f"\n📈 Final Results:")
                                print(f"  - Dogs processed: {len(final_df)}")
                                print(f"  - Dogs with names found: {dogs_with_names}")
                                print(f"  - Success rate: {(dogs_with_names/len(final_df))*100:.1f}%")
                            except Exception as e:
                                print(f"⚠️ Could not read final stats: {e}")
                        
                        print("\n💡 Ready for your evaluation!")
                    else:
                        print("\n❌ Failed to create enhanced dataset")
                else:
                    print(f"\n❌ Missing input files for merging:")
                    print(f"  - {sample_file}: {'✅' if os.path.exists(sample_file) else '❌'}")
                    print(f"  - dog_info_1000.csv: {'✅' if os.path.exists('dog_info_1000.csv') else '❌'}")
            else:
                print("\n❌ Name scraping failed")
        else:
            print("❌ Could not create 1000 dog sample")
            print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
            print("1. Check if the CSV file is corrupted")
            print("2. Try using dogs5.csv instead of scraped_data.csv")
            print("3. Check file permissions")
            print("4. Verify the CSV format is correct")
    else:
        print(f"❌ Main data file not found in any of these locations:")
        for file_path in possible_files:
            print(f"  - {file_path}")
        print("\n💡 Make sure dogs5.csv or scraped_data.csv exists")
        print("🔍 Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                file_size = os.path.getsize(file) / (1024 * 1024)  # MB
                print(f"  - {file} ({file_size:.1f} MB)")

