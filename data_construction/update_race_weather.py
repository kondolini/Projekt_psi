import os
import sys
import pickle
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Adjust parent path to import project modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race

def get_weather_safe(date_str, time_str, track_name):
    """Safe weather import and call with reduced delay"""
    try:
        from scraping.weather_checker import get_weather
        # Small random delay to avoid bursts
        time.sleep(random.uniform(0.4, 0.6))  # 0.4-0.6s = 1.6-2.5 req/sec
        return get_weather(date_str, time_str, track_name)
    except Exception as e:
        return None

def process_weather_batch(weather_requests, weather_cache, cache_lock):
    """Process a batch of weather requests"""
    results = {}
    
    for weather_key in weather_requests:
        date_str, track_name = weather_key
        
        # Check cache first
        with cache_lock:
            if weather_key in weather_cache:
                results[weather_key] = weather_cache[weather_key]
                continue
        
        # Fetch weather data
        weather_data = get_weather_safe(date_str, "12:00", track_name)
        
        # Update cache
        with cache_lock:
            weather_cache[weather_key] = weather_data
        
        results[weather_key] = weather_data
    
    return results

def update_race_weather(race_output_dir="data/races", max_workers=4, batch_size=20):
    """Update existing races with weather data using parallel processing"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    race_output_dir = os.path.join(project_root, race_output_dir)
    
    # Find all race bucket files
    bucket_files = [f for f in os.listdir(race_output_dir) 
                   if f.startswith('races_bucket_') and f.endswith('.pkl')]
    
    print(f"Found {len(bucket_files)} race bucket files")
    
    # Step 1: Analyze all races and collect unique weather requests
    print("🔍 Analyzing races for weather requirements...")
    unique_weather_requests = set()
    races_needing_weather = []
    
    for bucket_file in tqdm(bucket_files, desc="Analyzing buckets"):
        bucket_path = os.path.join(race_output_dir, bucket_file)
        
        with open(bucket_path, 'rb') as f:
            races_bucket = pickle.load(f)
        
        for storage_key, race in races_bucket.items():
            # Check if race needs weather update
            needs_weather = (
                not race.rainfall_7d or 
                race.rainfall_7d == [0.0] * 7 or
                race.temperature == 50.0 or 
                race.humidity == 15.0
            )
            
            if needs_weather:
                weather_key = (race.race_date.strftime('%Y-%m-%d'), race.track_name)
                unique_weather_requests.add(weather_key)
                races_needing_weather.append((bucket_file, storage_key, weather_key))
    
    print(f"📊 Analysis complete:")
    print(f"  - Races needing weather: {len(races_needing_weather):,}")
    print(f"  - Unique weather requests: {len(unique_weather_requests):,}")
    print(f"  - Estimated time (at 2 req/sec): {len(unique_weather_requests) / 2 / 60:.1f} minutes")
    
    # Step 2: Fetch all unique weather data in parallel
    print(f"🌦️ Fetching weather data with {max_workers} workers...")
    weather_cache = {}
    cache_lock = threading.Lock()
    
    # Create batches of weather requests
    weather_list = list(unique_weather_requests)
    weather_batches = [weather_list[i:i + batch_size] 
                      for i in range(0, len(weather_list), batch_size)]
    
    successful_requests = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all weather batches
        future_to_batch = {
            executor.submit(process_weather_batch, batch, weather_cache, cache_lock): i
            for i, batch in enumerate(weather_batches)
        }
        
        # Collect results
        with tqdm(total=len(weather_batches), desc="Fetching weather", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    successful_requests += sum(1 for data in batch_results.values() if data)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': successful_requests,
                        'cache': len(weather_cache)
                    })
                except Exception as e:
                    print(f"Error in weather batch: {e}")
                    pbar.update(1)
    
    print(f"✅ Weather fetching complete: {successful_requests}/{len(unique_weather_requests)} successful")
    
    # Step 3: Update all races with cached weather data
    print("📝 Updating races with weather data...")
    races_updated = 0
    races_with_weather = 0
    
    # Group races by bucket for efficient processing
    bucket_updates = defaultdict(list)
    for bucket_file, storage_key, weather_key in races_needing_weather:
        bucket_updates[bucket_file].append((storage_key, weather_key))
    
    for bucket_file, race_updates in tqdm(bucket_updates.items(), desc="Updating races"):
        bucket_path = os.path.join(race_output_dir, bucket_file)
        
        # Load bucket
        with open(bucket_path, 'rb') as f:
            races_bucket = pickle.load(f)
        
        bucket_modified = False
        
        # Update races in this bucket
        for storage_key, weather_key in race_updates:
            race = races_bucket[storage_key]
            weather_data = weather_cache.get(weather_key)
            
            if weather_data:
                race.rainfall_7d = weather_data['rainfall_7d']
                race.humidity = weather_data['humidity']
                race.temperature = weather_data['temperature']
                races_with_weather += 1
            else:
                # Set defaults
                race.rainfall_7d = [0.0] * 7
                race.humidity = 50.0
                race.temperature = 15.0
            
            races_updated += 1
            bucket_modified = True
        
        # Save bucket if modified
        if bucket_modified:
            with open(bucket_path, 'wb') as f:
                pickle.dump(races_bucket, f)
    
    print(f"\n🎉 WEATHER UPDATE COMPLETED!")
    print("=" * 50)
    print(f"📊 Statistics:")
    print(f"  - Races updated: {races_updated:,}")
    print(f"  - Races with weather: {races_with_weather:,}")
    print(f"  - Weather success rate: {(races_with_weather/races_updated*100):.1f}%")
    print(f"  - Unique weather locations: {len(weather_cache):,}")
    print(f"  - API requests made: {len(unique_weather_requests):,}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Weather requests per batch (default: 20)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting weather update with {args.max_workers} workers, batch size {args.batch_size}")
    update_race_weather(max_workers=args.max_workers, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
