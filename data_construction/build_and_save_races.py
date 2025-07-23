import os
import sys
import pickle
import csv
from glob import glob
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load environment variables
load_dotenv()

# Adjust parent path to import project modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.race_participation import parse_race_participation
from models.race import Race

# Only import weather checker if needed
def get_weather_safe(date_str, time_str, track_name):
    """Safe weather import and call"""
    try:
        from scraping.weather_checker import get_weather
        return get_weather(date_str, time_str, track_name)
    except Exception as e:
        print(f"Weather import/call failed: {e}")
        return None

# Configuration
NUM_BUCKETS = int(os.getenv('NUM_BUCKETS', 100))

def build_and_save_races(
    dogs_enhanced_dir=None,
    participation_dir=None,
    race_output_dir=None,
    unified_dir=None,
    dog_name_csv=None,
    max_workers=2,  # Reduced to 2 for weather API
    batch_size=50,  # Smaller batches when weather enabled
    disable_weather=False  # Enable weather by default but with smart caching
):
    # Get project root for absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Use environment variables as defaults with absolute paths
    dogs_enhanced_dir = dogs_enhanced_dir or os.path.join(project_root, os.getenv('DOGS_ENHANCED_DIR', 'data/dogs_enhanced'))
    participation_dir = participation_dir or os.path.join(project_root, os.getenv('RACE_PARTICIPATIONS_DIR', 'data/race_participations'))
    race_output_dir = race_output_dir or os.path.join(project_root, os.getenv('RACES_DIR', 'data/races'))
    unified_dir = unified_dir or os.path.join(project_root, os.getenv('UNIFIED_DIR', 'data/unified'))
    dog_name_csv = dog_name_csv or os.path.join(project_root, os.getenv('DOG_NAME_CSV', 'data/dog_name_dict.csv'))

    # Prepare directories
    os.makedirs(race_output_dir, exist_ok=True)
    os.makedirs(unified_dir, exist_ok=True)

    # Load dog name mapping (name -> id)
    dog_name_map = {}
    if os.path.exists(dog_name_csv):
        with open(dog_name_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dog_name_map[row['dogName']] = row['dogId']

    # Load all enhanced dogs
    dog_lookup = {}
    for path in glob(os.path.join(dogs_enhanced_dir, 'dogs_bucket_*.pkl')):
        with open(path, 'rb') as f:
            dog_lookup.update(pickle.load(f))

    print(f"Loaded {len(dog_lookup)} dogs from enhanced directory")

    # Build comprehensive parent-offspring mapping with artificial parent creation
    missing = defaultdict(list)
    parent_offspring = defaultdict(list)
    artificial_id_counter = 100000
    
    print("Fixing pedigree relationships...")
    
    # First pass: identify existing relationships and collect missing parents
    for dog in dog_lookup.values():
        for rel in ('sire', 'dam'):
            val = getattr(dog, rel)
            if isinstance(val, str) and val:
                if val in dog_name_map and dog_name_map[val] in dog_lookup:
                    pid = dog_name_map[val]
                    parent = dog_lookup[pid]
                    setattr(dog, rel, parent)
                    parent_offspring[pid].append(dog.id)
                else:
                    missing[val].append((dog.id, rel))
            elif isinstance(val, Dog):
                parent_offspring[val.id].append(dog.id)

    print(f"Found {len(missing)} missing parents to create artificially")

    # Create artificial parents with proper IDs and update dog_name_map
    artificial_parents_created = 0
    for pname, child_relationships in missing.items():
        if artificial_id_counter >= 400000:  # Proper upper limit for artificial parents
            print("Warning: Reached maximum artificial parent ID limit")
            break
            
        art_id = str(artificial_id_counter)
        artificial_id_counter += 1
        artificial_parents_created += 1
        
        # Create artificial parent
        art = Dog(dog_id=art_id)
        art.set_name(pname)
        
        # Calculate average weight from children
        child_weights = []
        for child_id, _ in child_relationships:
            child = dog_lookup[child_id]
            if child.weight:
                child_weights.append(child.weight)
        
        if child_weights:
            art.set_weight(sum(child_weights) / len(child_weights))
        
        # Set properties for artificial parent
        art.sire = None
        art.dam = None
        art.trainer = None
        art.birth_date = None
        art.race_participations = []
        
        # Add to lookup and name mapping
        dog_lookup[art_id] = art
        dog_name_map[pname] = art_id
        
        # Update children to reference artificial parent
        for child_id, rel_name in child_relationships:
            child = dog_lookup[child_id]
            setattr(child, rel_name, art)
            parent_offspring[art_id].append(child_id)

    print(f"Created {artificial_parents_created} artificial parents with IDs {100000}-{artificial_id_counter-1}")

    # Save updated dog name mapping
    with open(dog_name_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dogName', 'dogId'])
        writer.writeheader()
        for name, dog_id in dog_name_map.items():
            writer.writerow({'dogName': name, 'dogId': dog_id})
    
    print(f"Updated dog name dictionary with {len(dog_name_map)} entries")

    # Save enhanced dogs (including artificial parents) back to buckets
    dog_buckets = defaultdict(dict)
    
    for dog_id, dog in dog_lookup.items():
        bucket_idx = int(dog_id) % NUM_BUCKETS
        dog_buckets[bucket_idx][dog_id] = dog
    
    # Save updated dog buckets
    for bucket_idx, dogs_dict in dog_buckets.items():
        bucket_path = os.path.join(dogs_enhanced_dir, f"dogs_bucket_{bucket_idx}.pkl")
        with open(bucket_path, 'wb') as f:
            pickle.dump(dogs_dict, f)
    
    print("Saved enhanced dogs with fixed pedigree relationships")

    # Save parent-offspring mapping to CSV
    save_parent_offspring_mapping_to_csv(parent_offspring, dog_lookup, unified_dir)

    # Index participations per race (use both race_id and meeting_id for uniqueness)
    race_parts = defaultdict(list)
    for path in glob(os.path.join(participation_dir, 'participations_bucket_*.pkl')):
        with open(path, 'rb') as f:
            for p in pickle.load(f):
                # Fix: Convert meeting_id to string if it's a float
                meeting_id = p.meeting_id
                if isinstance(meeting_id, float):
                    meeting_id = str(int(meeting_id))
                elif meeting_id is not None:
                    meeting_id = str(meeting_id)
                
                race_key = (p.race_id, meeting_id) if meeting_id else p.race_id
                race_parts[race_key].append(p)

    print(f"Indexed {len(race_parts)} unique races")
    print(f"Using {max_workers} workers with batch size {batch_size}")
    print(f"Weather fetching: {'DISABLED' if disable_weather else 'ENABLED with smart caching'}")
    
    # Pre-populate weather cache with unique (date, location) combinations
    if not disable_weather:
        weather_cache = build_weather_cache(race_parts)
    else:
        weather_cache = {}
    
    # Create batches for parallel processing
    race_items = list(race_parts.items())
    race_batches = [race_items[i:i + batch_size] for i in range(0, len(race_items), batch_size)]
    
    # Shared weather cache to avoid duplicate API calls
    weather_cache_lock = threading.Lock()
    
    # Process batches in parallel
    race_buckets = defaultdict(dict)
    unified_index = {}
    races_built = 0
    races_with_weather = 0
    
    print(f"Processing {len(race_batches)} batches...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches with weather flag
        future_to_batch = {
            executor.submit(build_race_batch, batch, weather_cache, weather_cache_lock, disable_weather): i
            for i, batch in enumerate(race_batches)
        }
        
        # Collect results with progress bar
        with tqdm(total=len(race_batches), desc="Processing batches", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    built_races, batch_weather_count = future.result()
                    races_with_weather += batch_weather_count
                    
                    # Add races to buckets
                    for race_key, (race, storage_key) in built_races.items():
                        bucket_idx = int(race.race_id) % NUM_BUCKETS
                        race_buckets[bucket_idx][storage_key] = race
                        
                        unified_index[race_key] = {
                            'bucket': bucket_idx,
                            'key': storage_key,
                            'path': os.path.join(race_output_dir, f'races_bucket_{bucket_idx}.pkl')
                        }
                        races_built += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'races': races_built,
                        'weather': races_with_weather,
                        'cache_size': len(weather_cache)
                    })
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    pbar.update(1)

    # Save race buckets in parallel
    print(f"Saving {len(race_buckets)} race buckets...")
    
    def save_bucket(bucket_data):
        bucket_idx, races_dict = bucket_data
        bucket_path = os.path.join(race_output_dir, f'races_bucket_{bucket_idx}.pkl')
        with open(bucket_path, 'wb') as f:
            pickle.dump(races_dict, f)
        return bucket_idx, len(races_dict)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        save_futures = [
            executor.submit(save_bucket, (bucket_idx, races_dict))
            for bucket_idx, races_dict in race_buckets.items()
        ]
        
        for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving buckets"):
            bucket_idx, count = future.result()

    # Save unified indexes including parent-offspring
    with open(os.path.join(unified_dir, 'race_index.pkl'), 'wb') as f:
        pickle.dump(unified_index, f)
    with open(os.path.join(unified_dir, 'parent_offspring.pkl'), 'wb') as f:
        pickle.dump(dict(parent_offspring), f)

    print(f"\n🎉 BUILD COMPLETED!")
    print("=" * 50)
    print(f"📊 Statistics:")
    print(f"  - Dogs processed: {len(dog_lookup):,}")
    print(f"  - Artificial parents created: {artificial_parents_created:,}")
    print(f"  - Parent-offspring relationships: {len(parent_offspring):,}")
    print(f"  - Races built: {races_built:,}")
    print(f"  - Race buckets created: {len(race_buckets):,}")
    print(f"  - Races with weather: {races_with_weather:,}")
    print(f"  - Weather success rate: {(races_with_weather/races_built*100):.1f}%")
    print(f"  - Weather cache entries: {len(weather_cache):,}")
    
    return unified_index, parent_offspring

def build_weather_cache(race_parts):
    """Pre-build weather cache to minimize API calls"""
    print("Pre-analyzing races for weather optimization...")
    
    # Find unique (date, track) combinations
    unique_weather_keys = set()
    for race_key, parts in race_parts.items():
        if parts:
            p = parts[0]  # Get first participation to extract date/track
            date_str = p.race_datetime.date().strftime('%Y-%m-%d')
            track_name = p.track_name
            unique_weather_keys.add((date_str, track_name))
    
    print(f"Found {len(unique_weather_keys)} unique weather combinations")
    print(f"Estimated API calls needed: {len(unique_weather_keys)}")
    print(f"Estimated time (at 3s per call): {len(unique_weather_keys) * 3 / 60:.1f} minutes")
    
    # Pre-populate cache with empty entries - will be filled during processing
    weather_cache = {}
    for weather_key in unique_weather_keys:
        weather_cache[weather_key] = None
    
    return weather_cache

def build_race_batch(race_batch, weather_cache, weather_cache_lock, disable_weather=False):
    """Build a batch of races with shared weather cache and smart weather fetching"""
    built_races = {}
    races_with_weather = 0
    
    for race_key, parts in race_batch:
        if not parts:
            continue
            
        try:
            race = Race.from_participations(parts)
            
            # Add weather data with smart caching (only if enabled)
            if not disable_weather:
                # Create weather key based on date and approximate location
                weather_key = (race.race_date.strftime('%Y-%m-%d'), race.track_name)
                
                with weather_cache_lock:
                    weather_data = weather_cache.get(weather_key)
                
                if weather_data is None:
                    try:
                        # Only fetch if we haven't tried this combination recently
                        date_str = race.race_date.strftime('%Y-%m-%d')
                        time_str = "12:00"  # Use noon for all races to reduce API calls
                        
                        # Add random delay between 2-5 seconds to be respectful
                        import time
                        import random
                        time.sleep(random.uniform(2.0, 5.0))
                        
                        w = get_weather_safe(date_str, time_str, race.track_name)
                        
                        with weather_cache_lock:
                            weather_cache[weather_key] = w if w else False
                        
                        weather_data = w
                    except Exception as e:
                        # Mark as failed and don't retry
                        with weather_cache_lock:
                            weather_cache[weather_key] = False
                        weather_data = False
                
                if weather_data and weather_data != False:
                    race.rainfall_7d = weather_data['rainfall_7d']
                    race.humidity = weather_data['humidity']
                    race.temperature = weather_data['temperature']
                    races_with_weather += 1
                else:
                    # Set default weather values
                    race.rainfall_7d = [0.0] * 7
                    race.humidity = 50.0
                    race.temperature = 15.0
            else:
                # Skip weather fetching entirely
                race.rainfall_7d = [0.0] * 7
                race.humidity = 50.0
                race.temperature = 15.0
            
            # Create storage key
            if isinstance(race_key, tuple):
                race_id, meeting_id = race_key
                storage_key = f"{race_id}_{meeting_id}" if meeting_id else race_id
            else:
                storage_key = str(race_key)
            
            built_races[race_key] = (race, storage_key)
                
        except Exception as e:
            # Only log critical errors
            continue
    
    return built_races, races_with_weather

def save_parent_offspring_mapping_to_csv(parent_offspring_dict, dog_lookup, unified_dir):
    """Save the parent-offspring mapping to CSV file"""
    csv_path = os.path.join(unified_dir, 'parent_offspring_mapping.csv')
    
    # Prepare data for CSV
    csv_data = []
    for parent_id, offspring_ids in parent_offspring_dict.items():
        parent_dog = dog_lookup.get(parent_id)
        parent_name = parent_dog.name if parent_dog and parent_dog.name else "Unknown"
        is_artificial = int(parent_id) >= 100000
        
        for offspring_id in offspring_ids:
            offspring_dog = dog_lookup.get(offspring_id)
            offspring_name = offspring_dog.name if offspring_dog and offspring_dog.name else "Unknown"
            
            csv_data.append({
                'parent_id': parent_id,
                'parent_name': parent_name,
                'is_artificial_parent': is_artificial,
                'offspring_id': offspring_id,
                'offspring_name': offspring_name,
                'offspring_count': len(offspring_ids)
            })
    
    # Save to CSV
    if csv_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['parent_id', 'parent_name', 'is_artificial_parent', 
                                                 'offspring_id', 'offspring_name', 'offspring_count'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"Saved parent-offspring mapping to {csv_path}")
        
        # Statistics
        artificial_count = sum(1 for row in csv_data if row['is_artificial_parent'])
        real_count = len(csv_data) - artificial_count
        
        print(f"  - Real parent relationships: {real_count:,}")
        print(f"  - Artificial parent relationships: {artificial_count:,}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-weather', action='store_true', 
                       help='Disable weather data fetching')
    parser.add_argument('--max-workers', type=int, default=16,  # Increased default
                       help='Number of worker threads (default: 16 when weather disabled)')
    parser.add_argument('--batch-size', type=int, default=500,  # Increased default
                       help='Batch size for processing (default: 500)')
    
    args = parser.parse_args()
    
    # Optimize settings for weather disabled mode
    if args.disable_weather:
        # Weather disabled - use aggressive settings for speed
        max_workers = args.max_workers
        batch_size = args.batch_size
        print("🚀 SPEED MODE: Weather disabled, using maximum performance settings")
    else:
        # Weather enabled - use conservative settings
        max_workers = min(args.max_workers, 2)
        batch_size = min(args.batch_size, 50)
        print("🌦️ WEATHER MODE: Conservative settings for API respect")
    
    print(f"Workers: {max_workers}, Batch size: {batch_size}")
    
    build_and_save_races(
        disable_weather=args.disable_weather,
        max_workers=max_workers,
        batch_size=batch_size
    )

if __name__ == '__main__':
    main()
