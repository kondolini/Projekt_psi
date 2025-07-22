import os
import sys
import pickle
import csv
from glob import glob
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adjust parent path to import project modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.race_participation import parse_race_participation
from models.race import Race
from scraping.weather_checker import get_weather

# Configuration
NUM_BUCKETS = int(os.getenv('NUM_BUCKETS', 100))

def build_and_save_races(
    dogs_enhanced_dir=None,
    participation_dir=None,
    race_output_dir=None,
    unified_dir=None,
    dog_name_csv=None
):
    # Use environment variables as defaults
    dogs_enhanced_dir = dogs_enhanced_dir or os.getenv('DOGS_ENHANCED_DIR', 'data/dogs_enhanced')
    participation_dir = participation_dir or os.getenv('RACE_PARTICIPATIONS_DIR', 'data/race_participations')
    race_output_dir = race_output_dir or os.getenv('RACES_DIR', 'data/races')
    unified_dir = unified_dir or os.getenv('UNIFIED_DIR', 'data/unified')
    dog_name_csv = dog_name_csv or os.getenv('DOG_NAME_CSV', 'data/dog_name_dict.csv')

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
                race_key = (p.race_id, p.meeting_id) if p.meeting_id else p.race_id
                race_parts[race_key].append(p)

    print(f"Indexed {len(race_parts)} unique races")

    # Build races, attach weather, save
    unified_index = {}
    races_built = 0
    races_with_weather = 0
    
    for race_key, parts in race_parts.items():
        if not parts:
            continue
            
        try:
            race = Race.from_participations(parts)
            
            # Add weather data
            try:
                date_str = race.race_date.strftime('%Y-%m-%d')
                time_str = race.race_time.strftime('%H:%M')
                w = get_weather(date_str, time_str, race.track_name)
                if w:
                    race.rainfall_7d = w['rainfall_7d']
                    race.humidity = w['humidity']
                    race.temperature = w['temperature']
                    races_with_weather += 1
                else:
                    # Set default weather values
                    race.rainfall_7d = [0.0] * 7
                    race.humidity = 50.0
                    race.temperature = 15.0
            except Exception as e:
                print(f"Weather fetch failed for race {race.race_id}: {e}")
                race.rainfall_7d = [0.0] * 7
                race.humidity = 50.0
                race.temperature = 15.0
            
            # Save race
            if isinstance(race_key, tuple):
                race_id, meeting_id = race_key
                out = os.path.join(race_output_dir, f'race_{race_id}_{meeting_id}.pkl')
            else:
                out = os.path.join(race_output_dir, f'race_{race_key}.pkl')
            
            with open(out, 'wb') as f:
                pickle.dump(race, f)
            
            unified_index[race_key] = out
            races_built += 1
            
            # Progress update
            if races_built % 1000 == 0:
                print(f"Built {races_built} races, {races_with_weather} with weather data...")
                
        except Exception as e:
            print(f"Error building race {race_key}: {e}")
            continue

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
    print(f"  - Races with weather: {races_with_weather:,}")
    print(f"  - Weather success rate: {(races_with_weather/races_built*100):.1f}%")
    
    return unified_index, parent_offspring

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
    build_and_save_races()

if __name__ == '__main__':
    main()
