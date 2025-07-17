import os
import sys
import pickle
import random
from collections import defaultdict

# Extend module path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from models.track import Track
from models.race_participation import RaceParticipation

# Paths - Updated to use enhanced buckets
dogs_dir = "../data/dogs_enhanced"  # ← Changed to enhanced buckets
participations_dir = "../data/race_participations"
tracks_dir = "../data/tracks"
unified_dir = "../data/unified"
race_to_dog_index_path = os.path.join(unified_dir, "race_to_dog_index.pkl")

NUM_BUCKETS = 100

def get_bucket_index(dog_id: str) -> int:
    return int(dog_id) % NUM_BUCKETS

def load_dog(dog_id: str) -> Dog:
    bucket = get_bucket_index(dog_id)
    path = os.path.join(dogs_dir, f"dogs_bucket_{bucket}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[dog_id]

def load_all_participations() -> list[RaceParticipation]:
    participations = []
    for fname in os.listdir(participations_dir):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(participations_dir, fname), "rb") as f:
            participations.extend(pickle.load(f))
    return participations

def load_track_by_name(name: str) -> Track:
    path = os.path.join(tracks_dir, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def reconstruct_race(race_id: str, race_to_dog_index: dict[str, list[str]], all_participations: list[RaceParticipation]) -> Race:
    dog_ids = race_to_dog_index[race_id]
    dogs = [load_dog(did) for did in dog_ids]

    participations = []
    track_name = None
    
    for dog in dogs:
        try:
            part_list = dog.get_participation_by_race_id(race_id)  # Now returns a list
            if part_list:  # Check if list is not empty
                participations.extend(part_list)
                # Get track name from first participation
                if part_list and part_list[0].track_name:
                    track_name = part_list[0].track_name
        except Exception as e:
            print(f"Warning: Could not get participation for dog {dog.id} in race {race_id}: {e}")
            continue

    if not participations:
        raise ValueError(f"No participation data found for race_id={race_id}")

    # Create track object if we have track name
    track = None
    if track_name:
        try:
            track = load_track_by_name(track_name)
        except Exception as e:
            print(f"Warning: Could not load track {track_name}: {e}")
            # Create track from participations if loading fails
            try:
                track = Track.from_race_participations(participations)
            except Exception as e2:
                print(f"Warning: Could not create track from participations: {e2}")
    
    # Create race object using the enhanced Race.from_participations method
    race = Race.from_participations(race_id, participations, track)
    return race

def test_race():
    print("Loading race_to_dog_index...")
    with open(race_to_dog_index_path, "rb") as f:
        race_to_dog_index = pickle.load(f)

    print("Loading all participations...")
    all_participations = load_all_participations()

    # Only test races with dogs from buckets 0, 1, and 2
    available_buckets = [0, 1, 2]
    valid_race_ids = []
    
    print(f"Finding valid race IDs with dogs from buckets {available_buckets}...")
    
    for race_id, dog_ids in race_to_dog_index.items():
        try:
            # Check if ALL dogs in this race are from available buckets
            dog_buckets = [get_bucket_index(dog_id) for dog_id in dog_ids]
            
            # Only proceed if all dogs are from buckets 0, 1, or 2
            if all(bucket in available_buckets for bucket in dog_buckets):
                # Verify the bucket files exist and contain the dogs
                all_dogs_exist = True
                for dog_id in dog_ids:
                    bucket = get_bucket_index(dog_id)
                    path = os.path.join(dogs_dir, f"dogs_bucket_{bucket}.pkl")
                    
                    if not os.path.exists(path):
                        all_dogs_exist = False
                        break
                    
                    with open(path, "rb") as f:
                        bucket_data = pickle.load(f)
                    if dog_id not in bucket_data:
                        all_dogs_exist = False
                        break
                    
                    # Check if dog has participations for this race
                    dog = bucket_data[dog_id]
                    dog_participations = dog.get_participation_by_race_id(race_id)
                    if not dog_participations:
                        all_dogs_exist = False
                        break
                
                if all_dogs_exist:
                    valid_race_ids.append(race_id)
                    print(f"✅ Race {race_id}: Dogs {dog_ids} (buckets {dog_buckets})")
                    
                    if len(valid_race_ids) >= 5:  # Test with 5 races
                        break
                        
        except Exception as e:
            continue
    
    if not valid_race_ids:
        print("❌ No valid race IDs found with dogs from available buckets")
        print("💡 Debug info:")
        
        # Show some sample races and their required buckets
        print("\nSample races and their required buckets:")
        for race_id, dog_ids in list(race_to_dog_index.items())[:10]:
            dog_buckets = [get_bucket_index(dog_id) for dog_id in dog_ids]
            status = "✅" if all(b in available_buckets for b in dog_buckets) else "❌"
            print(f"{status} Race {race_id}: Dogs {dog_ids[:3]}... → Buckets {dog_buckets}")
        
        # Check what bucket files actually exist
        existing_buckets = []
        if os.path.exists(dogs_dir):
            for i in range(100):
                bucket_path = os.path.join(dogs_dir, f"dogs_bucket_{i}.pkl")
                if os.path.exists(bucket_path):
                    existing_buckets.append(i)
        
        print(f"\nExisting bucket files: {existing_buckets[:10]}...")
        return

    print(f"Found {len(valid_race_ids)} valid race IDs for testing")
    print("\n--- Testing Enhanced Race Reconstruction ---\n")
    
    for race_id in valid_race_ids:
        try:
            print(f"Reconstructing Race ID: {race_id}")
            dog_ids = race_to_dog_index[race_id]
            dog_buckets = [get_bucket_index(dog_id) for dog_id in dog_ids]
            print(f"Dogs in race: {dog_ids} (buckets: {dog_buckets})")
            
            race = reconstruct_race(race_id, race_to_dog_index, all_participations)
            print(race)  # Calls __repr__
            print("\nDetailed Info:")
            race.print_info()  # Now shows all the detailed information
            
            # Also show race summary for structured data access
            print("\n" + "="*50)
            print("RACE SUMMARY DATA STRUCTURE:")
            summary = race.get_race_summary()
            
            print(f"Race Metadata: {summary['race_metadata']}")
            print(f"Race Statistics: {summary['race_statistics']}")
            print("Sample Participant Detail:")
            if summary['participants']:
                import json
                print(json.dumps(summary['participants'][0], indent=2, default=str))
            
            print("\n" + "=" * 80 + "\n")
            
        except Exception as e:
            print(f"Error processing race {race_id}: {e}")
            dog_buckets = [get_bucket_index(dog_id) for dog_id in race_to_dog_index.get(race_id, [])]
            print(f"Dog IDs for this race: {race_to_dog_index.get(race_id, [])} (buckets: {dog_buckets})")
            print()

if __name__ == "__main__":
    test_race()