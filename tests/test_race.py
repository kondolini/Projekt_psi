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

# Paths
dogs_dir = "data/dogs"
participations_dir = "data/race_participations"
tracks_dir = "data/tracks"
unified_dir = "data/unified"
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
    
    # Create race object - check if Race.from_participations accepts track parameter
    try:
        # Try with track parameter first
        race = Race.from_participations(race_id, participations, track)
    except TypeError:
        # If that fails, try without track parameter
        try:
            race = Race.from_participations(race_id, participations)
            # Set track separately if the race object has a track attribute
            if track and hasattr(race, 'track'):
                race.track = track
        except Exception as e:
            print(f"Error creating race: {e}")
            # Last resort: try a basic constructor
            race = Race(race_id)
            race.participations = participations
            if hasattr(race, 'track'):
                race.track = track
    
    return race

def test_race():
    print("Loading race_to_dog_index...")
    with open(race_to_dog_index_path, "rb") as f:
        race_to_dog_index = pickle.load(f)

    print("Loading all participations...")
    all_participations = load_all_participations()

    # Filter race IDs to only include ones that have dogs
    valid_race_ids = []
    print("Finding valid race IDs with existing dogs...")
    
    for race_id, dog_ids in list(race_to_dog_index.items())[:50]:  # Check more races
        try:
            # Try to load the first dog to see if bucket exists
            first_dog_id = dog_ids[0]
            bucket = get_bucket_index(first_dog_id)
            path = os.path.join(dogs_dir, f"dogs_bucket_{bucket}.pkl")
            
            if os.path.exists(path):
                with open(path, "rb") as f:
                    bucket_data = pickle.load(f)
                if first_dog_id in bucket_data:
                    # Also check if the dog actually has participations for this race
                    dog = bucket_data[first_dog_id]
                    dog_participations = dog.get_participation_by_race_id(race_id)
                    if dog_participations:  # Only add if dog has participations for this race
                        valid_race_ids.append(race_id)
                        if len(valid_race_ids) >= 3:  # Test with 3 races for now
                            break
        except Exception as e:
            continue
    
    if not valid_race_ids:
        print("❌ No valid race IDs found with existing dog data and participations")
        print("💡 This might mean:")
        print("   - The dog buckets haven't been created yet (run build_and_save_dogs.py)")
        print("   - The race participation data is missing")
        print("   - There's a mismatch between race IDs and dog participation data")
        
        # Debug: Show what we have
        print(f"\n🔍 Debug info:")
        print(f"   - race_to_dog_index has {len(race_to_dog_index)} races")
        print(f"   - Dogs directory exists: {os.path.exists(dogs_dir)}")
        if os.path.exists(dogs_dir):
            bucket_files = [f for f in os.listdir(dogs_dir) if f.startswith('dogs_bucket_')]
            print(f"   - Found {len(bucket_files)} dog bucket files")
        return

    print(f"Found {len(valid_race_ids)} valid race IDs for testing")
    print("\n--- Testing Enhanced Race Reconstruction ---\n")
    
    for race_id in valid_race_ids:
        try:
            print(f"Reconstructing Race ID: {race_id}")
            dog_ids = race_to_dog_index[race_id]
            print(f"Dogs in race: {dog_ids}")
            
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
            print(f"Dog IDs for this race: {race_to_dog_index.get(race_id, [])}")
            print()

if __name__ == "__main__":
    test_race()
