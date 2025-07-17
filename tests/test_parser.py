import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Extend module path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.track import Track
from models.race_participation import parse_race_participation

# Parameters
NUM_BUCKETS = 10  # Only 10 buckets for testing 100 rows

# Paths
data_dir = "data/scraped"
test_data_dir = "test_data"
dogs_output_dir = os.path.join(test_data_dir, "dogs")
tracks_output_dir = os.path.join(test_data_dir, "tracks")
participation_output_dir = os.path.join(test_data_dir, "race_participations")

# Ensure dirs
os.makedirs(dogs_output_dir, exist_ok=True)
os.makedirs(tracks_output_dir, exist_ok=True)
os.makedirs(participation_output_dir, exist_ok=True)

# Helper
def get_bucket_index(dog_id: str) -> int:
    return int(dog_id) % NUM_BUCKETS

def test_parser():
    df = pd.read_csv(os.path.join(data_dir, "scraped_data.csv"))
    df = df.head(100)

    dog_buckets = defaultdict(dict)
    participation_buckets = defaultdict(list)
    track_cache = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        participation = parse_race_participation(row)
        if not participation:
            continue

        # DOG
        dog_id = participation.dog_id
        bucket_idx = get_bucket_index(dog_id)
        dog = dog_buckets[bucket_idx].get(dog_id) or Dog(dog_id=dog_id)
        dog.add_participation(participation)
        dog_buckets[bucket_idx][dog_id] = dog

        # PARTICIPATION
        participation_buckets[bucket_idx].append(participation)

        # TRACK
        track_name = participation.track_name
        if track_name and track_name not in track_cache:
            track = Track.from_race_participations([participation])
            track_cache[track_name] = track

    # Save dogs
    for idx, dog_dict in dog_buckets.items():
        with open(os.path.join(dogs_output_dir, f"dogs_bucket_{idx}.pkl"), "wb") as f:
            pickle.dump(dog_dict, f)

    # Save participations
    for idx, part_list in participation_buckets.items():
        with open(os.path.join(participation_output_dir, f"participations_bucket_{idx}.pkl"), "wb") as f:
            pickle.dump(part_list, f)

    # Save tracks
    for name, track in track_cache.items():
        path = os.path.join(tracks_output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(track, f)

    # Display a few
    print("\n--- Example Saved Objects ---")

    dog_files = os.listdir(dogs_output_dir)[:1]
    for f in dog_files:
        with open(os.path.join(dogs_output_dir, f), "rb") as file:
            dogs = pickle.load(file)
            for d in list(dogs.values())[:2]:
                print("Dog:", d)

    track_files = os.listdir(tracks_output_dir)[:2]
    for f in track_files:
        with open(os.path.join(tracks_output_dir, f), "rb") as file:
            print("Track:", pickle.load(file))

    part_files = os.listdir(participation_output_dir)[:1]
    for f in part_files:
        with open(os.path.join(participation_output_dir, f), "rb") as file:
            parts = pickle.load(file)
            for p in parts[:2]:
                print("Participation:", p)


if __name__ == "__main__":
    test_parser()
