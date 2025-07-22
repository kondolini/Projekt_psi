import os
import sys
import pickle
from typing import Optional
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.track import Track
from models.race_participation import parse_race_participation

# Config
NUM_BUCKETS = int(os.getenv('NUM_BUCKETS', 100))

# Paths from environment
data_dir = os.getenv('SCRAPED_DIR', 'data/scraped')
dogs_output_dir = os.getenv('DOGS_DIR', 'data/dogs')
tracks_output_dir = os.getenv('TRACKS_DIR', 'data/tracks')
participation_output_dir = os.getenv('RACE_PARTICIPATIONS_DIR', 'data/race_participations')
unified_dir = os.getenv('UNIFIED_DIR', 'data/unified')
scraped_data_csv = os.getenv('SCRAPED_DATA_CSV', 'data/scraped/scraped_data.csv')

# Create directories
os.makedirs(dogs_output_dir, exist_ok=True)
os.makedirs(tracks_output_dir, exist_ok=True)
os.makedirs(participation_output_dir, exist_ok=True)
os.makedirs(unified_dir, exist_ok=True)

# Hashing strategy
def get_bucket_index(dog_id: str) -> int:
    return int(dog_id) % NUM_BUCKETS

def load_dog_by_id(dog_id: str) -> Optional[Dog]:
    bucket = get_bucket_index(dog_id)
    path = os.path.join(dogs_output_dir, f"dogs_bucket_{bucket}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        bucket_data = pickle.load(f)
    return bucket_data.get(dog_id)

def save_dog(dog: Dog):
    bucket = get_bucket_index(dog.id)
    path = os.path.join(dogs_output_dir, f"dogs_bucket_{bucket}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            bucket_data = pickle.load(f)
    else:
        bucket_data = {}
    bucket_data[dog.id] = dog
    with open(path, "wb") as f:
        pickle.dump(bucket_data, f)

# Main function
def build_and_save_dogs():
    df = pd.read_csv(scraped_data_csv)
    grouped = df.groupby("dogId")

    race_to_dog_index = defaultdict(list)
    race_to_participation_files = defaultdict(list)

    dog_buckets = defaultdict(dict)
    participation_buckets = defaultdict(list)
    track_cache = {}

    for dog_id, group in tqdm(grouped, desc="Processing dogs"):
        dog = Dog(dog_id=dog_id)
        participations = []

        for _, row in group.iterrows():
            participation = parse_race_participation(row)
            if participation is None:
                continue
            participations.append(participation)

            bucket_idx = get_bucket_index(dog_id)
            participation_buckets[bucket_idx].append(participation)

            # Indexing
            race_to_dog_index[participation.race_id].append(dog_id)

            # Track caching
            track_name = participation.track_name
            if track_name and track_name not in track_cache:
                track = Track.from_race_participations([participation])
                track_cache[track_name] = track

        dog.add_participations(participations)
        dog_buckets[get_bucket_index(dog_id)][dog_id] = dog

    # Save dog buckets
    for idx, dog_dict in dog_buckets.items():
        with open(os.path.join(dogs_output_dir, f"dogs_bucket_{idx}.pkl"), "wb") as f:
            pickle.dump(dog_dict, f)

    # Save participation buckets
    for idx, part_list in participation_buckets.items():
        with open(os.path.join(participation_output_dir, f"participations_bucket_{idx}.pkl"), "wb") as f:
            pickle.dump(part_list, f)

    # Save tracks
    for name, track in track_cache.items():
        path = os.path.join(tracks_output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(track, f)

    # Save unified indexes
    with open(unified_race_index_path, "wb") as f:
        pickle.dump(dict(race_to_dog_index), f)

    with open(unified_participation_index_path, "wb") as f:
        pickle.dump(dict(race_to_participation_files), f)

if __name__ == "__main__":
    build_and_save_dogs()
