import os
import sys
import pickle
import pandas as pd
import shutil
import time
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
NUM_BUCKETS = 10
ROWS_PER_TEST = 100_000  # Now using 10,000 rows

# Paths
data_dir = "data/scraped"
test_data_dir = "test_data"
dogs_output_dir = os.path.join(test_data_dir, "dogs")
tracks_output_dir = os.path.join(test_data_dir, "tracks")
participation_output_dir = os.path.join(test_data_dir, "race_participations")

def ensure_dirs():
    os.makedirs(dogs_output_dir, exist_ok=True)
    os.makedirs(tracks_output_dir, exist_ok=True)
    os.makedirs(participation_output_dir, exist_ok=True)

def clean_test_data():
    shutil.rmtree(test_data_dir, ignore_errors=True)

def get_bucket_index(dog_id: str) -> int:
    return int(dog_id) % NUM_BUCKETS

def run_test(df: pd.DataFrame):
    ensure_dirs()
    dog_buckets = defaultdict(dict)
    participation_buckets = defaultdict(list)
    track_cache = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        participation = parse_race_participation(row)
        if not participation:
            continue

        dog_id = participation.dog_id
        bucket_idx = get_bucket_index(dog_id)
        dog = dog_buckets[bucket_idx].get(dog_id) or Dog(dog_id=dog_id)
        dog.add_participation(participation)
        dog_buckets[bucket_idx][dog_id] = dog
        participation_buckets[bucket_idx].append(participation)

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

def test_parser():
    csv_path = os.path.join(data_dir, "scraped_data.csv")
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return

    df = pd.read_csv(csv_path)
    if len(df) < ROWS_PER_TEST:
        print(f"Not enough data rows. Only {len(df)} rows available.")
        return

    sample_df = df.sample(n=ROWS_PER_TEST, random_state=42)

    print(f"\nRunning test on {ROWS_PER_TEST} rows...\n")

    start = time.perf_counter()
    run_test(sample_df)
    end = time.perf_counter()

    total_time = end - start

    print("\n==============================")
    print(f"Total time for {ROWS_PER_TEST} rows: {total_time:.3f} seconds")
    print("==============================\n")

    print("Cleaning up test files...")
    clean_test_data()
    print("Cleanup completed.")

if __name__ == "__main__":
    test_parser()
