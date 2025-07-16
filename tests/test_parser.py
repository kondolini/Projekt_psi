import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm

# Extend module path to include project root
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.track import Track
from models.race_participation import parse_race_participation

# Directory paths
data_dir = "data/scraped"
test_data_dir = "test_data"
dogs_output_dir = os.path.join(test_data_dir, "dogs")
tracks_output_dir = os.path.join(test_data_dir, "tracks")
participation_output_dir = os.path.join(test_data_dir, "race_participations")

# Ensure output directories exist
os.makedirs(dogs_output_dir, exist_ok=True)
os.makedirs(tracks_output_dir, exist_ok=True)
os.makedirs(participation_output_dir, exist_ok=True)

def save_dog(dog):
    path = os.path.join(dogs_output_dir, f"{dog.id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(dog, f)

def save_track(track):
    path = os.path.join(tracks_output_dir, f"{track.name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(track, f)

def save_participation(part):
    path = os.path.join(participation_output_dir, f"{part.race_id}_{part.dog_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(part, f)

def load_dog(dog_id):
    path = os.path.join(dogs_output_dir, f"{dog_id}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def test_parser():
    df = pd.read_csv(os.path.join(data_dir, "scraped_data.csv"))
    df = df.head(100)  # Limit to 100 rows for testing

    track_cache = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        participation = parse_race_participation(row)
        if not participation:
            continue

        # Save race participation
        save_participation(participation)

        # Save or update dog
        dog = load_dog(participation.dog_id) or Dog(dog_id=participation.dog_id)
        dog.add_participations([participation])
        save_dog(dog)

        # Save track if not already cached
        track_name = participation.track_name
        if track_name and track_name not in track_cache:
            track = Track.from_race_participations([participation])
            save_track(track)
            track_cache[track_name] = track

    print("\n--- Example Saved Objects ---")
    sample_dog_files = os.listdir(dogs_output_dir)[:3]
    sample_track_files = os.listdir(tracks_output_dir)[:3]
    sample_part_files = os.listdir(participation_output_dir)[:3]

    for f in sample_dog_files:
        with open(os.path.join(dogs_output_dir, f), "rb") as file:
            print("Dog:", pickle.load(file))

    for f in sample_track_files:
        with open(os.path.join(tracks_output_dir, f), "rb") as file:
            print("Track:", pickle.load(file))

    for f in sample_part_files:
        with open(os.path.join(participation_output_dir, f), "rb") as file:
            print("Participation:", pickle.load(file))

if __name__ == "__main__":
    test_parser()
