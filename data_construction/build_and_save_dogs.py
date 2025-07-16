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
from models.race_participation import parse_race_participation

# Directory paths
data_dir = "data/scraped"
dogs_output_dir = "data/dogs"
participation_output_dir = "data/race_participations"
unified_dir = "data/unified"
unified_race_index_path = os.path.join(unified_dir, "race_to_dog_index.pkl")
unified_participation_index_path = os.path.join(unified_dir, "race_index.pkl")

# Ensure output directories exist
os.makedirs(dogs_output_dir, exist_ok=True)
os.makedirs(participation_output_dir, exist_ok=True)
os.makedirs(unified_dir, exist_ok=True)

def load_dog(dog_id):
    path = os.path.join(dogs_output_dir, f"{dog_id}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def save_dog(dog):
    path = os.path.join(dogs_output_dir, f"{dog.id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(dog, f)

def save_participation(part):
    path = os.path.join(participation_output_dir, f"{part.race_id}_{part.dog_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(part, f)
    return path

def build_participation(row):
    return parse_race_participation(row)

def build_and_save_dogs():
    df = pd.read_csv(os.path.join(data_dir, "scraped_data.csv"))
    grouped = df.groupby("dogId")
    race_to_dog_index = {}
    race_to_participation_files = {}

    for dog_id, group in tqdm(grouped):
        dog = load_dog(dog_id) or Dog(dog_id=dog_id)
        participations = []

        for _, row in group.iterrows():
            participation = build_participation(row)
            if participation is not None:
                participations.append(participation)
                part_path = save_participation(participation)

                # Index dog for this race
                race_to_dog_index.setdefault(participation.race_id, []).append(dog_id)

                # Index file for this race
                race_to_participation_files.setdefault(participation.race_id, []).append(part_path)

        dog.add_participations(participations)
        save_dog(dog)

    with open(unified_race_index_path, "wb") as f:
        pickle.dump(race_to_dog_index, f)

    with open(unified_participation_index_path, "wb") as f:
        pickle.dump(race_to_participation_files, f)

if __name__ == "__main__":
    build_and_save_dogs()
