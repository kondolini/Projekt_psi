import os
import sys

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.join(script_dir, '..')

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import pickle
import pandas as pd
from tqdm import tqdm

from models.dog import Dog
from models.race_participation import RaceParticipation, parse_race_participation

data_dir = "data/scraped"
dogs_output_dir = "data/dogs"
participation_output_dir = "data/race_participations"
unified_output_path = "data/unified/race_to_dog_index.pkl"

os.makedirs(dogs_output_dir, exist_ok=True)
os.makedirs(participation_output_dir, exist_ok=True)
os.makedirs("data/unified", exist_ok=True)

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
    path = os.path.join(participation_output_dir, f"{part.race_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(part, f)

def build_participation(row):
    return parse_race_participation(row)

def build_and_save_dogs():
    df = pd.read_csv(os.path.join(data_dir, "scraped_data.csv"))
    grouped = df.groupby("dogId")
    race_to_dog_index = {}

    for dog_id, group in tqdm(grouped):
        dog = load_dog(dog_id) or Dog(dog_id=dog_id)
        participations = []

        for _, row in group.iterrows():
            participation = build_participation(row)
            if participation is not None:
                participations.append(participation)
                save_participation(participation)
                race_to_dog_index.setdefault(participation.race_id, []).append(dog_id)

        dog.add_participations(participations)
        save_dog(dog)

    with open(unified_output_path, "wb") as f:
        pickle.dump(race_to_dog_index, f)

if __name__ == "__main__":
    build_and_save_dogs()
