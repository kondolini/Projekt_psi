import os
import sys
import pickle
from typing import Optional
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Extend module path to include project root
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.race_participation import parse_race_participation

# Config
NUM_BUCKETS = 100

# Paths
data_dir = "data/scraped"
dogs_output_dir = "data/dogs"
participation_output_dir = "data/race_participations"
unified_dir = "data/unified"
unified_race_index_path = os.path.join(unified_dir, "race_to_dog_index.pkl")
unified_participation_index_path = os.path.join(unified_dir, "race_index.pkl")

os.makedirs(dogs_output_dir, exist_ok=True)
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

# Main function
def build_and_save_dogs():
    df = pd.read_csv(os.path.join(data_dir, "scraped_data.csv"))
    grouped = df.groupby("dogId")

    race_to_dog_index = defaultdict(list)
    race_to_participation_files = defaultdict(list)

    dog_buckets = defaultdict(dict)
    participation_buckets = defaultdict(list)

    for dog_id, group in tqdm(grouped):
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

    # Save unified indexes
    with open(unified_race_index_path, "wb") as f:
        pickle.dump(dict(race_to_dog_index), f)

    with open(unified_participation_index_path, "wb") as f:
        pickle.dump(dict(race_to_participation_files), f)


if __name__ == "__main__":
    build_and_save_dogs()
