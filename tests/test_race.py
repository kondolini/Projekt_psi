import os
import sys
import pickle
from tqdm import tqdm

# ---------- Extend module path ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from models.track import Track

# ---------- Paths ----------
dogs_dir = "data/dogs"
participations_dir = "data/race_participations"
tracks_dir = "data/tracks"
output_dir = "test_data/race"
os.makedirs(output_dir, exist_ok=True)

# ---------- Load Functions ----------

def load_dog_buckets() -> dict[str, Dog]:
    dogs = {}
    for fname in os.listdir(dogs_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(dogs_dir, fname), "rb") as f:
                bucket = pickle.load(f)
                dogs.update(bucket)
    return dogs

def load_participation_buckets() -> list:
    participations = []
    for fname in os.listdir(participations_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(participations_dir, fname), "rb") as f:
                bucket = pickle.load(f)
                participations.extend(bucket)
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
        part = dog.get_participation_by_race_id(race_id)
        if part:
            participations.extend(part)
            if part.track_name:
                track_name = part.track_name

    if not participations:
        raise ValueError(f"No participation data for race_id={race_id}")

    track = load_track_by_name(track_name) if track_name else None
    race = Race.from_participations(race_id, participations, track=track)
    return race

def test_race():
    print("Loading race_to_dog_index...")
    with open(race_to_dog_index_path, "rb") as f:
        race_to_dog_index = pickle.load(f)

    print("Loading all participations...")
    all_participations = load_all_participations()

    sample_race_ids = random.sample(list(race_to_dog_index.keys()), 3)

    print("\n--- Testing Race Reconstruction ---\n")
    for race_id in sample_race_ids:
        try:
            print(f"Reconstructing Race ID: {race_id}")
            race = reconstruct_race(race_id, race_to_dog_index, all_participations)
            print(race)  # Calls __repr__
            print("\nDetailed Info:")
            race.print_info()
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"Error processing race {race_id}: {e}")

if __name__ == "__main__":
    test_race()
