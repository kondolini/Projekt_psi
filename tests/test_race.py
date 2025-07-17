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

def load_tracks() -> dict[str, Track]:
    tracks = {}
    for fname in os.listdir(tracks_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(tracks_dir, fname), "rb") as f:
                track = pickle.load(f)
                tracks[track.name] = track
    return tracks

# ---------- Race Construction ----------

def build_races(dog_lookup: dict[str, Dog], participations: list) -> list[Race]:
    race_groups = {}
    for p in participations:
        race_groups.setdefault(p.race_id, []).append(p)

    races = []
    for race_id, parts in tqdm(race_groups.items(), desc="Constructing races"):
        try:
            race = Race.from_participations(parts)
            races.append(race)
        except Exception as e:
            print(f"Skipping race {race_id}: {e}")
    return races

# ---------- Main Test Function ----------

def test_race():
    print("Loading dogs...")
    dog_lookup = load_dog_buckets()

    print("Loading race participations...")
    participations = load_participation_buckets()

    print("Loading tracks...")
    track_lookup = load_tracks()

    print("Building races...")
    races = build_races(dog_lookup, participations)

    print(f"Saving {len(races)} races to {output_dir}...")
    for race in tqdm(races, desc="Saving races"):
        path = os.path.join(output_dir, f"{race.race_id}.pkl")
        race.save(path)

    print("\nSample race info:\n")
    for race in races[:5]:
        race.print_info()
        print()

if __name__ == "__main__":
    test_race()
