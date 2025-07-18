import os
import sys
import pickle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from models.track import Track

DOGS_DIR = "data/dogs"
TRACKS_DIR = "data/tracks"
PARTICIPATIONS_DIR = "data/race_participations"
RACES_OUT = "data/races"
os.makedirs(RACES_OUT, exist_ok=True)

def load_all_dogs() -> dict:
    dogs = {}
    for fname in os.listdir(DOGS_DIR):
        if fname.endswith(".pkl"):
            with open(os.path.join(DOGS_DIR, fname), "rb") as f:
                dogs.update(pickle.load(f))
    return dogs

def load_all_tracks() -> dict:
    tracks = {}
    for fname in os.listdir(TRACKS_DIR):
        if fname.endswith(".pkl"):
            with open(os.path.join(TRACKS_DIR, fname), "rb") as f:
                track = pickle.load(f)
                tracks[track.name] = track
    return tracks

def test_race_construction():
    dogs = load_all_dogs()
    tracks = load_all_tracks()

    sample_race_ids = set()
    for dog in dogs.values():
        for p in dog.race_participations:
            sample_race_ids.add(p.race_id)
        if len(sample_race_ids) >= 3:
            break

    for race_id in list(sample_race_ids)[:3]:
        race = Race.from_dogs(list(dogs.values()), race_id)
        if race:
            print("\n--- RACE INFO ---")
            race.print_info()

            print("\n--- DOG OBJECTS ---")
            for dog in race.get_dogs(dogs):
                dog.print_info()

            print("\n--- TRACK INFO ---")
            track = race.get_track(tracks)
            if track:
                track.print_info()

            race_path = os.path.join(RACES_OUT, f"{race_id}.pkl")
            race.save(race_path)
            reloaded = Race.load(race_path)

            assert race.race_id == reloaded.race_id
            assert race.dog_ids == reloaded.dog_ids
            assert race.track_name == reloaded.track_name
            print(f"\n✔ Reloaded race {race.race_id} verified successfully.\n")

if __name__ == "__main__":
    test_race_construction()
