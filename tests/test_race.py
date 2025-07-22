# test_race.py

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

    sample_races = set()
    for dog in dogs.values():
        for p in dog.race_participations:
            if p.race_id and p.meeting_id:
                sample_races.add((p.race_id, p.meeting_id))
        if len(sample_races) >= 30:
            break

    for race_id, meeting_id in list(sample_races)[:30]:
        trap_to_dog = {}
        for dog in dogs.values():
            for p in dog.race_participations:
                if p.race_id == race_id and p.meeting_id == meeting_id and p.trap_number is not None:
                    trap_to_dog[p.trap_number] = dog
        if not trap_to_dog:
            continue

        race = Race.from_dogs(trap_to_dog, race_id, meeting_id)
        if race:
            print("\n========================")
            race.print_info()
            print("--- Loaded Dogs ---")
            for trap, dog in race.get_dogs(dogs).items():
                print(f"Trap {trap}: {dog.name or dog.id}")
            print("--- Track Info ---")
            track = race.get_track(tracks)
            if track:
                print(f"Track: {track.name}")
            # Save and reload
            race_path = os.path.join(RACES_OUT, f"{race_id}_{meeting_id}.pkl")
            race.save(race_path)
            reloaded = Race.load(race_path)
            assert reloaded.race_id == race.race_id
            assert reloaded.dog_ids == race.dog_ids
            assert reloaded.meeting_id == race.meeting_id
            print(f"✔ Race {race_id} saved and reloaded correctly.")

if __name__ == "__main__":
    test_race_construction()
