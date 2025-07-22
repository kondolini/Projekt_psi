import os
import sys
import pickle
import requests
from datetime import datetime

# --- Project path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from models.dog import Dog
from models.race_participation import RaceParticipation

NUM_BUCKETS = 100
DOGS_DIR = os.path.join(project_root, "data", "dogs_enhanced")
os.makedirs(DOGS_DIR, exist_ok=True)

def get_bucket_index(dog_id: str) -> int:
    return int(dog_id) % NUM_BUCKETS

def load_dogs_bucket(bucket_idx: int):
    path = os.path.join(DOGS_DIR, f"dogs_bucket_{bucket_idx}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_dogs_bucket(bucket_idx: int, dogs_dict):
    path = os.path.join(DOGS_DIR, f"dogs_bucket_{bucket_idx}.pkl")
    with open(path, "wb") as f:
        pickle.dump(dogs_dict, f)
    print(f"💾 Saved bucket {bucket_idx} ({len(dogs_dict)} dogs)")

def fetch_dog_races_from_api(dog_id):
    url = f"https://api.gbgb.org.uk/api/results/dog/{dog_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
    except Exception as e:
        print(f"❌ API error for dog {dog_id}: {e}")
        return []

def participation_from_api_item(item, dog_id):
    from datetime import datetime  # Ensure import
    # Parse date
    race_date = item.get("raceDate")
    race_datetime = None
    if race_date:
        try:
            race_datetime = datetime.strptime(race_date, "%d/%m/%Y")
        except Exception:
            race_datetime = None
    return RaceParticipation(
        dog_id = str(dog_id),
        race_id = str(item.get("raceId", "")),
        meeting_id = str(item.get("meetingId", "")),
        race_datetime = race_datetime,
        track_name = item.get("trackName", ""),
        trap_number = item.get("trapNumber", ""),
        position = item.get("resultPosition", ""),
        run_time = float(item.get("resultRunTime", "")) if item.get("resultRunTime") else None,
        split_time = float(item.get("resultSectionalTime", "")) if item.get("resultSectionalTime") else None,
        sp = item.get("SP", ""),
        weight = float(item.get("resultDogWeight", "")) if item.get("resultDogWeight") else None,
        race_class = item.get("raceClass", ""),
        going = float(item.get("raceGoing", "")) if item.get("raceGoing") else None,
        btn_distance = item.get("resultBtnDistance", ""),
        adjusted_time = float(item.get("resultAdjustedTime", "")) if item.get("resultAdjustedTime") else None,
        winner_id = str(item.get("winnerOr2ndId", "")),
        comment = item.get("resultComment", "")
    )

def build_dog_from_api(dog_id, items):
    # Try to get name/trainer from the first item with a name
    name, trainer = None, None
    for item in items:
        if item.get("dogName"):
            name = item["dogName"]
        if item.get("trainerName"):
            trainer = item["trainerName"]
        if name and trainer:
            break
    dog = Dog(dog_id=dog_id)
    dog.name = name
    dog.trainer = trainer
    dog.race_participations = []
    for item in items:
        rp = participation_from_api_item(item, dog_id)  # Pass dog_id here!
        if rp:
            dog.race_participations.append(rp)
    return dog

def add_or_update_dog(dog_id):
    print(f"🔎 Fetching data for dog {dog_id}...")
    items = fetch_dog_races_from_api(dog_id)
    if not items:
        print(f"❌ No races found for dog {dog_id}.")
        return
    dog = build_dog_from_api(dog_id, items)
    print(f"✅ Dog '{dog.name}' with {len(dog.race_participations)} participations.")
    bucket_idx = get_bucket_index(dog_id)
    dogs_dict = load_dogs_bucket(bucket_idx)
    dogs_dict[str(dog_id)] = dog
    save_dogs_bucket(bucket_idx, dogs_dict)
    print(f"🎉 Dog {dog_id} added/updated in bucket {bucket_idx}.")

if __name__ == "__main__":
    dog_id = input("Enter dog ID to scrape and add: ").strip()
    add_or_update_dog(dog_id)