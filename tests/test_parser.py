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

# --- Directory Setup ---
data_path = os.path.join(parent_dir, "data", "scraped", "scraped_data.csv")
output_dir = os.path.join(parent_dir, "data", "test_parsed")
os.makedirs(output_dir, exist_ok=True)

# --- Containers ---
race_participations = []
dogs = {}
tracks = {}

# --- Parse and Build ---
print("🔍 Reading and parsing first 100 rows...")
df = pd.read_csv(data_path)
for i, row in tqdm(df.iterrows(), total=min(100, len(df)), desc="Processing Rows"):
    if i >= 100:
        break

    rp = parse_race_participation(row)
    if not rp:
        continue

    race_participations.append(rp)

    # Add dog
    if rp.dog_id and rp.dog_id not in dogs:
        dogs[rp.dog_id] = Dog(dog_id=rp.dog_id)

    # Add track
    if rp.track_name and rp.track_name not in tracks:
        tracks[rp.track_name] = Track(name=rp.track_name)

# --- Save ---
with open(os.path.join(output_dir, "race_participations.pkl"), "wb") as f:
    pickle.dump(race_participations, f)
with open(os.path.join(output_dir, "dogs.pkl"), "wb") as f:
    pickle.dump(dogs, f)
with open(os.path.join(output_dir, "tracks.pkl"), "wb") as f:
    pickle.dump(tracks, f)

print("\n✅ Objects saved to:", output_dir)

# --- Reload and Print Summary ---
print("\n🔄 Reloading and previewing saved objects...")

with open(os.path.join(output_dir, "race_participations.pkl"), "rb") as f:
    rps = pickle.load(f)
with open(os.path.join(output_dir, "dogs.pkl"), "rb") as f:
    dgs = pickle.load(f)
with open(os.path.join(output_dir, "tracks.pkl"), "rb") as f:
    trks = pickle.load(f)

print(f"\n📊 Loaded {len(rps)} RaceParticipations")
print(f"🐶 Loaded {len(dgs)} Dogs")
print(f"🏟 Loaded {len(trks)} Tracks")

print("\n🎯 Sample RaceParticipations:")
for rp in rps[:5]:
    print("   ", rp)

print("\n🐕 Sample Dogs:")
for dog in list(dgs.values())[:3]:
    print("   ", dog)

print("\n🏁 Sample Tracks:")
for track in list(trks.values())[:3]:
    print("   ", track)
