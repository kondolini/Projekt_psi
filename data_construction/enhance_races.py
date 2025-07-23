import datetime
import pytz

import os
import sys
import pickle
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Robust tqdm import: fallback to dummy if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        for x in iterable:
            yield x

# Ensure sys.path includes project root for weather_checker import
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import get_weather from weather_checker (now uses Meteostat)
try:
    from scraping.weather_checker import get_weather
except ImportError:
    # Fallback dummy get_weather
    def get_weather(date_str, time_str, place):
        print("[WARN] get_weather not available, returning None.")
        return None

# Load environment variables
load_dotenv()

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog

# Configuration
NUM_BUCKETS = int(os.getenv('NUM_BUCKETS', 100))
MAX_WORKERS = 15  # Concurrent API calls
BATCH_SIZE = 25   # Process dogs in batches
API_TIMEOUT = 5
SAVE_PROGRESS_EVERY = 50

# Paths from environment
dogs_dir = os.getenv('DOGS_DIR', 'data/dogs')
enhanced_dogs_dir = os.getenv('DOGS_ENHANCED_DIR', 'data/dogs_enhanced')

def view_races_in_bucket(bucket_idx, races_dir=None):
    if races_dir is None:
        races_dir = os.path.abspath(os.path.join(script_dir, '../data/races'))
    path = os.path.join(races_dir, f"races_bucket_{bucket_idx}.pkl")
    if not os.path.exists(path):
        print(f"Bucket {bucket_idx} not found: {path}")
        return
    with open(path, "rb") as f:
        races = pickle.load(f)
    for key, race in list(races.items())[:10]:  # Show first 10 races
        print(f"Key: {key}")
        print(f"  Race: {race}")
        # Show weather fields if present
        rainfall = getattr(race, 'rainfall_7d', None)
        temp = getattr(race, 'temperature', None)
        hum = getattr(race, 'humidity', None)
        print(f"  Weather:")
        print(f"    Rainfall 7d: {rainfall}")
        print(f"    Temperature: {temp}")
        print(f"    Humidity: {hum}")
        print(f"  Weather: rainfall_7d={getattr(race, 'rainfall_7d', None)}, temperature={getattr(race, 'temperature', None)}, humidity={getattr(race, 'humidity', None)}")

def enhance_race_bucket(bucket_idx, races_dir=None):
    """Enhance all races in a single bucket with weather data (using Meteostat)."""
    if races_dir is None:
        races_dir = os.path.abspath(os.path.join(script_dir, '../data/races'))
    bucket_path = os.path.join(races_dir, f"races_bucket_{bucket_idx}.pkl")
    if not os.path.exists(bucket_path):
        print(f"❌ Bucket {bucket_idx} not found: {bucket_path}")
        return

    with open(bucket_path, "rb") as f:
        races_dict = pickle.load(f)

    enhanced = 0
    london = pytz.timezone('Europe/London')

    # Collect races needing enhancement
    races_to_enhance = []
    for race_id, race in races_dict.items():
        rainfall_7d = getattr(race, 'rainfall_7d', None)
        temperature = getattr(race, 'temperature', None)
        humidity = getattr(race, 'humidity', None)
        needs_enhancement = False
        if rainfall_7d is None or temperature is None or humidity is None:
            needs_enhancement = True
        elif (
            isinstance(rainfall_7d, list) and len(rainfall_7d) == 7 and all(x == 0.0 for x in rainfall_7d)
            and temperature == 15.0 and humidity == 50.0
        ):
            needs_enhancement = True
        if needs_enhancement:
            races_to_enhance.append((race_id, race))


    # --- Weather cache: (date_str, place) -> weather dict ---
    weather_cache = {}

    def enhance_race_weather(args):
        race_id, race = args
        try:
            dt = None
            if hasattr(race, 'race_date') and hasattr(race, 'race_time'):
                try:
                    dt = datetime.datetime.combine(race.race_date, race.race_time)
                except Exception:
                    dt = race.race_date
            if dt is not None:
                if dt.tzinfo is None:
                    dt = london.localize(dt)
                else:
                    dt = dt.astimezone(london)
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M')
            else:
                date_str = str(race.race_date)
                time_str = str(race.race_time)
            # Patch: Replace problematic track names
            place = race.track_name
            if place == 'Star Pelaw':
                place = 'Newcastle'
            cache_key = (date_str, place)
            if cache_key in weather_cache:
                weather = weather_cache[cache_key]
            else:
                weather = get_weather(date_str, time_str, place)
                if weather and all(k in weather and weather[k] is not None for k in ('temperature', 'humidity', 'rainfall_7d')):
                    weather_cache[cache_key] = weather
            if weather and all(k in weather and weather[k] is not None for k in ('temperature', 'humidity', 'rainfall_7d')):
                race.rainfall_7d = weather.get('rainfall_7d')
                race.temperature = weather.get('temperature')
                race.humidity = weather.get('humidity')
                return True
        except Exception as e:
            print(f"[WeatherEnhance] Error for race {race_id}: {e}")
        return False

    max_workers = 8  # Tune as needed (avoid geocoding bans)
    if races_to_enhance:
        print(f"Enhancing {len(races_to_enhance)} races in parallel (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(enhance_race_weather, races_to_enhance), total=len(races_to_enhance), desc=f"Enhancing Bucket {bucket_idx}"))
        enhanced = sum(results)
    else:
        print("No races need enhancement in this bucket.")

    # Continue with rounding and saving as before
    for race_id, race in races_dict.items():
        # --- Patch: Round all run_time and related fields in race and participations ---
        # Round race_times dict (trap: time)
        if hasattr(race, 'race_times') and isinstance(race.race_times, dict):
            for trap, t in race.race_times.items():
                if t is not None:
                    try:
                        race.race_times[trap] = round(float(t), 2)
                    except Exception:
                        pass
        # Round weights dict (trap: weight)
        if hasattr(race, 'weights') and isinstance(race.weights, dict):
            for trap, w in race.weights.items():
                if w is not None:
                    try:
                        race.weights[trap] = round(float(w), 2)
                    except Exception:
                        pass
        # If race has participations, round their run_time and adjusted_time
        if hasattr(race, 'participations') and isinstance(race.participations, list):
            for p in race.participations:
                if hasattr(p, 'run_time') and p.run_time is not None:
                    try:
                        p.run_time = round(float(p.run_time), 2)
                    except Exception:
                        pass
                if hasattr(p, 'adjusted_time') and p.adjusted_time is not None:
                    try:
                        p.adjusted_time = round(float(p.adjusted_time), 2)
                    except Exception:
                        pass

    # NOTE: If you want to erase previous enhancements, you must do so manually before running this script again.
    with open(bucket_path, "wb") as f:
        pickle.dump(races_dict, f)
    print(f"✅ Enhanced {enhanced}/{len(races_dict)} races in bucket {bucket_idx}")

def enhance_all_race_buckets(races_dir=None):
    """Enhance a range of race buckets with weather data."""
    start = 0
    end = 99
    try:
        user_range = input("Enter bucket range to enhance (e.g. 0-19, default 0-99): ").strip()
        if user_range:
            if '-' in user_range:
                parts = user_range.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start = int(parts[0])
                    end = int(parts[1])
            elif user_range.isdigit():
                start = end = int(user_range)
    except Exception:
        pass
    for i in range(start, end+1):
        enhance_race_bucket(i, races_dir=races_dir)

def test_enhance_race_bucket(bucket_idx=0, races_dir=None):
    """Test enhancement on a single bucket."""
    enhance_race_bucket(bucket_idx, races_dir=races_dir)

def main_menu():
    while True:
        print("\n=== Race Enhancement Menu ===")
        print("1. View races in a bucket")
        print("2. Enhance a single race bucket with weather")
        print("3. Enhance ALL race buckets with weather")
        print("4. Test enhance (single bucket)")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            idx = input("Enter bucket index (0-99): ").strip()
            if idx.isdigit():
                view_races_in_bucket(int(idx))
        elif choice == "2":
            idx = input("Enter bucket index (0-99): ").strip()
            if idx.isdigit():
                enhance_race_bucket(int(idx))
        elif choice == "3":
            confirm = input("Enhance a range of buckets? This may take a long time. (y/N): ").strip().lower()
            if confirm == "y":
                enhance_all_race_buckets()
        elif choice == "4":
            idx = input("Enter bucket index (0-99) for test: ").strip()
            if idx.isdigit():
                test_enhance_race_bucket(int(idx))
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main_menu()