import requests
import os
import pandas as pd
import time

API_BASE = "https://api.gbgb.org.uk/api/results/dog"

# CSV fields exactly as in API 'items'
CSV_FIELDS = [
    "dogId",  # Add this first - we'll inject it manually
    "dogName",  # Add missing dog name
    "SP",
    "resultPosition",
    "resultBtnDistance",
    "resultSectionalTime",
    "resultComment",
    "resultRunTime",
    "resultDogWeight",
    "winnerOr2ndName",
    "winnerOr2ndId",
    "resultAdjustedTime",
    "trapNumber",
    "raceTime",
    "raceDate",
    "raceId",
    "raceNumber",  # Keep if exists in API
    "raceType",
    "raceClass",
    "raceDistance",
    "raceGoing",
    "raceWinTime",
    "meetingId",
    "trackName",
    "trainerName",  # Keep if exists in API
    "ownerName"    # Keep if exists in API
]


def fetch_items(dog_id, per_page=1000):
    """Fetch up to 'per_page' items in one request."""
    url = f"{API_BASE}/{dog_id}"
    params = {"page": 1, "itemsPerPage": per_page}
    resp = requests.get(url, params=params)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])


def normalize_item(item, dog_id):
    """Extract fields from API item and add dog_id"""
    record = {field: item.get(field, "") for field in CSV_FIELDS}
    # Override dogId since it's not in the API response
    record["dogId"] = dog_id
    return record


def save_to_csv(records, filename="dogs5.csv", header=False):
    if not records:
        return 0  # Return 0 instead of None
    df = pd.DataFrame(records, columns=CSV_FIELDS)
    df.to_csv(filename, mode="a", index=False, header=header)
    return len(records)  # Return the count of records saved


def get_existing_dog_ids(filename):
    """Retrieve existing dog IDs from the CSV file."""
    if not os.path.exists(filename):
        return set()
    df = pd.read_csv(filename, usecols=["dogId"])
    return set(df["dogId"].astype(str))


def main(start_id=600050, end_id=600050, output_file="dogs5.csv"):
    """Smart append mode - only scrapes new dogs"""
    print(f"🚀 SMART APPEND MODE: Scraping dogs {start_id} to {end_id}")
    print(f"📂 Output file: {output_file}")
    
    # Get existing dog IDs to avoid duplicates
    existing_dog_ids = get_existing_dog_ids(output_file)
    
    # Check if file exists to determine if we need header
    file_exists = os.path.exists(output_file)
    header_needed = not file_exists
    
    if file_exists:
        print(f"✅ File exists - will append new data only")
    else:
        print(f"🆕 Creating new file")
    
    print("=" * 60)
    
    total_records = 0
    successful_dogs = 0
    skipped_dogs = 0
    missing_dogs = 0
    start_time = time.time()
    
    for dog_id in range(start_id, end_id + 1):
        # Skip if already scraped
        if str(dog_id) in existing_dog_ids:
            skipped_dogs += 1
            continue
        
        items = fetch_items(dog_id)
        if items is None:
            missing_dogs += 1
            print(f"No profile for dog {dog_id}.")
            continue
        if not items:
            print(f"Dog {dog_id} has no race items.")
            continue

        # Pass dog_id to normalize_item
        records = [normalize_item(item, dog_id) for item in items]
        saved_count = save_to_csv(records, output_file, header_needed)
        
        # Fix: saved_count will always be an integer now
        total_records += saved_count
        if saved_count > 0:
            successful_dogs += 1
            header_needed = False  # Only write header once
            print(f"Saved {saved_count} records for dog {dog_id}")
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("=" * 60)
    print(f"✅ Done scraping!")
    print(f"📈 Total records saved: {total_records}")
    print(f"🐶 Successful dogs: {successful_dogs}")
    print(f"⏱️ Elapsed time: {elapsed_time:.2f} seconds")
    print(f"📂 CSV saved to: {output_file}")


if __name__ == '__main__':
    main()
