import os
import sys
import pickle
import requests
import time
from datetime import datetime
from typing import Optional, Dict, Any
from tqdm import tqdm

# Fix encoding issues for Windows console
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except AttributeError:
    # Already wrapped or not needed
    pass

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog

# Configuration
API_BASE = "https://api.gbgb.org.uk/api/results/meeting"
DOGS_DIR = "../data/dogs"  # Updated path to point to correct location
BATCH_SIZE = 10  # Reduced batch size to prevent long loops
MAX_DOGS_TO_PROCESS = 100  # Add limit to prevent infinite processing

def load_dog(dog_id: str) -> Optional[Dog]:
    """Load a dog from pickle file"""
    path = os.path.join(DOGS_DIR, f"{dog_id}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading dog {dog_id}: {e}")
        return None

def save_dog(dog: Dog):
    """Save a dog to pickle file"""
    path = os.path.join(DOGS_DIR, f"{dog.id}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(dog, f)
    except Exception as e:
        print(f"❌ Error saving dog {dog.id}: {e}")

def get_all_dog_files() -> list:
    """Get list of all dog pickle files"""
    if not os.path.exists(DOGS_DIR):
        print(f"❌ Dogs directory {DOGS_DIR} does not exist!")
        return []
    
    dog_files = [f for f in os.listdir(DOGS_DIR) if f.endswith('.pkl')]
    print(f"📊 Found {len(dog_files)} dog files")
    return dog_files

def get_first_meeting_id(dog: Dog) -> Optional[str]:
    """Get the first meeting ID from dog's race participations"""
    if not dog.race_participations:
        return None
    
    # Sort participations by date and get the earliest one
    sorted_participations = sorted(dog.race_participations, key=lambda x: x.race_datetime)
    
    # Look for meetingId in the participation data
    # Note: This assumes meetingId is available in the race participation data
    # If not available, we might need to derive it from raceId or use a different approach
    for participation in sorted_participations:
        if hasattr(participation, 'meeting_id') and participation.meeting_id:
            return participation.meeting_id
    
    return None

def fetch_meeting_data(meeting_id: str) -> Optional[Dict[str, Any]]:
    """Fetch meeting data from API"""
    url = f"{API_BASE}/{meeting_id}"
    params = {"meeting": meeting_id}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching meeting {meeting_id}: {e}")
        return None

def find_dog_in_meeting(meeting_data: Dict[str, Any], dog_id: str) -> Optional[Dict[str, Any]]:
    """Find dog data in meeting races"""
    if not meeting_data or 'races' not in meeting_data:
        return None
    
    for race in meeting_data['races']:
        if 'traps' not in race:
            continue
            
        for trap in race['traps']:
            if str(trap.get('dogId', '')) == str(dog_id):
                return trap
    
    return None

def parse_birth_date(birth_str: str) -> Optional[datetime]:
    """Parse birth date string like 'May-2020' to datetime"""
    if not birth_str:
        return None
    
    try:
        # Handle format like "May-2020"
        month_year = birth_str.strip()
        if '-' in month_year:
            month_str, year_str = month_year.split('-')
            year = int(year_str)
            
            # Convert month name to number
            month_names = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            
            month = month_names.get(month_str, 1)  # Default to January if not found
            return datetime(year, month, 1)  # Use 1st day of month
            
    except Exception as e:
        print(f"⚠️ Could not parse birth date '{birth_str}': {e}")
    
    return None

def update_dog_with_meeting_data(dog: Dog, trap_data: Dict[str, Any]) -> bool:
    """Update dog with data from meeting API"""
    try:
        # Update name
        if 'dogName' in trap_data and trap_data['dogName']:
            dog.set_name(trap_data['dogName'])
        
        # Update color
        if 'dogColour' in trap_data and trap_data['dogColour']:
            dog.set_color(trap_data['dogColour'])
        
        # Update trainer
        if 'trainerName' in trap_data and trap_data['trainerName']:
            dog.set_trainer(trap_data['trainerName'])
        
        # Update weight (from resultDogWeight)
        if 'resultDogWeight' in trap_data and trap_data['resultDogWeight']:
            try:
                weight = float(trap_data['resultDogWeight'])
                dog.set_weight(weight)
            except (ValueError, TypeError):
                pass
        
        # Update birth date
        if 'dogBorn' in trap_data and trap_data['dogBorn']:
            birth_date = parse_birth_date(trap_data['dogBorn'])
            if birth_date:
                dog.set_birth_date(birth_date)
        
        # Update pedigree (sire and dam)
        sire_name = trap_data.get('dogSire')
        dam_name = trap_data.get('dogDam')
        
        if sire_name or dam_name:
            # Create placeholder Dog objects for sire and dam
            # Note: These will only have names, not full data
            sire = Dog(dog_id=f"sire_{dog.id}") if sire_name else None
            dam = Dog(dog_id=f"dam_{dog.id}") if dam_name else None
            
            if sire and sire_name:
                sire.set_name(sire_name)
            if dam and dam_name:
                dam.set_name(dam_name)
            
            if sire or dam:
                dog.set_pedigree(sire, dam)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating dog {dog.id} with meeting data: {e}")
        return False

def scrape_dog_details():
    """Main function to scrape dog details from meetings"""
    print("🚀 Starting dog details scraping...")
    
    # Get all dog files
    dog_files = get_all_dog_files()
    if not dog_files:
        print("❌ No dog files found!")
        return
    
    # Statistics
    total_dogs = len(dog_files)
    updated_dogs = 0
    failed_dogs = 0
    no_meeting_dogs = 0
    dogs_processed = 0
    
    print(f"📊 Processing {total_dogs} dogs...")
    print("=" * 60)
    
    # Process each dog
    for i, dog_file in enumerate(tqdm(dog_files, desc="Processing dogs")):
        dog_id = dog_file.replace('.pkl', '')
        dogs_processed += 1
        
        # Load dog
        dog = load_dog(dog_id)
        if not dog:
            failed_dogs += 1
            continue
        
        # Skip if dog already has name (already processed)
        if dog.name:
            if dogs_processed % 100 == 0:
                print(f"⏭️ Dog {dog_id} already has name '{dog.name}' - skipping")
            continue
        
        # Get first meeting ID - we need to implement this differently
        # For now, let's try to get it from the race participation data
        meeting_id = None
        if dog.race_participations:
            # Try to extract meeting ID from race data
            # This is a workaround - we might need to get this from the CSV data
            for participation in dog.race_participations[:5]:  # Check first 5 races
                # Try to derive meeting ID - this might need adjustment based on actual data structure
                if hasattr(participation, 'race_id') and participation.race_id:
                    # For now, we'll need to implement a way to get meeting ID
                    # This is a placeholder that might need adjustment
                    pass
        
        if not meeting_id:
            no_meeting_dogs += 1
            if dogs_processed % 50 == 0:
                print(f"⚠️ Dog {dog_id}: No meeting ID found")
            continue
        
        # Fetch meeting data
        meeting_data = fetch_meeting_data(meeting_id)
        if not meeting_data:
            failed_dogs += 1
            continue
        
        # Find dog in meeting
        trap_data = find_dog_in_meeting(meeting_data, dog_id)
        if not trap_data:
            failed_dogs += 1
            continue
        
        # Update dog with meeting data
        if update_dog_with_meeting_data(dog, trap_data):
            save_dog(dog)
            updated_dogs += 1
            print(f"✅ Updated dog {dog_id}: {dog.name}")
        else:
            failed_dogs += 1
        
        # Save progress every BATCH_SIZE dogs
        if dogs_processed % BATCH_SIZE == 0:
            print(f"📊 Progress: {dogs_processed}/{total_dogs} dogs processed")
            print(f"   ✅ Updated: {updated_dogs}")
            print(f"   ❌ Failed: {failed_dogs}")
            print(f"   ⚠️ No meeting: {no_meeting_dogs}")
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("✅ DOG DETAILS SCRAPING COMPLETED!")
    print(f"📊 Total dogs: {total_dogs}")
    print(f"✅ Successfully updated: {updated_dogs}")
    print(f"❌ Failed to update: {failed_dogs}")
    print(f"⚠️ No meeting ID found: {no_meeting_dogs}")
    print(f"📈 Success rate: {(updated_dogs/total_dogs)*100:.1f}%")

def check_and_create_dogs():
    """Check if dogs exist, if not, create them from CSV"""
    print("Checking if Dog objects exist...")
    
    if os.path.exists(DOGS_DIR) and os.listdir(DOGS_DIR):
        dog_files = [f for f in os.listdir(DOGS_DIR) if f.endswith('.pkl')]
        print("Found {} existing dog files".format(len(dog_files)))
        return True
    
    print("No dog files found. Need to create them from CSV first.")
    print("Running build_and_save_dogs.py to create Dog objects...")
    
    # Update the build_and_save_dogs.py to use the correct CSV path
    import subprocess
    
    try:
        # Set encoding for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Try to run the build script
        result = subprocess.run([
            sys.executable, 
            "build_and_save_dogs.py"
        ], capture_output=True, text=True, cwd=script_dir, env=env, encoding='utf-8')
        
        if result.returncode == 0:
            print("Successfully created Dog objects")
            print("Output: {}".format(result.stdout))
            return True
        else:
            print("Error running build_and_save_dogs.py:")
            print("stdout: {}".format(result.stdout))
            print("stderr: {}".format(result.stderr))
            return False
            
    except Exception as e:
        print("Failed to run build_and_save_dogs.py: {}".format(e))
        print("\nPlease run the following command manually:")
        print("cd {}".format(script_dir))
        print("python build_and_save_dogs.py")
        return False

def get_meeting_id_from_csv():
    """Helper function to extract meeting IDs from the original CSV"""
    import pandas as pd
    
    # Try multiple possible CSV locations
    possible_paths = [
        "../dogs5.csv",  # Parent directory
        "../data/dogs5.csv",  # Data subdirectory
        "dogs5.csv",  # Current directory
        "../scraping/dogs5.csv"  # Scraping directory
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print("Found CSV file at: {}".format(path))
            break
    
    if not csv_path:
        print("CSV file not found in any of these locations:")
        for path in possible_paths:
            print("- {} (exists: {})".format(path, os.path.exists(path)))
        return {}
    
    try:
        print("Reading CSV file: {}".format(csv_path))
        # Read only first 1000 rows for testing to prevent memory issues
        df = pd.read_csv(csv_path, nrows=10000)
        print("CSV contains {} rows and {} columns".format(len(df), len(df.columns)))
        
        # Check if required columns exist
        if 'dogId' not in df.columns:
            print("CSV missing 'dogId' column")
            return {}
        if 'meetingId' not in df.columns:
            print("CSV missing 'meetingId' column")
            return {}
        
        # Create mapping of dogId to first meetingId
        dog_to_meeting = {}
        
        # Process only unique dogs to prevent duplicates
        unique_dogs = df['dogId'].unique()[:MAX_DOGS_TO_PROCESS]  # Limit number of dogs
        
        for dog_id in unique_dogs:
            dog_races = df[df['dogId'] == dog_id].sort_values('raceDate')
            if not dog_races.empty and 'meetingId' in dog_races.columns:
                first_meeting = dog_races.iloc[0]['meetingId']
                if pd.notna(first_meeting):  # Check for NaN values
                    dog_to_meeting[str(dog_id)] = str(int(first_meeting))
        
        print("Created mapping for {} dogs".format(len(dog_to_meeting)))
        return dog_to_meeting
        
    except Exception as e:
        print("Error reading CSV: {}".format(e))
        return {}

def scrape_dog_details_enhanced(dog_to_meeting_map):
    """Enhanced main function that uses meeting ID mapping"""
    print("Starting enhanced dog details scraping...")
    
    # Get all dog files
    dog_files = get_all_dog_files()
    if not dog_files:
        print("No dog files found!")
        return
    
    # Limit number of dogs to process to prevent infinite loops
    dog_files = dog_files[:MAX_DOGS_TO_PROCESS]
    
    # Statistics
    total_dogs = len(dog_files)
    updated_dogs = 0
    failed_dogs = 0
    no_meeting_dogs = 0
    dogs_processed = 0
    
    print("Processing {} dogs...".format(total_dogs))
    print("Meeting mapping available for {} dogs".format(len(dog_to_meeting_map)))
    print("=" * 60)
    
    # Process each dog
    for i, dog_file in enumerate(dog_files):
        dogs_processed += 1
        dog_id = dog_file.replace('.pkl', '')
        
        print("Processing dog {}/{}: {}".format(dogs_processed, total_dogs, dog_id))
        
        # Load dog
        dog = load_dog(dog_id)
        if not dog:
            failed_dogs += 1
            print("Failed to load dog {}".format(dog_id))
            continue
        
        # Skip if dog already has name (already processed)
        if dog.name:
            print("Dog {} already has name '{}' - skipping".format(dog_id, dog.name))
            continue
        
        # Get meeting ID from mapping
        meeting_id = dog_to_meeting_map.get(str(dog_id))
        
        if not meeting_id:
            no_meeting_dogs += 1
            print("Dog {}: No meeting ID found in mapping".format(dog_id))
            continue
        
        print("Dog {}: Using meeting ID {}".format(dog_id, meeting_id))
        
        # Fetch meeting data
        meeting_data = fetch_meeting_data(meeting_id)
        if not meeting_data:
            failed_dogs += 1
            print("Dog {}: Failed to fetch meeting {}".format(dog_id, meeting_id))
            continue
        
        # Find dog in meeting
        trap_data = find_dog_in_meeting(meeting_data, dog_id)
        if not trap_data:
            failed_dogs += 1
            print("Dog {}: Not found in meeting {}".format(dog_id, meeting_id))
            continue
        
        # Update dog with meeting data
        if update_dog_with_meeting_data(dog, trap_data):
            save_dog(dog)
            updated_dogs += 1
            print("Updated dog {}: {}".format(dog_id, dog.name))
        else:
            failed_dogs += 1
            print("Failed to update dog {}".format(dog_id))
        
        # Save progress every BATCH_SIZE dogs
        if dogs_processed % BATCH_SIZE == 0:
            print("Progress: {}/{} dogs processed".format(dogs_processed, total_dogs))
            print("   Updated: {}".format(updated_dogs))
            print("   Failed: {}".format(failed_dogs))
            print("   No meeting: {}".format(no_meeting_dogs))
        
        # Small delay to be nice to the API
        time.sleep(0.2)
        
        # Allow for keyboard interrupt
        try:
            pass
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            break
    
    # Final statistics
    print("\n" + "=" * 60)
    print("DOG DETAILS SCRAPING COMPLETED!")
    print("Total dogs: {}".format(total_dogs))
    print("Successfully updated: {}".format(updated_dogs))
    print("Failed to update: {}".format(failed_dogs))
    print("No meeting ID found: {}".format(no_meeting_dogs))
    if total_dogs > 0:
        print("Success rate: {:.1f}%".format((updated_dogs/total_dogs)*100))

if __name__ == "__main__":
    print("DOG NAME SCRAPER")
    print("=" * 50)
    
    try:
        # Step 1: Check if Dog objects exist, create if needed
        if not check_and_create_dogs():
            print("\nCannot proceed without Dog objects")
            print("To fix this:")
            print("1. Make sure dogs5.csv exists in the parent directory")
            print("2. Run: python build_and_save_dogs.py")
            print("3. Then run this script again")
            sys.exit(1)
        
        # Step 2: Create meeting ID mapping from CSV
        print("\nCreating dog to meeting ID mapping from CSV...")
        dog_to_meeting = get_meeting_id_from_csv()
        
        if not dog_to_meeting:
            print("No meeting ID mapping available. Cannot proceed.")
            sys.exit(1)
        
        # Step 3: Start scraping
        print("\nStarting enhanced scraping with meeting ID mapping...")
        scrape_dog_details_enhanced(dog_to_meeting)
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print("\nUnexpected error: {}".format(e))
        sys.exit(1)
