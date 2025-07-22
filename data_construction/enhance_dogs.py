import os
import sys
import pickle
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from typing import Optional, Dict, List

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog

# Configuration
NUM_BUCKETS = 100
MAX_WORKERS = 15  # Concurrent API calls
BATCH_SIZE = 25   # Process dogs in batches
API_TIMEOUT = 5
SAVE_PROGRESS_EVERY = 50

# Paths - Fix these to match your actual directory structure
dogs_dir = "../data/dogs_enhanced"  # Changed from "../data/dogs_enhanced"
enhanced_dogs_dir = "../data/dogs_enhanced"  # Changed from "../data/dogs_enhanced"

# Ensure output directory exists
os.makedirs(enhanced_dogs_dir, exist_ok=True)

# Global session for connection pooling
session = None
session_lock = threading.Lock()

def get_session():
    """Get or create a global requests session for connection pooling"""
    global session
    with session_lock:
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=30,
                pool_maxsize=30,
                max_retries=1
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            })
        return session

@lru_cache(maxsize=5000)
def fetch_meeting_data_cached(meeting_id):
    """Cached API call to fetch meeting data"""
    url = f"https://api.gbgb.org.uk/api/results/meeting/{meeting_id}"
    
    try:
        response = get_session().get(url, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def extract_dog_info_from_meeting(meeting_data, target_dog_id):
    """Extract dog information from meeting data"""
    if isinstance(meeting_data, dict):
        meeting = meeting_data
    elif isinstance(meeting_data, list) and meeting_data:
        meeting = meeting_data[0]
    else:
        return None
    races = meeting.get('races', [])
    for race in races:
        for trap in race.get('traps', []):
            if str(trap.get('dogId')) == str(target_dog_id):
                return {
                    'name': trap.get('dogName', ''),
                    'sire': trap.get('dogSire', ''),
                    'dam': trap.get('dogDam', ''),
                    'born': trap.get('dogBorn', ''),
                    'colour': trap.get('dogColour', ''),
                    'sex': trap.get('dogSex', ''),
                    'trainer': trap.get('trainerName', ''),
                    'owner': trap.get('ownerName', '')
                }
    return None

def get_bucket_index(dog_id: str) -> int:
    """Get bucket index for a dog ID"""
    return int(dog_id) % NUM_BUCKETS

def load_dogs_bucket(bucket_idx: int) -> Optional[Dict[str, Dog]]:
    """Load a bucket of dogs from pickle file"""
    # Try multiple possible locations based on where build_and_save_dogs.py saves
    possible_paths = [
        os.path.join("../data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # Main project data directory
        os.path.join("data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # If run from project root
        os.path.join("../data/dogs", f"dogs_bucket_{bucket_idx}.pkl"),  # Alternative location
        os.path.join("data/dogs", f"dogs_bucket_{bucket_idx}.pkl"),  # Alternative from project root
    ]
    
    for bucket_path in possible_paths:
        abs_path = os.path.abspath(bucket_path)
        if os.path.exists(bucket_path):
            try:
                with open(bucket_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"❌ Error loading bucket {bucket_idx} from {bucket_path}: {e}")
                continue
    
    print(f"❌ Bucket {bucket_idx} not found in any location")
    print(f"💡 Looking for buckets in:")
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"  - {path} -> {abs_path} ({'✅' if os.path.exists(path) else '❌'})")
    
    # Debug: Show what files actually exist in current directory
    print(f"💡 Current working directory: {os.getcwd()}")
    if os.path.exists("../data"):
        print(f"💡 Contents of ../data/ directory:")
        for item in os.listdir("../data"):
            item_path = os.path.join("../data", item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                bucket_files = [f for f in files if f.startswith('dogs_bucket_')]
                print(f"  📁 {item}: {len(bucket_files)} bucket files")
    
    return None

def save_dogs_bucket(bucket_idx: int, dogs_dict: Dict[str, Dog], enhanced: bool = False):
    """Save a bucket of dogs to pickle file"""
    # Save to the correct directory that matches where the buckets actually are
    if enhanced:
        output_dir = "../data/dogs_enhanced"  # Main project directory
    else:
        output_dir = "../data/dogs"  # Main project directory
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    bucket_path = os.path.join(output_dir, f"dogs_bucket_{bucket_idx}.pkl")
    
    try:
        with open(bucket_path, "wb") as f:
            pickle.dump(dogs_dict, f)
        print(f"💾 Saved to: {os.path.abspath(bucket_path)}")  # Debug: show where it's saved
        return True
    except Exception as e:
        print(f"❌ Error saving bucket {bucket_idx}: {e}")
        return False

def dog_needs_enhancement(dog: Dog) -> bool:
    """A dog needs enhancement if it has no name."""
    return not dog.name or dog.name == ''

def get_meeting_ids_for_dog(dog: Dog) -> List[str]:
    """Get meeting IDs from dog's race participations with improved extraction"""
    meeting_ids = []
    
    for participation in dog.race_participations:
        meeting_id = None
        
        # Method 1: Use the meeting_id attribute (now available!)
        if hasattr(participation, 'meeting_id') and participation.meeting_id:
            meeting_id = str(int(participation.meeting_id))  # Convert float to int to string
        
        # Method 2: Fallback to race_id if meeting_id not available
        elif hasattr(participation, 'race_id') and participation.race_id:
            meeting_id = str(participation.race_id)
        
        if meeting_id and meeting_id not in meeting_ids:
            meeting_ids.append(meeting_id)
    
    # Return first 5 meetings to avoid too many API calls
    return meeting_ids[:5]

def try_direct_dog_api_call(dog_id: str) -> Optional[Dict]:
    """Try to get dog info directly from the dog API endpoint"""
    url = f"https://api.gbgb.org.uk/api/results/dog/{dog_id}"
    
    try:
        response = get_session().get(url, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        
        # Extract basic info from the response
        items = data.get("items", [])
        if items:
            # Get info from the first race record
            first_item = items[0]
            return {
                'name': first_item.get('dogName', ''),
                'trainer': first_item.get('trainerName', ''),
                'owner': first_item.get('ownerName', ''),
                'track': first_item.get('trackName', ''),
                'sire': '',  # Not available in this endpoint
                'dam': '',   # Not available in this endpoint
                'born': '',  # Not available in this endpoint
                'colour': '', # Not available in this endpoint
                'sex': ''    # Not available in this endpoint
            }
    except Exception as e:
        print(f"    ⚠️ Direct API call failed for {dog_id}: {e}")
        return None
    
    return None

def try_race_api_call(race_id: str, dog_id: str) -> Optional[Dict]:
    """Try to get dog info from race API endpoint"""
    url = f"https://api.gbgb.org.uk/api/results/race/{race_id}"
    
    try:
        response = get_session().get(url, timeout=API_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        
        # Look for our dog in the race results
        traps = data.get('traps', [])
        for trap in traps:
            if str(trap.get('dogId', '')) == str(dog_id):
                return {
                    'name': trap.get('dogName', ''),
                    'trainer': trap.get('trainerName', ''),
                    'owner': trap.get('ownerName', ''),
                    'sire': trap.get('dogSire', ''),
                    'dam': trap.get('dogDam', ''),
                    'born': trap.get('dogBorn', ''),
                    'colour': trap.get('dogColour', ''),
                    'sex': trap.get('dogSex', '')
                }
    except Exception:
        return None
    
    return None

def enhance_single_dog(dog_data):
    """Enhanced version with multiple API strategies and better fallback/debugging"""
    dog_id, dog = dog_data

    # Try direct dog API (basic info)
    dog_info = try_direct_dog_api_call(dog_id)
    time.sleep(0.05)
    direct_success = False
    if dog_info:
        if dog_info.get('name') and (not dog.name or dog.name == ''):
            dog.set_name(dog_info['name'])
            direct_success = True
        if dog_info.get('trainer') and (not dog.trainer or dog.trainer == ''):
            dog.set_trainer(dog_info['trainer'])
        # After setting name, trainer, birth_date, color...
        if dog_info.get('sire'):
            dog.sire = dog_info['sire']  # Save as string for now
        if dog_info.get('dam'):
            dog.dam = dog_info['dam']    # Save as string for now    
        if direct_success:
            return dog_id, dog, True, "direct_api"

    # Try all meeting_ids from all participations
    meeting_ids = get_meeting_ids_for_dog(dog)
    tried_meeting = False
    if meeting_ids:
        for meeting_id in meeting_ids:
            meeting_data = fetch_meeting_data_cached(meeting_id)
            if meeting_data:
                tried_meeting = True
                dog_info = extract_dog_info_from_meeting(meeting_data, int(dog_id))
                if dog_info and dog_info.get('name') and (not dog.name or dog.name == ''):
                    dog.set_name(dog_info['name'])
                if dog_info and dog_info.get('trainer') and (not dog.trainer or dog.trainer == ''):
                    dog.set_trainer(dog_info['trainer'])
                if dog_info and dog_info.get('sire'):
                    dog.sire = dog_info['sire']  # Save as string for now
                if dog_info and dog_info.get('dam'):
                    dog.dam = dog_info['dam']    # Save as string for now
                if dog_info and dog_info.get('name'):
                    if dog_info.get('born'):
                        try:
                            from datetime import datetime
                            birth_date = datetime.strptime(dog_info['born'], '%Y-%m-%d')
                            dog.set_birth_date(birth_date)
                        except Exception:
                            pass
                    if dog_info.get('colour'):
                        dog.set_color(dog_info['colour'])
                    return dog_id, dog, True, "meeting_api"
            time.sleep(0.05)

    # Try all race_ids from all participations
    race_ids = []
    for participation in dog.race_participations:
        if hasattr(participation, 'race_id') and participation.race_id:
            race_ids.append(str(participation.race_id))
    tried_race = False
    for race_id in race_ids:
        dog_info = try_race_api_call(race_id, dog_id)
        time.sleep(0.05)
        if dog_info and dog_info.get('name') and (not dog.name or dog.name == ''):
            dog.set_name(dog_info['name'])
        if dog_info and dog_info.get('trainer') and (not dog.trainer or dog.trainer == ''):
            dog.set_trainer(dog_info['trainer'])
        if dog_info and dog_info.get('sire'):
            dog.sire = dog_info['sire']  # Save as string for now
        if dog_info and dog_info.get('dam'):
            dog.dam = dog_info['dam']    # Save as string for now
        if dog_info and dog_info.get('name'):
            if dog_info.get('born'):
                try:
                    from datetime import datetime
                    birth_date = datetime.strptime(dog_info['born'], '%Y-%m-%d')
                    dog.set_birth_date(birth_date)
                except Exception:
                    pass
            if dog_info.get('colour'):
                dog.set_color(dog_info['colour'])
            return dog_id, dog, True, "race_api"
        tried_race = True

    # Fallback: Use direct_api.py logic to fill missing info
    try:
        from data_construction.direct_api import fetch_dog_races_from_api, build_dog_from_api
        items = fetch_dog_races_from_api(dog_id)
        if items:
            new_dog = build_dog_from_api(dog_id, items)
            # Only overwrite if missing
            if (not dog.name or dog.name == '') and new_dog.name:
                dog.set_name(new_dog.name)
            if (not dog.trainer or dog.trainer == '') and new_dog.trainer:
                dog.set_trainer(new_dog.trainer)
            if not dog.race_participations and new_dog.race_participations:
                dog.race_participations = new_dog.race_participations
            # If we filled anything, count as enhanced
            if (dog.name and dog.trainer) or dog.race_participations:
                return dog_id, dog, True, "direct_api_rebuild"
    except Exception as e:
        print(f"    ⚠️ Fallback direct_api.py failed for {dog_id}: {e}")

    # If all methods fail, log the reason
    print(f"⚠️ Could not enhance dog {dog_id}:")
    print(f"   - Name: '{dog.name}' | Trainer: '{dog.trainer}'")
    print(f"   - Participations: {len(dog.race_participations)}")
    print(f"   - Meeting IDs tried: {meeting_ids if tried_meeting else 'None'}")
    print(f"   - Race IDs tried: {race_ids if tried_race else 'None'}")
    print(f"   - API returned no data for this dog in any endpoint.")

    return dog_id, dog, False, "no_method_worked"

def enhance_dogs_bucket(bucket_idx: int) -> Dict:
    """Enhanced bucket processing with better debugging and fallback"""
    print(f"\n🔧 Processing bucket {bucket_idx}")
    
    # Load bucket
    dogs_dict = load_dogs_bucket(bucket_idx)
    if not dogs_dict:
        print(f"  ⚠️ Bucket {bucket_idx} not found or empty")
        return {'processed': 0, 'enhanced': 0, 'errors': 0}
    
    # Find dogs that need enhancement
    dogs_to_enhance = [
        (dog_id, dog) for dog_id, dog in dogs_dict.items() 
        if dog_needs_enhancement(dog)
    ]
    
    print(f"  📊 Found {len(dogs_to_enhance)} dogs needing enhancement out of {len(dogs_dict)} total")
    
    if not dogs_to_enhance:
        print(f"  ✅ All dogs in bucket {bucket_idx} already enhanced")
        return {'processed': len(dogs_dict), 'enhanced': 0, 'errors': 0}
    
    # Show debug info for first few dogs
    if len(dogs_to_enhance) > 0:
        sample_dog_id, sample_dog = dogs_to_enhance[0]
        print(f"  🔍 Sample dog {sample_dog_id}:")
        print(f"    - Current name: '{sample_dog.name}'")
        print(f"    - Current trainer: '{sample_dog.trainer}'")
        print(f"    - Race participations: {len(sample_dog.race_participations)}")
        
        if sample_dog.race_participations:
            sample_participation = sample_dog.race_participations[0]
            meeting_id = getattr(sample_participation, 'meeting_id', 'None')
            race_id = getattr(sample_participation, 'race_id', 'None')
            print(f"    - Sample meeting_id: {meeting_id}")
            print(f"    - Sample race_id: {race_id}")

    # Process dogs in batches with concurrent execution
    enhanced_count = 0
    error_count = 0
    method_stats = {'direct_api': 0, 'meeting_api': 0, 'race_api': 0, 'direct_api_rebuild': 0, 'no_method_worked': 0}
    
    # Use smaller batch size for testing
    batch_size = min(BATCH_SIZE, 10)
    
    for i in range(0, len(dogs_to_enhance), batch_size):
        batch = dogs_to_enhance[i:i + batch_size]
        print(f"    📦 Processing batch {i//batch_size + 1}: {len(batch)} dogs")
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 5)) as executor:
            future_to_dog = {
                executor.submit(enhance_single_dog, dog_data): dog_data[0] 
                for dog_data in batch
            }
            
            for future in as_completed(future_to_dog):
                dog_id = future_to_dog[future]
                try:
                    dog_id, enhanced_dog, success, method = future.result(timeout=30)
                    # Fallback: If all else failed, try direct_api.py logic as last resort
                    if not success:
                        try:
                            from data_construction.direct_api import fetch_dog_races_from_api, build_dog_from_api
                            items = fetch_dog_races_from_api(dog_id)
                            if items:
                                new_dog = build_dog_from_api(dog_id, items)
                                if new_dog.name:
                                    if (not dog.name or dog.name == ''):
                                        dog.set_name(new_dog.name)
                                    if (not dog.trainer or dog.trainer == '') and new_dog.trainer:
                                        dog.set_trainer(new_dog.trainer)
                                    if not dog.race_participations and new_dog.race_participations:
                                        dog.race_participations = new_dog.race_participations
                                    return dog_id, dog, True, "direct_api_rebuild"
                        except Exception as e:
                            print(f"      ⚠️ Final fallback direct_api.py failed for {dog_id}: {e}")

                    dogs_dict[str(dog_id)] = enhanced_dog  # Always use str key
                    method_stats[method] = method_stats.get(method, 0) + 1
                    
                    if success:
                        enhanced_count += 1
                        print(f"      ✅ {dog_id}: {enhanced_dog.name} (via {method})")
                    else:
                        print(f"      ❌ {dog_id}: No info found")
                        
                except Exception as e:
                    error_count += 1
                    print(f"      💥 {dog_id}: Error - {str(e)[:50]}")
        
        # Save after each batch
        if save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True):
            print(f"    💾 Progress saved after batch {i//batch_size + 1}")
        else:
            print(f"    ❌ Failed to save after batch {i//batch_size + 1}")
    
    return {
        'processed': len(dogs_to_enhance), 
        'enhanced': enhanced_count, 
        'errors': error_count,
        'method_stats': method_stats
    }

def get_enhancement_stats():
    """Get statistics on how many dogs need enhancement"""
    total_dogs = 0
    dogs_needing_enhancement = 0
    
    print("📊 Analyzing dogs for enhancement needs...")
    
    for bucket_idx in range(NUM_BUCKETS):
        dogs_dict = load_dogs_bucket(bucket_idx)
        if dogs_dict:
            bucket_total = len(dogs_dict)
            bucket_needs_enhancement = sum(1 for dog in dogs_dict.values() if dog_needs_enhancement(dog))
            
            total_dogs += bucket_total
            dogs_needing_enhancement += bucket_needs_enhancement
            
            if bucket_needs_enhancement > 0:
                print(f"  Bucket {bucket_idx:2d}: {bucket_needs_enhancement:4d}/{bucket_total:4d} dogs need enhancement")
    
    print(f"\n📈 Enhancement Statistics:")
    print(f"  - Total dogs: {total_dogs:,}")
    print(f"  - Dogs needing enhancement: {dogs_needing_enhancement:,}")
    print(f"  - Enhancement percentage: {(dogs_needing_enhancement/total_dogs*100):.1f}%")
    
    return total_dogs, dogs_needing_enhancement

def enhance_all_dogs(start_bucket: int = 0):
    """Main function to enhance all dogs, with optional starting bucket"""
    print("🚀 STARTING DOG ENHANCEMENT PROCESS")
    print("=" * 60)
    
    # Get initial statistics
    ##total_dogs, dogs_needing_enhancement = get_enhancement_stats()
    
    #if dogs_needing_enhancement == 0:
       ##print("✅ All dogs already enhanced!")
       # return
    
    # Estimate runtime
    #estimated_time_hours = (dogs_needing_enhancement * 3) / 3600  # 3 seconds per dog average
    #print(f"\n⏱️ Estimated runtime: {estimated_time_hours:.1f} hours")
    
    response = input(f"\n⚠️ This will enhance 5 dogs. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    # Ask for starting bucket if not default
    if start_bucket == 0:
        start_bucket_input = input("Enter starting bucket (0-99, default 0): ").strip()
        if start_bucket_input.isdigit():
            start_bucket = int(start_bucket_input)
        else:
            start_bucket = 0

    # Process all buckets
    start_time = time.time()
    total_stats = {'processed': 0, 'enhanced': 0, 'errors': 0}
    
    print(f"\n🔍 Processing buckets {start_bucket} to {NUM_BUCKETS-1}...")
    
    for bucket_idx in range(start_bucket, NUM_BUCKETS):
        bucket_stats = enhance_dogs_bucket(bucket_idx)
        
        # Update totals
        for key in total_stats:
            total_stats[key] += bucket_stats[key]
        
        # Progress update
        if (bucket_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time_per_bucket = elapsed / (bucket_idx - start_bucket + 1)
            remaining_buckets = NUM_BUCKETS - (bucket_idx + 1)
            eta_minutes = (remaining_buckets * avg_time_per_bucket) / 60
            
            print(f"\n📊 Progress: {bucket_idx + 1}/{NUM_BUCKETS} buckets ({(bucket_idx + 1)/NUM_BUCKETS*100:.1f}%)")
            print(f"⏰ ETA: {eta_minutes:.1f} minutes remaining")
            print(f"✅ Enhanced so far: {total_stats['enhanced']} dogs")
    
    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("🎉 DOG ENHANCEMENT COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🐕 Dogs processed: {total_stats['processed']:,}")
    print(f"✅ Dogs enhanced: {total_stats['enhanced']:,}")
    print(f"❌ Errors: {total_stats['errors']:,}")
    print(f"📈 Success rate: {(total_stats['enhanced']/total_stats['processed']*100):.1f}%")
    print(f"⚡ Speed: {total_stats['processed']/total_time:.1f} dogs/second")
    print(f"📁 Enhanced data saved to: {enhanced_dogs_dir}")

def test_enhancement(num_buckets: int = 1):
    """Test enhancement on a few buckets first"""
    print(f"🧪 TESTING ENHANCEMENT ON {num_buckets} BUCKET(S)")
    print("=" * 50)
    
    total_stats = {'processed': 0, 'enhanced': 0, 'errors': 0}
    total_method_stats = {'direct_api': 0, 'meeting_api': 0, 'race_api': 0, 'alternative_api': 0, 'no_method_worked': 0}
    
    for bucket_idx in range(num_buckets):
        bucket_stats = enhance_dogs_bucket(bucket_idx)
        for key in ['processed', 'enhanced', 'errors']:
            total_stats[key] += bucket_stats[key]
        
        if 'method_stats' in bucket_stats:
            for method, count in bucket_stats['method_stats'].items():
                # Initialize method if not exists to prevent KeyError
                if method not in total_method_stats:
                    total_method_stats[method] = 0
                total_method_stats[method] += count
    
    print(f"\n🧪 Test Results:")
    print(f"  - Dogs processed: {total_stats['processed']}")
    print(f"  - Dogs enhanced: {total_stats['enhanced']}")
    print(f"  - Success rate: {(total_stats['enhanced']/max(total_stats['processed'],1)*100):.1f}%")
    print(f"  - Method breakdown:")
    for method, count in total_method_stats.items():
        if count > 0:  # Only show methods that were actually used
            print(f"    - {method}: {count}")

def load_enhanced_dog_by_id(dog_id: str) -> Optional[Dog]:
    """Load a dog from the enhanced dogs folder"""
    bucket_idx = get_bucket_index(dog_id)
    
    # Try multiple possible locations for enhanced dogs
    possible_paths = [
        os.path.join("../data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # Main project directory
        os.path.join("data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # If run from project root
        os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl"),  # Config directory
    ]
    
    for bucket_path in possible_paths:
        if os.path.exists(bucket_path):
            try:
                with open(bucket_path, "rb") as f:
                    dogs_dict = pickle.load(f)
                
                # Try different key formats to find the dog
                dog_id_str = str(dog_id)
                
                # Direct lookup
                if dog_id_str in dogs_dict:
                    return dogs_dict[dog_id_str]
                
                # Try as integer key
                try:
                    dog_id_int = int(dog_id)
                    if dog_id_int in dogs_dict:
                        return dogs_dict[dog_id_int]
                except ValueError:
                    pass
                
                # Try searching through all keys with string conversion
                for key in dogs_dict.keys():
                    if str(key) == dog_id_str:
                        return dogs_dict[key]
                
            except Exception as e:
                print(f"❌ Error loading enhanced dog {dog_id} from {bucket_path}: {e}")
                continue
    
    return None
def check_data_structure():
    """Check the actual structure of the dog data to understand the issue"""
    print("🔍 ANALYZING DOG DATA STRUCTURE")
    print("=" * 50)
    
    # Check what directories exist - updated paths
    possible_dirs = [
        "data/dogs_enhanced",  # Where build_and_save_dogs.py saves
        "../data/dogs_enhanced",  # If run from scraping directory
        "data/dogs", 
        "../data/dogs",
        "data/scraped",  # Original data location
        "../data/scraped"
    ]
    
    print("📁 Available directories:")
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                bucket_files = [f for f in files if f.startswith('dogs_bucket_')]
                pkl_files = [f for f in files if f.endswith('.pkl')]
                csv_files = [f for f in files if f.endswith('.csv')]
                print(f"  ✅ {dir_path}: {len(bucket_files)} bucket files, {len(pkl_files)} pkl files, {len(csv_files)} csv files")
            except Exception as e:
                print(f"  ⚠️ {dir_path}: Error reading - {e}")
        else:
            print(f"  ❌ {dir_path}: Not found")
    
    # Load a bucket and examine the structure
    dogs_dict = load_dogs_bucket(0)
    if not dogs_dict:
        print("\n❌ No dogs found in bucket 0")
        print("💡 Checking if build_and_save_dogs.py created the buckets...")
        
        # Check the specific directory where build_and_save_dogs.py saves
        target_dir = "data/dogs_enhanced"
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            print(f"📁 Files in {target_dir}:")
            for file in files[:10]:  # Show first 10 files
                print(f"  - {file}")
        return
    
    # Get first dog
    first_dog_id, first_dog = next(iter(dogs_dict.items()))
    print(f"\n📊 Examining dog {first_dog_id}")
    print(f"  - Name: '{first_dog.name}'")
    print(f"  - Trainer: '{first_dog.trainer}'")
    print(f"  - Race participations: {len(first_dog.race_participations)}")
    
    if first_dog.race_participations:
        print(f"\n📋 First race participation structure:")
        first_race = first_dog.race_participations[0]
        
        # Show specific attributes we care about
        important_attrs = ['meeting_id', 'race_id', 'dog_id', 'race_datetime', 'track_name']
        for attr in important_attrs:
            if hasattr(first_race, attr):
                value = getattr(first_race, attr)
                print(f"    {attr}: {value}")
        
        # Test meeting ID extraction
        meeting_ids = get_meeting_ids_for_dog(first_dog)
        print(f"\n📋 Extracted meeting IDs: {meeting_ids}")
    
    print(f"\n💡 Data insights:")
    print(f"  - Dogs now have proper meeting_id in race participations")
    print(f"  - Can use meeting API for enhancement")
    print(f"  - Should have much better success rate")

def test_single_dog(dog_id: str):
    """Test enhancement on a single dog for debugging"""
    print(f"🔍 TESTING SINGLE DOG: {dog_id}")
    print("=" * 40)
    
    # Find which bucket this dog is in
    bucket_idx = get_bucket_index(dog_id)
    print(f"💡 Dog {dog_id} should be in bucket {bucket_idx}")
    
    dogs_dict = load_dogs_bucket(bucket_idx)
    
    if not dogs_dict:
        print(f"❌ Bucket {bucket_idx} not found or empty")
        return
    
    # Debug: show what keys are actually in the dictionary
    sample_keys = list(dogs_dict.keys())[:10]
    print(f"💡 Available dogs in bucket {bucket_idx}: {sample_keys}")
    print(f"💡 Looking for dog_id: '{dog_id}' (type: {type(dog_id)})")
    print(f"💡 Total dogs in bucket: {len(dogs_dict)}")
    
    # Ensure both dog_id and keys are strings for comparison
    dog_id = str(dog_id)
    if dog_id not in dogs_dict:
        # Try to find if the dog exists with different string formatting
        found_key = None
        for key in dogs_dict.keys():
            if str(key) == dog_id:
                found_key = key
                break
        
        if found_key:
            print(f"✅ Found dog with key: '{found_key}'")
            dog = dogs_dict[found_key]
        else:
            print(f"❌ Dog {dog_id} not found in bucket {bucket_idx}")
            print(f"💡 First 10 available dogs:")
            for i, key in enumerate(list(dogs_dict.keys())[:10]):
                print(f"    {i+1}. {key}")
            return
    else:
        dog = dogs_dict[dog_id]
    
    print(f"📊 Dog info:")
    print(f"  - Current name: '{dog.name}'")
    print(f"  - Current trainer: '{dog.trainer}'")
    print(f"  - Race participations: {len(dog.race_participations)}")
    
    # Show meeting IDs
    meeting_ids = get_meeting_ids_for_dog(dog)
    print(f"  - Available meeting IDs: {meeting_ids}")
    
    # Test direct API first
    print(f"\n🔍 Testing direct dog API...")
    direct_result = try_direct_dog_api_call(dog_id)
    if direct_result:
        print(f"✅ Direct API success: {direct_result}")
    else:
        print(f"❌ Direct API failed")
    
    # Test meeting API if available
    if meeting_ids:
        print(f"\n🔍 Testing meeting API with meeting_id {meeting_ids[0]}...")
        meeting_data = fetch_meeting_data_cached(meeting_ids[0])
        if meeting_data:
            meeting_result = extract_dog_info_from_meeting(meeting_data, int(dog_id))
            if meeting_result:
                print(f"✅ Meeting API success: {meeting_result}")
            else:
                print(f"❌ Dog not found in meeting data")
        else:
            print(f"❌ Meeting API failed to fetch data")
    
    # Test the full enhancement process
    print(f"\n🚀 Running full enhancement test...")
    result = enhance_single_dog((dog_id, dog))
    dog_id_result, enhanced_dog, success, method = result
    dogs_dict[str(dog_id)] = enhanced_dog
    save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True)
    
    print(f"\n📈 Enhancement result:")
    print(f"  - Success: {success}")
    print(f"  - Method: {method}")
    print(f"  - New name: '{enhanced_dog.name}'")
    print(f"  - New trainer: '{enhanced_dog.trainer}'")

def view_enhanced_dog(dog_id: str):
    """View detailed information about an enhanced dog, including sire/dam info if available"""
    import csv

    print(f"🔍 VIEWING ENHANCED DOG: {dog_id}")
    print("=" * 40)

    # Debug: Calculate and show bucket info first
    bucket_idx = get_bucket_index(dog_id)
    print(f"💡 Dog {dog_id} should be in bucket {bucket_idx}")

    # Try to load from enhanced folder first
    enhanced_dog = load_enhanced_dog_by_id(dog_id)

    # Load dog name dictionary for sire/dam lookup
    dog_name_to_id = {}
    dog_name_dict_path = os.path.join("../data/dogs_enhanced", "dog_name_dict.csv")
    if os.path.exists(dog_name_dict_path):
        with open(dog_name_dict_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dog_name_to_id[row['dogName']] = row['dogId']

    def get_dog_by_name(name: str):
        dog_id_lookup = dog_name_to_id.get(name)
        if dog_id_lookup:
            bucket = get_bucket_index(dog_id_lookup)
            dogs_dict = load_dogs_bucket(bucket)
            if dogs_dict and dog_id_lookup in dogs_dict:
                return dogs_dict[dog_id_lookup]
        return None

    if enhanced_dog:
        print(f"✅ Found enhanced dog {dog_id}")
        print(f"📊 Enhanced Dog Information:")
        print(f"  - ID: {enhanced_dog.id}")
        print(f"  - Name: '{enhanced_dog.name if enhanced_dog.name else 'No name'}'")
        print(f"  - Trainer: '{enhanced_dog.trainer if enhanced_dog.trainer else 'No trainer'}'")
        print(f"  - Birth Date: {enhanced_dog.birth_date if enhanced_dog.birth_date else 'Unknown'}")
        print(f"  - Color: '{enhanced_dog.color if enhanced_dog.color else 'Unknown'}'")
        print(f"  - Weight: {enhanced_dog.weight if enhanced_dog.weight else 'Unknown'}")

                # Sire/Dam logic: Show as strings (raw values)
        print(f"  - Sire: ", end="")
        # Try .sire_name first, then .sire, then 'Unknown'
        sire_str = None
        if hasattr(enhanced_dog, "sire_name") and enhanced_dog.sire_name:
            sire_str = enhanced_dog.sire_name
        elif hasattr(enhanced_dog, "sire") and enhanced_dog.sire:
            # If .sire is a string, show it; if it's a Dog object, show its name
            if isinstance(enhanced_dog.sire, str):
                sire_str = enhanced_dog.sire
            elif hasattr(enhanced_dog.sire, "name"):
                sire_str = enhanced_dog.sire.name
        print(sire_str if sire_str else "Unknown")

        print(f"  - Dam: ", end="")
        dam_str = None
        if hasattr(enhanced_dog, "dam_name") and enhanced_dog.dam_name:
            dam_str = enhanced_dog.dam_name
        elif hasattr(enhanced_dog, "dam") and enhanced_dog.dam:
            if isinstance(enhanced_dog.dam, str):
                dam_str = enhanced_dog.dam
            elif hasattr(enhanced_dog.dam, "name"):
                dam_str = enhanced_dog.dam.name
        print(dam_str if dam_str else "Unknown")

        print(f"  - Race Participations: {len(enhanced_dog.race_participations)}")

        # ...existing code for race participations and statistics...
        
        if enhanced_dog.race_participations:
            print(f"\n📋 Detailed Race Participations (showing first 5):")
            print("-" * 80)
            
            # Sort by date descending to show most recent first
            sorted_participations = sorted(enhanced_dog.race_participations, 
                                         key=lambda x: x.race_datetime if x.race_datetime else datetime.min, 
                                         reverse=True)
            
            for i, race in enumerate(sorted_participations[:5]):
                print(f"  {i+1}. Race ID: {race.race_id}")
                print(f"     - Date: {race.race_datetime}")
                print(f"     - Track: {race.track_name}")
                print(f"     - Meeting ID: {getattr(race, 'meeting_id', 'None')}")
                print(f"     - Distance: {race.distance}m" if race.distance else "     - Distance: Unknown")
                print(f"     - Trap: {race.trap_number}" if race.trap_number else "     - Trap: Unknown")
                print(f"     - Position: {race.position}")
                print(f"     - Run Time: {race.run_time}s" if race.run_time else "     - Run Time: N/A")
                print(f"     - Split Time: {race.split_time}s" if race.split_time else "     - Split Time: N/A")
                print(f"     - Starting Price (SP): {race.sp}" if race.sp else "     - Starting Price: N/A")
                print(f"     - Weight: {race.weight}kg" if race.weight else "     - Weight: N/A")
                print(f"     - Race Class: {race.race_class}" if race.race_class else "     - Race Class: Unknown")
                print(f"     - Going: {race.going}" if race.going else "     - Going: Unknown")
                print(f"     - Btn Distance: {race.btn_distance}" if race.btn_distance else "     - Btn Distance: N/A")
                print(f"     - Adjusted Time: {race.adjusted_time}s" if race.adjusted_time else "     - Adjusted Time: N/A")
                print(f"     - Winner ID: {race.winner_id}" if race.winner_id else "     - Winner ID: N/A")
                print(f"     - Comments: {race.comment}" if race.comment else "     - Comments: None")
                
                # Show additional attributes if they exist
                if hasattr(race, 'color') and race.color:
                    print(f"     - Dog Color: {race.color}")
                if hasattr(race, 'birth_date') and race.birth_date:
                    print(f"     - Dog Birth Date: {race.birth_date}")
                if hasattr(race, 'sire') and race.sire:
                    print(f"     - Sire: {race.sire}")
                if hasattr(race, 'dam') and race.dam:
                    print(f"     - Dam: {race.dam}")
                
                print()  # Empty line between races
            
            # Show race statistics
            print(f"\n📊 Race Statistics:")
            all_times = [r.run_time for r in enhanced_dog.race_participations if r.run_time]
            if all_times:
                print(f"  - Best Time: {min(all_times):.2f}s")
                print(f"  - Average Time: {sum(all_times)/len(all_times):.2f}s")
                print(f"  - Worst Time: {max(all_times):.2f}s")
            
            # Position statistics
            positions = [r.position for r in enhanced_dog.race_participations if r.position and str(r.position).replace('.','').isdigit()]
            if positions:
                numeric_positions = [float(p) for p in positions]
                wins = sum(1 for p in numeric_positions if p == 1.0)
                places = sum(1 for p in numeric_positions if p <= 3.0)
                print(f"  - Wins: {wins}/{len(enhanced_dog.race_participations)} ({wins/len(enhanced_dog.race_participations)*100:.1f}%)")
                print(f"  - Places (1st-3rd): {places}/{len(enhanced_dog.race_participations)} ({places/len(enhanced_dog.race_participations)*100:.1f}%)")
                print(f"  - Average Position: {sum(numeric_positions)/len(numeric_positions):.1f}")
            
            # Track statistics
            tracks = {}
            for race in enhanced_dog.race_participations:
                if race.track_name:
                    tracks[race.track_name] = tracks.get(race.track_name, 0) + 1
            if tracks:
                print(f"  - Most raced track: {max(tracks, key=tracks.get)} ({tracks[max(tracks, key=tracks.get)]} races)")
        
        # Compare with original if available
        original_dogs = load_dogs_bucket(bucket_idx)
        if original_dogs and str(dog_id) in original_dogs:
            orig = original_dogs[str(dog_id)]
            print(f"\n📈 Enhancement Comparison:")
            print(f"  - Name: '{orig.name}' → '{enhanced_dog.name}'")
            print(f"  - Trainer: '{orig.trainer}' → '{enhanced_dog.trainer}'")
            print(f"  - Color: '{orig.color}' → '{enhanced_dog.color}'")
            
            name_improved = bool(enhanced_dog.name) and not bool(orig.name)
            trainer_improved = bool(enhanced_dog.trainer) and not bool(orig.trainer)
            color_improved = bool(enhanced_dog.color) and not bool(orig.color)
            
            if name_improved:
                print(f"  ✅ Name was enhanced!")
            if trainer_improved:
                print(f"  ✅ Trainer was enhanced!")
            if color_improved:
                print(f"  ✅ Color was enhanced!")
            if not any([name_improved, trainer_improved, color_improved]):
                print(f"  ℹ️ No enhancement needed (already had data)")
        
        return enhanced_dog
    else:
        print(f"❌ Enhanced dog {dog_id} not found")
        
        # Try to load from original folder - but use the correct bucket
        original_dogs = load_dogs_bucket(bucket_idx)
        if original_dogs:
            # Convert both to strings for comparison
            str_dog_id = str(dog_id)
            if str_dog_id in original_dogs:
                print(f"💡 Found in original dogs folder (not enhanced yet)")
                orig = original_dogs[str_dog_id]
                print(f"  - Name: '{orig.name if orig.name else 'No name'}'")
                print(f"  - Trainer: '{orig.trainer if orig.trainer else 'No trainer'}'")
                print(f"  - Races: {len(orig.race_participations)}")
                return orig
            else:
                # Try with different key types
                for key in original_dogs.keys():
                    if str(key) == str_dog_id:
                        print(f"💡 Found in original dogs folder with key '{key}' (not enhanced yet)")
                        orig = original_dogs[key]
                        print(f"  - Name: '{orig.name if orig.name else 'No name'}'")
                        print(f"  - Trainer: '{orig.trainer if orig.trainer else 'No trainer'}'")
                        print(f"  - Races: {len(orig.race_participations)}")
                        return orig
                
                print(f"❌ Dog {dog_id} not found in bucket {bucket_idx}")
                print(f"💡 Available dogs in bucket (first 10): {list(original_dogs.keys())[:10]}")
        else:
            print(f"❌ Could not load bucket {bucket_idx}")
            
        return None


def compare_enhancement_results():
    """Compare results between original and enhanced dogs"""
    print("📊 COMPARING ENHANCEMENT RESULTS")
    print("=" * 50)
    
    enhanced_stats = {'total': 0, 'with_names': 0, 'with_trainers': 0}
    original_stats = {'total': 0, 'with_names': 0, 'with_trainers': 0}
    
    # Count enhanced dogs
    for bucket_idx in range(NUM_BUCKETS):
        enhanced_path = os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl")
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, "rb") as f:
                    enhanced_dogs = pickle.load(f)
                
                for dog in enhanced_dogs.values():
                    enhanced_stats['total'] += 1
                    if dog.name:
                        enhanced_stats['with_names'] += 1
                    if dog.trainer:
                        enhanced_stats['with_trainers'] += 1
            except:
                continue
    
    # Count original dogs
    for bucket_idx in range(NUM_BUCKETS):
        original_dogs = load_dogs_bucket(bucket_idx)
        if original_dogs:
            for dog in original_dogs.values():
                original_stats['total'] += 1
                if dog.name:
                    original_stats['with_names'] += 1
                if dog.trainer:
                    original_stats['with_trainers'] += 1
    
    print(f"📈 Enhancement Results:")
    print(f"  Original Dogs:")
    print(f"    - Total: {original_stats['total']:,}")
    print(f"    - With Names: {original_stats['with_names']:,} ({original_stats['with_names']/max(original_stats['total'],1)*100:.1f}%)")
    print(f"    - With Trainers: {original_stats['with_trainers']:,} ({original_stats['with_trainers']/max(original_stats['total'],1)*100:.1f}%)")
    
    print(f"  Enhanced Dogs:")
    print(f"    - Total: {enhanced_stats['total']:,}")
    print(f"    - With Names: {enhanced_stats['with_names']:,} ({enhanced_stats['with_names']/max(enhanced_stats['total'],1)*100:.1f}%)")
    print(f"    - With Trainers: {enhanced_stats['with_trainers']:,} ({enhanced_stats['with_trainers']/max(enhanced_stats['total'],1)*100:.1f}%)")
    
    if enhanced_stats['total'] > 0:
        name_improvement = enhanced_stats['with_names'] - original_stats['with_names']
        trainer_improvement = enhanced_stats['with_trainers'] - original_stats['with_trainers']
        
        print(f"  Improvements:")
        print(f"    - Names added: {name_improvement:,}")
        print(f"    - Trainers added: {trainer_improvement:,}")

def list_sample_enhanced_dogs(count: int = 10):
    """List sample enhanced dogs to see the results"""
    print(f"📋 SAMPLE ENHANCED DOGS (First {count})")
    print("=" * 50)
    
    found_count = 0
    
    for bucket_idx in range(NUM_BUCKETS):
        if found_count >= count:
            break
        
        enhanced_path = os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl")
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, "rb") as f:
                    enhanced_dogs = pickle.load(f)
                
                for dog_id, dog in enhanced_dogs.items():
                    if found_count >= count:
                        break
                    
                    # Only show dogs that have been enhanced (have names)
                    if dog.name and dog.name != '':
                        found_count += 1
                        print(f"  {found_count}. Dog {dog_id}: {dog.name}")
                        print(f"     - Trainer: {dog.trainer if dog.trainer else 'No trainer'}")
                        print(f"     - Races: {len(dog.race_participations)}")
                        print(f"     - Bucket: {bucket_idx}")
                        print()
            except:
                continue
    
    if found_count == 0:
        print("❌ No enhanced dogs found")
        print("💡 Run the enhancement process first")
    else:
        print(f"✅ Found {found_count} enhanced dogs")

def create_dog_buckets_from_csv():
    """Create dog buckets from the CSV file if they don't exist"""
    print("🔨 CREATING DOG BUCKETS FROM CSV DATA")
    print("=" * 50)
    
    # Look for CSV files in the right locations
    csv_files = [
        "data/scraped/scraped_data.csv",  # Where build_and_save_dogs.py looks
        "dogs5.csv",
        "../data/dogs5.csv",
        "../scraped_data.csv",
        "scraped_data.csv"
    ]
    
    csv_file = None
    for file_path in csv_files:
        if os.path.exists(file_path):
            csv_file = file_path
            print(f"✅ Found CSV file: {csv_file}")
            break
    
    if not csv_file:
        print("❌ No CSV file found!")
        print("💡 Available files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        
        # Also check data/scraped directory
        scraped_dir = "data/scraped"
        if os.path.exists(scraped_dir):
            print(f"💡 Files in {scraped_dir}:")
            for file in os.listdir(scraped_dir):
                if file.endswith('.csv'):
                    print(f"  - {file}")
        return False
    
    # If CSV file is not in the expected location, copy it
    target_csv = "data/scraped/scraped_data.csv"
    if csv_file != target_csv:
        import shutil
        os.makedirs("data/scraped", exist_ok=True)
        shutil.copy2(csv_file, target_csv)
        print(f"📋 Copied {csv_file} to {target_csv}")
    
    # Run the build_and_save_dogs script
    try:
        import subprocess
        result = subprocess.run(
            ["python", "build_and_save_dogs.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Successfully created dog buckets!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Error creating buckets:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running build_and_save_dogs.py: {e}")
        return False

def check_enhancement_progress():
    """Check which buckets have been enhanced and overall progress"""
    print("📊 CHECKING ENHANCEMENT PROGRESS")
    print("=" * 50)
    
    enhanced_buckets = 0
    total_buckets = NUM_BUCKETS
    
    print(f"🔍 Checking {total_buckets} buckets for enhancement status...")
    
    for bucket_idx in range(NUM_BUCKETS):
        enhanced_path = os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl")
        original_path = os.path.join("data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl")
        
        enhanced_exists = os.path.exists(enhanced_path)
        original_exists = os.path.exists(original_path)
        
        if enhanced_exists:
            enhanced_buckets += 1
            if bucket_idx < 10:  # Show details for first 10 buckets
                try:
                    with open(enhanced_path, "rb") as f:
                        dogs_dict = pickle.load(f)
                    dogs_with_names = sum(1 for dog in dogs_dict.values() if dog.name and dog.name != '')
                    print(f"  ✅ Bucket {bucket_idx:2d}: {len(dogs_dict)} dogs, {dogs_with_names} with names")
                except Exception as e:
                    print(f"  ⚠️ Bucket {bucket_idx:2d}: Error reading - {e}")
        elif original_exists:
            if bucket_idx < 10:
                print(f"  ⏳ Bucket {bucket_idx:2d}: Original exists, not enhanced yet")
        else:
            if bucket_idx < 10:
                print(f"  ❌ Bucket {bucket_idx:2d}: No data found")
    
    print(f"\n📈 Enhancement Progress:")
    print(f"  - Buckets enhanced: {enhanced_buckets}/{total_buckets}")
    print(f"  - Progress: {(enhanced_buckets/total_buckets)*100:.1f}%")
    
    if enhanced_buckets == 0:
        print(f"\n💡 No buckets have been enhanced yet. Run option 2 to test first!")
    elif enhanced_buckets < total_buckets:
        print(f"\n💡 You're making progress! {total_buckets - enhanced_buckets} buckets remaining.")
        print(f"💡 Run option 6 to enhance all remaining buckets.")
    else:
        print(f"\n🎉 All buckets have been enhanced!")

def find_dogs_by_name_pattern(name_pattern: str, limit: int = 10):
    """Find dogs whose names contain a pattern"""
    print(f"🔍 SEARCHING FOR DOGS WITH NAME CONTAINING: '{name_pattern}'")
    print("=" * 60)
    
    found_dogs = []
    
    for bucket_idx in range(NUM_BUCKETS):
        # Try enhanced buckets first
        enhanced_path = os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl")
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, "rb") as f:
                    dogs_dict = pickle.load(f)
                
                for dog_id, dog in dogs_dict.items():
                    if dog.name and name_pattern.lower() in dog.name.lower():
                        found_dogs.append({
                            'dog_id': dog_id,
                            'name': dog.name,
                            'trainer': dog.trainer or 'Unknown',
                            'bucket': bucket_idx,
                            'races': len(dog.race_participations),
                            'enhanced': True
                        })
                        
                        if len(found_dogs) >= limit:
                            break
            except:
                continue
        
        if len(found_dogs) >= limit:
            break
    
    # If not enough found in enhanced, check original buckets
    if len(found_dogs) < limit:
        for bucket_idx in range(NUM_BUCKETS):
            original_dogs = load_dogs_bucket(bucket_idx)
            if original_dogs:
                for dog_id, dog in original_dogs.items():
                    if dog.name and name_pattern.lower() in dog.name.lower():
                        # Check if already found in enhanced
                        if not any(f['dog_id'] == dog_id for f in found_dogs):
                            found_dogs.append({
                                'dog_id': dog_id,
                                'name': dog.name,
                                'trainer': dog.trainer or 'Unknown',
                                'bucket': bucket_idx,
                                'races': len(dog.race_participations),
                                'enhanced': False
                            })
                            
                            if len(found_dogs) >= limit:
                                break
            
            if len(found_dogs) >= limit:
                break
    
    print(f"📋 Found {len(found_dogs)} dogs:")
    for i, dog_info in enumerate(found_dogs):
        status = "✅ Enhanced" if dog_info['enhanced'] else "⏳ Original"
        print(f"  {i+1}. Dog {dog_info['dog_id']}: {dog_info['name']}")
        print(f"     - Trainer: {dog_info['trainer']}")
        print(f"     - Bucket: {dog_info['bucket']}, Races: {dog_info['races']}")
        print(f"     - Status: {status}")
        print()
    
    return found_dogs

def quick_bucket_stats(bucket_idx: int):
    """Show quick stats for a specific bucket"""
    print(f"📊 BUCKET {bucket_idx} STATISTICS")
    print("=" * 40)
    
    dogs_dict = load_dogs_bucket(bucket_idx)
    if not dogs_dict:
        print(f"❌ Bucket {bucket_idx} not found")
        return
    
    total_dogs = len(dogs_dict)
    dogs_with_names = sum(1 for dog in dogs_dict.values() if dog.name and dog.name != '')
    dogs_with_trainers = sum(1 for dog in dogs_dict.values() if dog.trainer and dog.trainer != '')
    dogs_needing_enhancement = sum(1 for dog in dogs_dict.values() if dog_needs_enhancement(dog))
    
    print(f"  Total dogs: {total_dogs}")
    print(f"  Dogs with names: {dogs_with_names} ({(dogs_with_names/total_dogs)*100:.1f}%)")
    print(f"  Dogs with trainers: {dogs_with_trainers} ({(dogs_with_trainers/total_dogs)*100:.1f}%)")
    print(f"  Dogs needing enhancement: {dogs_needing_enhancement} ({(dogs_needing_enhancement/total_dogs)*100:.1f}%)")
    
    # Show sample dogs
    print(f"\n📋 Sample dogs in bucket {bucket_idx}:")
    for i, (dog_id, dog) in enumerate(list(dogs_dict.items())[:5]):
        status = "✅" if not dog_needs_enhancement(dog) else "❌"
        print(f"  {status} Dog {dog_id}: {dog.name or 'No name'} (Trainer: {dog.trainer or 'No trainer'})")

def convert_sp_to_decimal(sp_str):
    """Convert British fractional odds to decimal format using D = (a+b)/b"""
    if not sp_str or sp_str == '' or sp_str == 'N/A':
        return None
    
    try:
        # Convert to string and strip whitespace
        sp_clean = str(sp_str).strip()
        
        # Handle special cases
        if sp_clean.lower() in ['evens', 'evs', '1/1', 'evensf', 'evsf']:
            return 2.0
        if sp_clean.lower() in ['no price', 'np', 'withdrawn', 'wd', 'void']:
            return None
        
        # Remove betting suffixes (F = favourite, JF = joint favourite, C = co-favourite, etc.)
        betting_suffixes = ['F', 'JF', 'CF', 'C', 'FAV', 'JFAV', 'CFAV']
        original_sp = sp_clean
        
        for suffix in betting_suffixes:
            if sp_clean.upper().endswith(suffix):
                sp_clean = sp_clean[:-len(suffix)].strip()
                break
        
        # Handle fractional odds like "9/2", "5/1", "11/4", "5/3F", etc.
        if '/' in sp_clean:
            parts = sp_clean.split('/')
            if len(parts) == 2:
                try:
                    a = float(parts[0].strip())
                    b = float(parts[1].strip())
                    if b != 0:
                        decimal_odds = (a + b) / b
                        return round(decimal_odds, )
                except ValueError:
                    pass
        
        # Handle already decimal format
        try:
            return float(sp_clean)
        except ValueError:
            pass
            
    except Exception as e:
        # Debug: print problematic SP values
        if total_conversions < 20:  # Only show first 20 for debugging
            print(f"    ⚠️ Could not convert SP: '{sp_str}' - {e}")
    
    return None

def convert_all_sp_to_decimal():
    """Convert all SP values in all dog buckets from fractional to decimal format"""
    print("💰 CONVERTING ALL SP VALUES TO DECIMAL FORMAT")
    print("=" * 60)
    print("Formula: D = (a+b)/b where a/b is the fractional odds")
    print()
    
    total_conversions = 0
    total_participations = 0
    buckets_processed = 0
    
    for bucket_idx in range(NUM_BUCKETS):
        # Load bucket
        dogs_dict = load_dogs_bucket(bucket_idx)
        if not dogs_dict:
            continue
        
        bucket_conversions = 0
        bucket_participations = 0
        
        print(f"🔧 Processing bucket {bucket_idx}...")
        
        # Process each dog in the bucket
        for dog_id, dog in dogs_dict.items():
            # Process each race participation
            for participation in dog.race_participations:
                bucket_participations += 1
                
                # Check if SP exists and needs conversion
                if hasattr(participation, 'sp') and participation.sp:
                    original_sp = participation.sp
                    
                    # Only convert if it's not already a decimal number
                    if isinstance(original_sp, str) and ('/' in original_sp or original_sp.lower() in ['evens', 'evs']):
                        decimal_sp = convert_sp_to_decimal(original_sp)
                        if decimal_sp is not None:
                            participation.sp = decimal_sp
                            bucket_conversions += 1
                            
                            # Show first few conversions for verification
                            if total_conversions < 10:
                                print(f"    Dog {dog_id}: '{original_sp}' → {decimal_sp}")
        
        # Save the updated bucket if any conversions were made
        if bucket_conversions > 0:
            if save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True):
                print(f"  ✅ Bucket {bucket_idx}: {bucket_conversions} conversions, saved successfully")
            else:
                print(f"  ❌ Bucket {bucket_idx}: Failed to save after {bucket_conversions} conversions")
        else:
            print(f"  ⏭️ Bucket {bucket_idx}: No SP conversions needed")
        
        total_conversions += bucket_conversions
        total_participations += bucket_participations
        buckets_processed += 1
        
        # Progress update every 10 buckets
        if buckets_processed % 10 == 0:
            print(f"📊 Progress: {buckets_processed}/{NUM_BUCKETS} buckets processed")
    
    print(f"\n🎉 SP CONVERSION COMPLETED!")
    print("=" * 40)
    print(f"📊 Statistics:")
    print(f"  - Buckets processed: {buckets_processed}")
    print(f"  - Total race participations: {total_participations:,}")
    print(f"  - SP values converted: {total_conversions:,}")
    print(f"  - Conversion rate: {(total_conversions/max(total_participations,1)*100):.1f}%")
    print(f"💾 All changes saved to enhanced dog buckets")

def analyze_unconverted_sp_formats():
    """Analyze SP formats that couldn't be converted to understand the patterns"""
    print("🔍 ANALYZING UNCONVERTED SP FORMATS")
    print("=" * 50)
    
    unconverted_samples = set()
    total_checked = 0
    
    for bucket_idx in range(min(5, NUM_BUCKETS)):  # Check first 5 buckets
        dogs_dict = load_dogs_bucket(bucket_idx)
        if not dogs_dict:
            continue
        
        for dog_id, dog in dogs_dict.items():
            for participation in dog.race_participations:
                if hasattr(participation, 'sp') and participation.sp:
                    total_checked += 1
                    original_sp = participation.sp
                    
                    # Try conversion
                    if isinstance(original_sp, str):
                        decimal_sp = convert_sp_to_decimal(original_sp)
                        if decimal_sp is None and original_sp.strip() != '':
                            unconverted_samples.add(original_sp)
                            
                            if len(unconverted_samples) >= 50:  # Collect 50 samples
                                break
                if len(unconverted_samples) >= 50:
                    break
            if len(unconverted_samples) >= 50:
                break
        if len(unconverted_samples) >= 50:
            break
    
    print(f"📊 Analysis Results:")
    print(f"  - Total SP values checked: {total_checked:,}")
    print(f"  - Unconverted samples found: {len(unconverted_samples)}")
    print()
    
    if unconverted_samples:
        print("📋 Sample unconverted SP formats:")
        for i, sp in enumerate(sorted(unconverted_samples)[:20]):
            print(f"  {i+1:2d}. '{sp}'")
        
        # Analyze patterns
        print(f"\n🔍 Pattern Analysis:")
        patterns = {
            'with_F': sum(1 for sp in unconverted_samples if 'F' in sp.upper()),
            'with_JF': sum(1 for sp in unconverted_samples if 'JF' in sp.upper()),
            'with_C': sum(1 for sp in unconverted_samples if 'C' in sp.upper()),
            'with_slash': sum(1 for sp in unconverted_samples if '/' in sp),
            'numeric_only': sum(1 for sp in unconverted_samples if sp.replace('.', '').replace('-', '').isdigit()),
            'contains_letters': sum(1 for sp in unconverted_samples if any(c.isalpha() for c in sp)),
        }
        
        for pattern, count in patterns.items():
            if count > 0:
                print(f"  - {pattern}: {count} samples")
    
    return unconverted_samples



def verify_sp_conversions(sample_bucket: int = 0):
    """Verify that SP conversions were applied correctly"""
    print(f"🔍 VERIFYING SP CONVERSIONS IN BUCKET {sample_bucket}")
    print("=" * 50)
    
    dogs_dict = load_dogs_bucket(sample_bucket)
    if not dogs_dict:
        print(f"❌ Bucket {sample_bucket} not found")
        return
    
    fractional_count = 0
    decimal_count = 0
    none_count = 0
    sample_conversions = []
    
    # Check first few dogs
    for dog_id, dog in list(dogs_dict.items())[:5]:
        print(f"\n🐕 Dog {dog_id} ({dog.name or 'No name'}):")
        
        # Check first few participations
        for i, participation in enumerate(dog.race_participations[:3]):
            if hasattr(participation, 'sp') and participation.sp is not None:
                sp_value = participation.sp
                
                if isinstance(sp_value, (int, float)):
                    decimal_count += 1
                    status = "✅ Decimal"
                    sample_conversions.append(f"Race {participation.race_id}: {sp_value}")
                elif isinstance(sp_value, str) and '/' in sp_value:
                    fractional_count += 1
                    status = "⚠️ Still fractional"
                else:
                    status = f"❓ Other format: {type(sp_value)}"
                
                print(f"  Race {i+1}: SP = {sp_value} ({status})")
            else:
                none_count += 1
                print(f"  Race {i+1}: SP = None")
    
    print(f"\n📊 Sample Verification Results:")
    print(f"  - Decimal SP values: {decimal_count}")
    print(f"  - Fractional SP values: {fractional_count}")
    print(f"  - None/missing SP values: {none_count}")
    
    if fractional_count > 0:
        print(f"⚠️ Warning: {fractional_count} fractional values still found!")
        print("💡 Run convert_all_sp_to_decimal() again if needed")
    else:
        print(f"✅ All SP values are in decimal format!")
    
    if sample_conversions:
        print(f"\n📋 Sample decimal SP values:")
        for conversion in sample_conversions[:5]:
            print(f"  - {conversion}")



def debug_enhancement_failures():
    """Debug multiple dogs that failed enhancement"""
    print("🔍 DEBUGGING ENHANCEMENT FAILURES")
    print("=" * 60)
    
    # Load bucket 0 and find dogs that need enhancement but might fail
    dogs_dict = load_dogs_bucket(0)
    if not dogs_dict:
        print("❌ Cannot load bucket 0")
        return
    
    # Find dogs that need enhancement
    dogs_to_debug = []
    for dog_id, dog in dogs_dict.items():
        if dog_needs_enhancement(dog):
            dogs_to_debug.append((dog_id, dog))
            if len(dogs_to_debug) >= 5:  # Debug first 5 problematic dogs
                break
    
    print(f"🔍 Found {len(dogs_to_debug)} dogs needing enhancement. Debugging first {min(5, len(dogs_to_debug))}...")
    
    for i, (dog_id, dog) in enumerate(dogs_to_debug):
        print(f"\n{'='*20} DEBUG {i+1}/{len(dogs_to_debug)} {'='*20}")
        debug_dog_enhancement_failure(dog_id, dog)

def debug_dog_enhancement_failure(dog_id: str, dog: Dog):
    """Detailed debugging for dogs that fail to be enhanced"""
    print(f"\n🔍 DEBUGGING DOG ENHANCEMENT FAILURE: {dog_id}")
    print("=" * 60)
    
    print(f"📊 Dog Current State:")
    print(f"  - ID: {dog.id}")
    print(f"  - Name: '{dog.name}' (needs enhancement: {not dog.name or dog.name == ''})")
    print(f"  - Trainer: '{dog.trainer}' (needs enhancement: {not dog.trainer or dog.trainer == ''})")
    print(f"  - Total participations: {len(dog.race_participations)}")
    
    if not dog.race_participations:
        print("❌ No race participations found - cannot enhance without race data")
        return
    
    # Show sample participations
    print(f"\n📋 Sample Race Participations (first 3):")
    for i, participation in enumerate(dog.race_participations[:3]):
        print(f"  {i+1}. Race ID: {participation.race_id}")
        print(f"     - Date: {participation.race_datetime}")
        print(f"     - Track: {participation.track_name}")
        print(f"     - Meeting ID: {getattr(participation, 'meeting_id', 'MISSING')}")
        print(f"     - Dog ID in participation: {participation.dog_id}")
        print()
    
    # Test Method 1: Direct Dog API
    print(f"🔍 METHOD 1: Testing Direct Dog API...")
    direct_result = try_direct_dog_api_call(dog_id)
    if direct_result:
        print(f"✅ Direct API returned data: {direct_result}")
        if direct_result.get('name'):
            print(f"✅ Name found: '{direct_result['name']}'")
        else:
            print(f"⚠️ No name in direct API response")
    else:
        print(f"❌ Direct API call failed or returned None")
        
        # Test if the API endpoint itself is accessible
        try:
            import requests
            url = f"https://api.gbgb.org.uk/api/results/dog/{dog_id}"
            response = get_session().get(url, timeout=API_TIMEOUT)
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 404:
                print(f"❌ Dog {dog_id} does not exist in GBGB API")
            elif response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                print(f"📊 API returned {len(items)} items")
                if items:
                    first_item = items[0]
                    print(f"📋 First item keys: {list(first_item.keys())}")
                    print(f"📋 Dog name in API: '{first_item.get('dogName', 'MISSING')}'")
                    print(f"📋 Trainer name in API: '{first_item.get('trainerName', 'MISSING')}'")
                else:
                    print(f"⚠️ API response has no items")
            else:
                print(f"⚠️ Unexpected API status: {response.status_code}")
                print(f"Response text: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error testing API directly: {e}")
    
    # Test Method 2: Meeting API
    print(f"\n🔍 METHOD 2: Testing Meeting API...")
    meeting_ids = get_meeting_ids_for_dog(dog)
    print(f"📋 Extracted meeting IDs: {meeting_ids}")
    
    if not meeting_ids:
        print(f"❌ No meeting IDs found - cannot use meeting API")
        
        # Debug why no meeting IDs were found
        print(f"🔍 Debugging meeting ID extraction:")
        for i, participation in enumerate(dog.race_participations[:3]):
            print(f"  Participation {i+1}:")
            print(f"    - meeting_id attribute exists: {hasattr(participation, 'meeting_id')}")
            if hasattr(participation, 'meeting_id'):
                print(f"    - meeting_id value: {participation.meeting_id} (type: {type(participation.meeting_id)})")
                print(f"    - meeting_id is truthy: {bool(participation.meeting_id)}")
            print(f"    - race_id: {participation.race_id} (type: {type(participation.race_id)})")
    else:
        # Test each meeting ID
        for i, meeting_id in enumerate(meeting_ids[:2]):  # Test first 2
            print(f"\n  Testing meeting ID {meeting_id}:")
            try:
                meeting_data = fetch_meeting_data_cached(meeting_id)
                if meeting_data:
                    print(f"  ✅ Meeting API returned data (length: {len(meeting_data)})")
                    
                    # Look for our dog in the meeting data
                    dog_info = extract_dog_info_from_meeting(meeting_data, int(dog_id))
                    if dog_info:
                        print(f"  ✅ Dog found in meeting data: {dog_info}")
                    else:
                        print(f"  ⚠️ Dog not found in meeting data")
                        
                        # Debug why dog wasn't found
                        if len(meeting_data) > 0:
                            meeting = meeting_data[0]
                            races = meeting.get('races', [])
                            print(f"    - Meeting has {len(races)} races")
                            
                            all_dog_ids = []
                            for race in races:
                                for trap in race.get('traps', []):
                                    trap_dog_id = trap.get('dogId')
                                    if trap_dog_id:
                                        all_dog_ids.append(trap_dog_id)
                            
                            print(f"    - All dog IDs in meeting: {set(all_dog_ids)}")
                            print(f"    - Looking for dog ID: {int(dog_id)}")
                            print(f"    - Target dog ID in meeting: {int(dog_id) in all_dog_ids}")
                else:
                    print(f"  ❌ Meeting API returned None")
                    
                    # Test the meeting API directly
                    try:
                        url = f"https://api.gbgb.org.uk/api/results/meeting/{meeting_id}"
                        response = get_session().get(url, timeout=API_TIMEOUT)
                        print(f"    📡 Meeting API Status: {response.status_code}")
                        if response.status_code == 404:
                            print(f"    ❌ Meeting {meeting_id} not found in API")
                        elif response.status_code != 200:
                            print(f"    ⚠️ Meeting API error: {response.text[:100]}")
                    except Exception as e:
                        print(f"    ❌ Error testing meeting API: {e}")
                        
            except Exception as e:
                print(f"  ❌ Error testing meeting {meeting_id}: {e}")
    
    # Test Method 3: Race API
    print(f"\n🔍 METHOD 3: Testing Race API...")
    race_ids = []
    for participation in dog.race_participations:
        if hasattr(participation, 'race_id') and participation.race_id:
            race_ids.append(str(participation.race_id))
    
    print(f"📋 Available race IDs: {race_ids[:5]}")  # Show first 5
    
    if not race_ids:
        print(f"❌ No race IDs found - cannot use race API")
    else:
        # Test first race ID
        test_race_id = race_ids[0]
        print(f"\n  Testing race ID {test_race_id}:")
        try:
            race_result = try_race_api_call(test_race_id, dog_id)
            if race_result:
                print(f"  ✅ Race API returned data: {race_result}")
            else:
                print(f"  ❌ Race API returned None")
                
                # Test the race API directly
                try:
                    url = f"https://api.gbgb.org.uk/api/results/race/{test_race_id}"
                    response = get_session().get(url, timeout=API_TIMEOUT)
                    print(f"    📡 Race API Status: {response.status_code}")
                    
                    if response.status_code == 404:
                        print(f"    ❌ Race {test_race_id} not found in API")
                    elif response.status_code == 200:
                        data = response.json()
                        traps = data.get('traps', [])
                        print(f"    📊 Race has {len(traps)} traps")
                        
                        trap_dog_ids = [str(trap.get('dogId', '')) for trap in traps]
                        print(f"    📋 Dog IDs in race: {trap_dog_ids}")
                        print(f"    📋 Looking for: {dog_id}")
                        print(f"    📋 Found in race: {dog_id in trap_dog_ids}")
                    else:
                        print(f"    ⚠️ Race API error: {response.text[:100]}")
                        
                except Exception as e:
                    print(f"    ❌ Error testing race API: {e}")
                    
        except Exception as e:
            print(f"  ❌ Error testing race {test_race_id}: {e}")
    
    # Summary and recommendations
    print(f"\n💡 DEBUGGING SUMMARY:")
    print(f"  - Dog has {len(dog.race_participations)} race participations")
    print(f"  - Meeting IDs available: {len(meeting_ids) > 0}")
    print(f"  - Race IDs available: {len(race_ids) > 0}")
    print(f"  - Direct API accessible: {direct_result is not None}")
    
    recommendations = []
    if not direct_result:
        recommendations.append("Dog may not exist in current GBGB API database")
    if not meeting_ids:
        recommendations.append("Check meeting_id extraction from race participations")
    if not race_ids:
        recommendations.append("Check race_id data in participations")
    
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print(f"\n🤔 All methods seem to have data - there might be a logic issue in enhance_single_dog()")

def test_api_endpoints():
    """Test API endpoints to see if they're working"""
    print("🔍 TESTING API ENDPOINTS")
    print("=" * 50)
    
    # Test dogs that we know should work
    test_dogs = ["400000", "652000", "445700"]  # Mix of known working and potentially problematic
    
    for dog_id in test_dogs:
        print(f"\n🔍 Testing Dog {dog_id}:")
        
        # Test Direct Dog API
        try:
            url = f"https://api.gbgb.org.uk/api/results/dog/{dog_id}"
            response = get_session().get(url, timeout=10)
            print(f"  Direct API: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                print(f"    - Items: {len(items)}")
                if items:
                    first = items[0]
                    print(f"    - Dog Name: '{first.get('dogName', 'MISSING')}'")
                    print(f"    - Trainer: '{first.get('trainerName', 'MISSING')}'")
                    print(f"    - Meeting ID: {first.get('meetingId', 'MISSING')}")
            
        except Exception as e:
            print(f"  Direct API: Error - {e}")
        
        # Get meeting ID and test Meeting API
        bucket_idx = get_bucket_index(dog_id)
        dogs_dict = load_dogs_bucket(bucket_idx)
        
        if dogs_dict and dog_id in dogs_dict:
            dog = dogs_dict[dog_id]
            meeting_ids = get_meeting_ids_for_dog(dog)
            
            if meeting_ids:
                meeting_id = meeting_ids[0]
                print(f"  Testing Meeting API with ID {meeting_id}:")
                
                try:
                    url = f"https://api.gbgb.org.uk/api/results/meeting/{meeting_id}"
                    response = get_session().get(url, timeout=10)
                    print(f"    Meeting API: Status {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"    - Meeting data length: {len(data)}")
                        if data:
                            meeting = data[0]
                            races = meeting.get('races', [])
                            print(f"    - Races in meeting: {len(races)}")
                            
                            # Look for our dog
                            found = False
                            for race in races:
                                for trap in race.get('traps', []):
                                    if trap.get('dogId') == int(dog_id):
                                        found = True
                                        print(f"    - Dog found in meeting: YES")
                                        print(f"    - Dog name in meeting: '{trap.get('dogName', 'MISSING')}'")
                                        break
                                if found:
                                    break
                            
                            if not found:
                                print(f"    - Dog found in meeting: NO")
                        
                except Exception as e:
                    print(f"    Meeting API: Error - {e}")
            else:
                print(f"  No meeting IDs available for testing")

def assign_parents_from_names_by_id():
    """
    For all dog IDs in dog_name_dict.csv, if the dog exists in a bucket,
    set sire/dam as Dog objects if possible, otherwise as strings from the name dictionary.
    """
    import csv

    print("🔗 ASSIGNING PARENT DOG OBJECTS OR STRINGS FROM NAMES (by dog IDs)")
    print("=" * 60)

    # Load dog name dictionary
    dog_name_to_id = {}
    dog_id_to_name = {}
    dog_name_dict_path = os.path.join("../data/dogs_enhanced", "dog_name_dict.csv")
    if os.path.exists(dog_name_dict_path):
        with open(dog_name_dict_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['dogName'].strip()
                dog_id = str(row['dogId']).strip()
                dog_name_to_id[name.lower()] = dog_id
                dog_id_to_name[dog_id] = name

    updated_count = 0
    for dog_id, dog_name in dog_id_to_name.items():
        bucket_idx = get_bucket_index(dog_id)
        dogs_dict = load_dogs_bucket(bucket_idx)
        if not dogs_dict or dog_id not in dogs_dict:
            continue

        dog = dogs_dict[dog_id]
        changed = False

        # Sire
        sire_name = None
        if isinstance(dog.sire, str):
            sire_name = dog.sire.strip().lower()
        elif isinstance(dog.sire, Dog) and dog.sire.name:
            sire_name = dog.sire.name.strip().lower()
        if sire_name and sire_name in dog_name_to_id:
            sire_id = dog_name_to_id[sire_name]
            sire_bucket = get_bucket_index(sire_id)
            sire_dict = load_dogs_bucket(sire_bucket)
            sire_obj = None
            if sire_dict:
                if sire_id in sire_dict:
                    sire_obj = sire_dict[sire_id]
                else:
                    try:
                        sire_id_int = int(sire_id)
                        if sire_id_int in sire_dict:
                            sire_obj = sire_dict[sire_id_int]
                    except Exception:
                        pass
                    for k in sire_dict.keys():
                        if str(k) == str(sire_id):
                            sire_obj = sire_dict[k]
                            break
            if sire_obj:
                dog.sire = sire_obj
            else:
                dog.sire = dog_id_to_name.get(sire_id, sire_name.title())
            changed = True
            updated_count += 1

        # Dam
        dam_name = None
        if isinstance(dog.dam, str):
            dam_name = dog.dam.strip().lower()
        elif isinstance(dog.dam, Dog) and dog.dam.name:
            dam_name = dog.dam.name.strip().lower()
        if dam_name and dam_name in dog_name_to_id:
            dam_id = dog_name_to_id[dam_name]
            dam_bucket = get_bucket_index(dam_id)
            dam_dict = load_dogs_bucket(dam_bucket)
            dam_obj = None
            if dam_dict:
                if dam_id in dam_dict:
                    dam_obj = dam_dict[dam_id]
                else:
                    try:
                        dam_id_int = int(dam_id)
                        if dam_id_int in dam_dict:
                            dam_obj = dam_dict[dam_id_int]
                    except Exception:
                        pass
                    for k in dam_dict.keys():
                        if str(k) == str(dam_id):
                            dam_obj = dam_dict[k]
                            break
            if dam_obj:
                dog.dam = dam_obj
            else:
                dog.dam = dog_id_to_name.get(dam_id, dam_name.title())
            changed = True
            updated_count += 1

        if changed:
            save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True)
            print(f"✅ Updated parents for dog {dog_id} in bucket {bucket_idx}")

    print(f"\n🎉 Finished assigning parents. Total parent links updated: {updated_count}")

def assign_parents_in_bucket(bucket_idx: int):
    """
    Enhance parents for all dogs in a specific bucket, with progress reporting.
    """
    import csv

    print(f"\n🔗 ASSIGNING PARENT DOG OBJECTS OR STRINGS FROM NAMES IN BUCKET {bucket_idx}")
    print("=" * 60)

    # Load dog name dictionary
    dog_name_to_id = {}
    dog_id_to_name = {}
    dog_name_dict_path = os.path.join("../data/dogs_enhanced", "dog_name_dict.csv")
    if os.path.exists(dog_name_dict_path):
        with open(dog_name_dict_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['dogName'].strip()
                dog_id = str(row['dogId']).strip()
                dog_name_to_id[name.lower()] = dog_id
                dog_id_to_name[dog_id] = name

    dogs_dict = load_dogs_bucket(bucket_idx)
    if not dogs_dict:
        print(f"❌ Bucket {bucket_idx} not found or empty")
        return

    updated_count = 0
    total_dogs = len(dogs_dict)
    for i, (dog_id, dog) in enumerate(dogs_dict.items(), 1):
        changed = False

        # Sire
        sire_name = None
        if isinstance(dog.sire, str):
            sire_name = dog.sire.strip().lower()
        elif isinstance(dog.sire, Dog) and dog.sire.name:
            sire_name = dog.sire.name.strip().lower()
        if sire_name and sire_name in dog_name_to_id:
            sire_id = dog_name_to_id[sire_name]
            sire_bucket = get_bucket_index(sire_id)
            sire_dict = load_dogs_bucket(sire_bucket)
            sire_obj = None
            if sire_dict:
                if sire_id in sire_dict:
                    sire_obj = sire_dict[sire_id]
                else:
                    try:
                        sire_id_int = int(sire_id)
                        if sire_id_int in sire_dict:
                            sire_obj = sire_dict[sire_id_int]
                    except Exception:
                        pass
                    for k in sire_dict.keys():
                        if str(k) == str(sire_id):
                            sire_obj = sire_dict[k]
                            break
            if sire_obj:
                dog.sire = sire_obj
            else:
                dog.sire = dog_id_to_name.get(sire_id, sire_name.title())
            changed = True
            updated_count += 1

        # Dam
        dam_name = None
        if isinstance(dog.dam, str):
            dam_name = dog.dam.strip().lower()
        elif isinstance(dog.dam, Dog) and dog.dam.name:
            dam_name = dog.dam.name.strip().lower()
        if dam_name and dam_name in dog_name_to_id:
            dam_id = dog_name_to_id[dam_name]
            dam_bucket = get_bucket_index(dam_id)
            dam_dict = load_dogs_bucket(dam_bucket)
            dam_obj = None
            if dam_dict:
                if dam_id in dam_dict:
                    dam_obj = dam_dict[dam_id]
                else:
                    try:
                        dam_id_int = int(dam_id)
                        if dam_id_int in dam_dict:
                            dam_obj = dam_dict[dam_id_int]
                    except Exception:
                        pass
                    for k in dam_dict.keys():
                        if str(k) == str(dam_id):
                            dam_obj = dam_dict[k]
                            break
            if dam_obj:
                dog.dam = dam_obj
            else:
                dog.dam = dog_id_to_name.get(dam_id, dam_name.title())
            changed = True
            updated_count += 1

        # Progress feedback every 50 dogs
        if i % 50 == 0 or i == total_dogs:
            print(f"  Progress: {i}/{total_dogs} dogs processed, {updated_count} parents updated so far")

    if updated_count > 0:
        save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True)
        print(f"✅ Updated parents for {updated_count} dogs in bucket {bucket_idx}")
    else:
        print(f"⏭️ No parent updates needed in bucket {bucket_idx}")

def assign_parents_from_meeting_api_in_bucket(bucket_idx: int):
    """
    For all dogs in a bucket, if sire/dam is missing or empty, fetch from meeting API and set as string.
    """
    print(f"\n🔗 ASSIGNING PARENT NAMES FROM MEETING API IN BUCKET {bucket_idx}")
    print("=" * 60)

    dogs_dict = load_dogs_bucket(bucket_idx)
    if not dogs_dict:
        print(f"❌ Bucket {bucket_idx} not found or empty")
        return

    updated_count = 0
    total_dogs = len(dogs_dict)
    for i, (dog_id, dog) in enumerate(dogs_dict.items(), 1):
        changed = False

        # Only update if sire or dam is missing or empty
        needs_sire = not dog.sire or (isinstance(dog.sire, str) and dog.sire.strip() == '')
        needs_dam = not dog.dam or (isinstance(dog.dam, str) and dog.dam.strip() == '')

        if needs_sire or needs_dam:
            meeting_ids = get_meeting_ids_for_dog(dog)
            if meeting_ids:
                meeting_data = fetch_meeting_data_cached(meeting_ids[0])
                if meeting_data:
                    dog_info = extract_dog_info_from_meeting(meeting_data, int(dog_id))
                    if dog_info:
                        if needs_sire and dog_info.get('sire'):
                            dog.sire = dog_info['sire']
                            changed = True
                        if needs_dam and dog_info.get('dam'):
                            dog.dam = dog_info['dam']
                            changed = True

        if changed:
            updated_count += 1

        # Progress feedback every 50 dogs
        if i % 50 == 0 or i == total_dogs:
            print(f"  Progress: {i}/{total_dogs} dogs processed, {updated_count} parents updated so far")

    if updated_count > 0:
        save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True)
        print(f"✅ Updated parents for {updated_count} dogs in bucket {bucket_idx}")
    else:
        print(f"⏭️ No parent updates needed in bucket {bucket_idx}")

# Update the main menu to include the new options
if __name__ == "__main__":
    print("🐕 Dog Enhancement Tool")
    print("Enhances dogs with names, trainers, and other missing data using GBGB API")
    print()
    
    # Check if dog buckets exist in any of the possible locations - Updated paths
    bucket_found = False
    test_dirs = [
        "../data/dogs_enhanced", 
        "data/dogs_enhanced", 
        "../data/dogs", 
        "data/dogs"
    ]
    
    for test_dir in test_dirs:
        test_bucket = os.path.join(test_dir, "dogs_bucket_0.pkl")
        if os.path.exists(test_bucket):
            bucket_found = True
            print(f"✅ Found dog buckets in: {test_dir}")
            print(f"   Full path: {os.path.abspath(test_dir)}")
            break
    
    if not bucket_found:
        print(f"❌ Dog buckets not found in any expected location")
        print("💡 Searched in:")
        for test_dir in test_dirs:
            abs_path = os.path.abspath(test_dir)
            exists = os.path.exists(test_dir)
            print(f"  - {test_dir} -> {abs_path} ({'✅' if exists else '❌'})")
        
        print("\n💡 Available options:")
        print("  1. Run build_and_save_dogs.py first to create dog buckets")
        print("  2. Create buckets from CSV data automatically")
        print("  3. Check data structure to see what exists")
        
        choice = input("Choose option (1/2/3): ").strip()
        if choice == '2':
            if create_dog_buckets_from_csv():
                print("✅ Buckets created successfully!")
            else:
                print("❌ Failed to create buckets")
                sys.exit(1)
        elif choice == '3':
            check_data_structure()
            sys.exit(0)
        else:
            print("💡 Please run build_and_save_dogs.py manually first")
            sys.exit(1)
    
    # Show available options
    print("Available options:")
    print("1. Get enhancement statistics")
    print("2. Test enhancement (1 bucket)")
    print("3. Test single dog")
    print("4. Check data structure")
    print("5. Check enhancement progress")
    print("6. Enhance all dogs")
    print("7. View enhanced dog")
    print("8. Compare enhancement results")
    print("9. List sample enhanced dogs")
    print("10. Search dogs by name")
    print("11. Quick bucket stats")
    print("12. Convert all SP values to decimal")
    print("13. Verify SP conversions")
    print("14. Analyze unconverted SP formats")
    print("15. Debug enhancement failures")  # New option
    print("16. Test API endpoints")  # New option
    print("17. Assign parent Dog objects from names (post-enhancement)")
    print("18. Assign parent Dog objects from names in a chosen bucket (progress shown)")
    print("19. Assign parent names from meeting API in a chosen bucket (progress shown)")
    print("20. Assign parent names from meeting API in ALL buckets (progress shown)")
    print()
    choice = input("Enter choice (1-20): ").strip()

    if choice == '1':
        get_enhancement_stats()
    elif choice == '2':
        test_enhancement(1)
    elif choice == '3':
        dog_id = input("Enter dog ID to test: ").strip()
        test_single_dog(dog_id)
    elif choice == '4':
        check_data_structure()
    elif choice == '5':
        check_enhancement_progress()
    elif choice == '6':
        enhance_all_dogs()
    elif choice == '7':
        dog_id = input("Enter dog ID to view: ").strip()
        view_enhanced_dog(dog_id)
    elif choice == '8':
        compare_enhancement_results()
    elif choice == '9':
        count = input("How many dogs to show (default 10): ").strip()
        count = int(count) if count.isdigit() else 10
        list_sample_enhanced_dogs(count)
    elif choice == '10':
        name_pattern = input("Enter name pattern to search: ").strip()
        find_dogs_by_name_pattern(name_pattern)
    elif choice == '11':
        bucket_num = input("Enter bucket number (0-99): ").strip()
        if bucket_num.isdigit():
            quick_bucket_stats(int(bucket_num))
        else:
            print("Invalid bucket number")
    elif choice == '12':
        convert_all_sp_to_decimal()
    elif choice == '13':
        bucket_num = input("Enter bucket number to verify (default 0): ").strip()
        bucket_num = int(bucket_num) if bucket_num.isdigit() else 0
        verify_sp_conversions(bucket_num)
    elif choice == '14':
        analyze_unconverted_sp_formats()
    elif choice == '15':
        debug_enhancement_failures()
    elif choice == '16':
        test_api_endpoints()
    elif choice == '17':
        assign_parents_from_names_by_id()    
    elif choice == '18':
        bucket_idx = input("Enter bucket index to assign parents (0-99): ").strip()
        if bucket_idx.isdigit():
            assign_parents_in_bucket(int(bucket_idx))
        else:
            print("Invalid bucket index")
    elif choice == '19':
        bucket_idx = input("Enter bucket index to assign parents from meeting API (0-99): ").strip()
        if bucket_idx.isdigit():
            assign_parents_from_meeting_api_in_bucket(int(bucket_idx))
        else:
            print("Invalid bucket index")
    elif choice == '20':
        print("Assign parent names from meeting API in ALL buckets (progress shown)")
        for bucket_idx in range(NUM_BUCKETS):
            assign_parents_from_meeting_api_in_bucket(bucket_idx)
    else:
        print("Invalid choice")