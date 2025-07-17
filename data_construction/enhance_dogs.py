import os
import sys
import pickle
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from typing import Optional, Dict, List

from models.dog import Dog

# Configuration
NUM_BUCKETS = 100
MAX_WORKERS = 15  # Concurrent API calls
BATCH_SIZE = 25   # Process dogs in batches
API_TIMEOUT = 5
SAVE_PROGRESS_EVERY = 50

# Paths
dogs_dir = "data/dogs_enhanced"  # This is where build_and_save_dogs.py saves to
enhanced_dogs_dir = "data/dogs_enhanced"  # Same directory for now

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
    if not meeting_data or len(meeting_data) == 0:
        return None
    
    meeting = meeting_data[0]
    races = meeting.get('races', [])
    
    for race in races:
        for trap in race.get('traps', []):
            if trap.get('dogId') == target_dog_id:
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
        os.path.join("data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # Current working directory
        os.path.join("../data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # If run from scraping
        os.path.join("data/dogs", f"dogs_bucket_{bucket_idx}.pkl"),  # Alternative location
        os.path.join("../data/dogs", f"dogs_bucket_{bucket_idx}.pkl"),  # Alternative relative path
    ]
    
    for bucket_path in possible_paths:
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
        print(f"  - {path} ({'✅' if os.path.exists(path) else '❌'})")
    return None

def save_dogs_bucket(bucket_idx: int, dogs_dict: Dict[str, Dog], enhanced: bool = False):
    """Save a bucket of dogs to pickle file"""
    output_dir = enhanced_dogs_dir if enhanced else dogs_dir
    bucket_path = os.path.join(output_dir, f"dogs_bucket_{bucket_idx}.pkl")
    
    try:
        with open(bucket_path, "wb") as f:
            pickle.dump(dogs_dict, f)
        return True
    except Exception as e:
        print(f"❌ Error saving bucket {bucket_idx}: {e}")
        return False

def dog_needs_enhancement(dog: Dog) -> bool:
    """Check if a dog needs enhancement (missing key data)"""
    return (
        not dog.name or 
        dog.name == '' or
        not dog.trainer or 
        dog.trainer == ''
    )

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
    """Enhanced version with multiple API strategies"""
    dog_id, dog = dog_data
    
    # Method 1: Try direct dog API call first (most reliable)
    dog_info = try_direct_dog_api_call(dog_id)
    if dog_info and dog_info.get('name'):
        # Update dog object with found information
        if dog_info['name']:
            dog.set_name(dog_info['name'])
        if dog_info['trainer']:
            dog.set_trainer(dog_info['trainer'])
        
        return dog_id, dog, True, "direct_api"
    
    # Method 2: Try meeting-based approach (now should work!)
    meeting_ids = get_meeting_ids_for_dog(dog)
    
    if meeting_ids:
        for meeting_id in meeting_ids[:3]:  # Try up to 3 meetings
            meeting_data = fetch_meeting_data_cached(meeting_id)
            if meeting_data:
                dog_info = extract_dog_info_from_meeting(meeting_data, int(dog_id))
                if dog_info and dog_info.get('name'):
                    # Update dog object
                    if dog_info['name']:
                        dog.set_name(dog_info['name'])
                    if dog_info['trainer']:
                        dog.set_trainer(dog_info['trainer'])
                    if dog_info['born']:
                        try:
                            from datetime import datetime
                            birth_date = datetime.strptime(dog_info['born'], '%Y-%m-%d')
                            dog.set_birth_date(birth_date)
                        except:
                            pass
                    if dog_info['colour']:
                        dog.set_color(dog_info['colour'])
                    
                    return dog_id, dog, True, "meeting_api"
            
            time.sleep(0.05)  # Small delay between API calls
    
    # Method 3: Try using race_id from participations as fallback
    race_ids = []
    for participation in dog.race_participations:
        if hasattr(participation, 'race_id') and participation.race_id:
            race_ids.append(str(participation.race_id))
    
    # Try first few race IDs
    for race_id in race_ids[:2]:
        dog_info = try_race_api_call(race_id, dog_id)
        if dog_info and dog_info.get('name'):
            # Update dog object
            if dog_info['name']:
                dog.set_name(dog_info['name'])
            if dog_info['trainer']:
                dog.set_trainer(dog_info['trainer'])
            if dog_info['born']:
                try:
                    from datetime import datetime
                    birth_date = datetime.strptime(dog_info['born'], '%Y-%m-%d')
                    dog.set_birth_date(birth_date)
                except:
                    pass
            if dog_info['colour']:
                dog.set_color(dog_info['colour'])
            
            return dog_id, dog, True, "race_api"
    
    return dog_id, dog, False, "no_method_worked"

def enhance_dogs_bucket(bucket_idx: int) -> Dict:
    """Enhanced bucket processing with better debugging"""
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
    method_stats = {'direct_api': 0, 'meeting_api': 0, 'race_api': 0, 'no_method_worked': 0}
    
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
                    dogs_dict[dog_id] = enhanced_dog  # Update in bucket
                    method_stats[method] += 1
                    
                    if success:
                        enhanced_count += 1
                        print(f"      ✅ {dog_id}: {enhanced_dog.name} (via {method})")
                    else:
                        print(f"      ❌ {dog_id}: No info found")
                        
                except Exception as e:
                    error_count += 1
                    print(f"      💥 {dog_id}: Error - {str(e)[:50]}")
    
    # Show method statistics
    print(f"  📊 Enhancement methods used:")
    for method, count in method_stats.items():
        print(f"    - {method}: {count} dogs")
    
    # Save enhanced bucket
    if save_dogs_bucket(bucket_idx, dogs_dict, enhanced=True):
        print(f"  💾 Saved enhanced bucket {bucket_idx}")
    else:
        print(f"  ❌ Failed to save bucket {bucket_idx}")
    
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

def enhance_all_dogs():
    """Main function to enhance all dogs"""
    print("🚀 STARTING DOG ENHANCEMENT PROCESS")
    print("=" * 60)
    
    # Get initial statistics
    total_dogs, dogs_needing_enhancement = get_enhancement_stats()
    
    if dogs_needing_enhancement == 0:
        print("✅ All dogs already enhanced!")
        return
    
    # Estimate runtime
    estimated_time_hours = (dogs_needing_enhancement * 3) / 3600  # 3 seconds per dog average
    print(f"\n⏱️ Estimated runtime: {estimated_time_hours:.1f} hours")
    
    response = input(f"\n⚠️ This will enhance {dogs_needing_enhancement:,} dogs. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    # Process all buckets
    start_time = time.time()
    total_stats = {'processed': 0, 'enhanced': 0, 'errors': 0}
    
    print(f"\n🔍 Processing {NUM_BUCKETS} buckets...")
    
    for bucket_idx in range(NUM_BUCKETS):
        bucket_stats = enhance_dogs_bucket(bucket_idx)
        
        # Update totals
        for key in total_stats:
            total_stats[key] += bucket_stats[key]
        
        # Progress update
        if (bucket_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time_per_bucket = elapsed / (bucket_idx + 1)
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
        os.path.join("data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # Current directory
        os.path.join("../data/dogs_enhanced", f"dogs_bucket_{bucket_idx}.pkl"),  # If run from scraping
        os.path.join(enhanced_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl"),  # Config directory
    ]
    
    for bucket_path in possible_paths:
        if os.path.exists(bucket_path):
            try:
                with open(bucket_path, "rb") as f:
                    dogs_dict = pickle.load(f)
                return dogs_dict.get(dog_id)
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
    dogs_dict = load_dogs_bucket(bucket_idx)
    
    if not dogs_dict:
        print(f"❌ Bucket {bucket_idx} not found or empty")
        return
    
    # Debug: show what keys are actually in the dictionary
    sample_keys = list(dogs_dict.keys())[:10]
    print(f"💡 Available dogs in bucket {bucket_idx}: {sample_keys}")
    print(f"💡 Looking for dog_id: '{dog_id}' (type: {type(dog_id)})")
    print(f"💡 Sample key type: {type(sample_keys[0]) if sample_keys else 'None'}")
    
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
    
    print(f"\n📈 Enhancement result:")
    print(f"  - Success: {success}")
    print(f"  - Method: {method}")
    print(f"  - New name: '{enhanced_dog.name}'")
    print(f"  - New trainer: '{enhanced_dog.trainer}'")

def view_enhanced_dog(dog_id: str):
    """View detailed information about an enhanced dog"""
    print(f"🔍 VIEWING ENHANCED DOG: {dog_id}")
    print("=" * 40)
    
    # Try to load from enhanced folder first
    enhanced_dog = load_enhanced_dog_by_id(dog_id)
    
    if enhanced_dog:
        print(f"✅ Found enhanced dog {dog_id}")
        print(f"📊 Enhanced Dog Information:")
        print(f"  - ID: {enhanced_dog.id}")
        print(f"  - Name: '{enhanced_dog.name if enhanced_dog.name else 'No name'}'")
        print(f"  - Trainer: '{enhanced_dog.trainer if enhanced_dog.trainer else 'No trainer'}'")
        print(f"  - Birth Date: {enhanced_dog.birth_date if enhanced_dog.birth_date else 'Unknown'}")
        print(f"  - Color: '{enhanced_dog.color if enhanced_dog.color else 'Unknown'}'")
        print(f"  - Weight: {enhanced_dog.weight if enhanced_dog.weight else 'Unknown'}")
        print(f"  - Race Participations: {len(enhanced_dog.race_participations)}")
        
        if enhanced_dog.race_participations:
            print(f"\n📋 Sample Race Participations:")
            for i, race in enumerate(enhanced_dog.race_participations[:3]):
                print(f"  {i+1}. Race ID: {race.race_id}")
                print(f"     - Date: {race.race_datetime}")
                print(f"     - Track: {race.track_name}")
                print(f"     - Meeting ID: {getattr(race, 'meeting_id', 'None')}")
                print(f"     - Position: {race.position}")
        
        # Compare with original if available
        original_dog = load_dogs_bucket(get_bucket_index(dog_id))
        if original_dog and dog_id in original_dog:
            orig = original_dog[dog_id]
            print(f"\n📈 Enhancement Comparison:")
            print(f"  - Name: '{orig.name}' → '{enhanced_dog.name}'")
            print(f"  - Trainer: '{orig.trainer}' → '{enhanced_dog.trainer}'")
            
            name_improved = bool(enhanced_dog.name) and not bool(orig.name)
            trainer_improved = bool(enhanced_dog.trainer) and not bool(orig.trainer)
            
            if name_improved:
                print(f"  ✅ Name was enhanced!")
            if trainer_improved:
                print(f"  ✅ Trainer was enhanced!")
            if not name_improved and not trainer_improved:
                print(f"  ℹ️ No enhancement needed (already had data)")
        
        return enhanced_dog
    else:
        print(f"❌ Enhanced dog {dog_id} not found")
        
        # Try to load from original folder
        original_dogs = load_dogs_bucket(get_bucket_index(dog_id))
        if original_dogs and dog_id in original_dogs:
            print(f"💡 Found in original dogs folder (not enhanced yet)")
            orig = original_dogs[dog_id]
            print(f"  - Name: '{orig.name if orig.name else 'No name'}'")
            print(f"  - Trainer: '{orig.trainer if orig.trainer else 'No trainer'}'")
            print(f"  - Races: {len(orig.race_participations)}")
            return orig
        else:
            print(f"❌ Dog {dog_id} not found in either original or enhanced folders")
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

if __name__ == "__main__":
    print("🐕 Dog Enhancement Tool")
    print("Enhances dogs with names, trainers, and other missing data using GBGB API")
    print()
    
    # Check if dog buckets exist in any of the possible locations
    bucket_found = False
    for test_dir in ["data/dogs_enhanced", "../data/dogs_enhanced", "data/dogs", "../data/dogs"]:
        test_bucket = os.path.join(test_dir, "dogs_bucket_0.pkl")
        if os.path.exists(test_bucket):
            bucket_found = True
            print(f"✅ Found dog buckets in: {test_dir}")
            break
    
    if not bucket_found:
        print(f"❌ Dog buckets not found in any expected location")
        print("💡 Available options:")
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
    print("5. Test API methods")
    print("6. Enhance all dogs")
    print("7. View enhanced dog")
    print("8. Compare enhancement results")
    print("9. List sample enhanced dogs")
    print()
    
    choice = input("Enter choice (1-9): ").strip()
    
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
        print("❌ Test API methods not implemented yet")
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
    else:
        print("Invalid choice")
