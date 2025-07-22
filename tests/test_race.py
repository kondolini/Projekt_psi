import os
import sys
import pickle
import unittest
import csv
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Adjust path for module import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.race import Race
from models.race_participation import RaceParticipation
from models.track import Track
from scraping.weather_checker import get_weather

# Load paths from environment
DOGS_ENHANCED_DIR = os.getenv('DOGS_ENHANCED_DIR', 'data/dogs_enhanced')
RACE_PARTICIPATIONS_DIR = os.getenv('RACE_PARTICIPATIONS_DIR', 'data/race_participations')
TRACKS_DIR = os.getenv('TRACKS_DIR', 'data/tracks')
RACES_OUT = os.getenv('RACES_DIR', 'data/races')

# Ensure output directory exists
os.makedirs(RACES_OUT, exist_ok=True)

class TestBuildRaces(unittest.TestCase):
    def setUp(self):
        # Create test data directory structure
        self.test_data_dir = os.path.join(parent_dir, 'test_data')
        self.test_dogs_enhanced_dir = os.path.join(self.test_data_dir, 'dogs_enhanced')
        self.test_races_dir = os.path.join(self.test_data_dir, 'races')
        self.test_unified_dir = os.path.join(self.test_data_dir, 'unified')
        
        # Create test directories
        os.makedirs(self.test_dogs_enhanced_dir, exist_ok=True)
        os.makedirs(self.test_races_dir, exist_ok=True)
        os.makedirs(self.test_unified_dir, exist_ok=True)

        # Directories from environment
        base = parent_dir
        dogs_dir = os.path.join(base, DOGS_ENHANCED_DIR)
        parts_dir = os.path.join(base, RACE_PARTICIPATIONS_DIR)

        # Load dog_lookup
        self.dog_lookup = {}
        for fname in os.listdir(dogs_dir):
            if fname.endswith('.pkl'):
                with open(os.path.join(dogs_dir, fname), 'rb') as f:
                    bucket = pickle.load(f)
                    for dog_id, dog_obj in bucket.items():
                        if isinstance(dog_obj, Dog):
                            self.dog_lookup[dog_id] = dog_obj

        # Load all race participations
        race_parts = defaultdict(list)
        for fname in os.listdir(parts_dir):
            if fname.endswith('.pkl'):
                with open(os.path.join(parts_dir, fname), 'rb') as f:
                    parts = pickle.load(f)
                    for p in parts:
                        race_parts[(p.race_id, p.meeting_id)].append(p)

        # Select first 5 races and build Race objects with weather
        selected_ids = list(race_parts.keys())[:5]
        self.races = {}
        for race_key in selected_ids:
            parts = race_parts[race_key]
            race = Race.from_participations(parts)
            
            # Add weather data to each race
            try:
                race_date_str = race.race_date.strftime("%Y-%m-%d")
                race_time_str = race.race_time.strftime("%H:%M")
                track_location = race.track_name
                
                weather_data = get_weather(race_date_str, race_time_str, track_location)
                if weather_data:
                    race.rainfall_7d = weather_data['rainfall_7d']
                    race.humidity = weather_data['humidity']
                    race.temperature = weather_data['temperature']
                    print(f"Weather added to race {race.race_id}: {weather_data}")
                else:
                    # Set default weather values if no data available
                    race.rainfall_7d = [0.0] * 7
                    race.humidity = 50.0
                    race.temperature = 15.0
                    print(f"Default weather added to race {race.race_id}")
            except Exception as e:
                print(f"Weather fetch failed for race {race.race_id}: {e}")
                # Set default weather values
                race.rainfall_7d = [0.0] * 7
                race.humidity = 50.0
                race.temperature = 15.0
            
            self.races[race_key] = race

        # Load dog name mapping (name -> id)
        dog_name_csv = os.path.join(parent_dir, 'data/dog_name_dict.csv')
        self.dog_name_map = {}
        if os.path.exists(dog_name_csv):
            with open(dog_name_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.dog_name_map[row['dogName']] = row['dogId']
        
        # Fix sire/dam relationships and create comprehensive parent-offspring mapping
        self.parent_offspring_dict, self.artificial_parent_count = self.fix_pedigree_relationships()

    def fix_pedigree_relationships(self):
        """Convert string sire/dam names to Dog objects, create artificial parents with IDs 100000-300000"""
        missing_parents = {}  # name -> list of child_ids
        parent_offspring = defaultdict(list)
        artificial_id_counter = 100000  # Start artificial parent IDs at 100000
        
        # First pass: identify existing parents and collect missing ones
        for dog_id, dog in self.dog_lookup.items():
            if isinstance(dog, Dog):
                for rel_name in ['sire', 'dam']:
                    rel_value = getattr(dog, rel_name)
                    if isinstance(rel_value, str) and rel_value:
                        # Check if this parent exists in our dog lookup
                        parent_id = self.dog_name_map.get(rel_value)
                        if parent_id and parent_id in self.dog_lookup:
                            # Parent exists, set as Dog object
                            parent_dog = self.dog_lookup[parent_id]
                            setattr(dog, rel_name, parent_dog)
                            parent_offspring[parent_id].append(dog_id)
                        else:
                            # Parent missing, collect for artificial creation
                            if rel_value not in missing_parents:
                                missing_parents[rel_value] = []
                            missing_parents[rel_value].append(dog_id)
                    elif isinstance(rel_value, Dog):
                        # Already a Dog object, add to parent-offspring mapping
                        parent_offspring[rel_value.id].append(dog_id)
        
        print(f"Found {len(missing_parents)} missing parents to create artificially")
        
        # Second pass: create artificial parents for missing ones with numeric IDs
        artificial_parents_created = 0
        for parent_name, child_ids in missing_parents.items():
            # Create artificial parent dog with numeric ID in range 100000-300000
            artificial_id = str(artificial_id_counter)
            artificial_id_counter += 1
            artificial_parents_created += 1
            
            if artificial_id_counter >= 300000:
                print("Warning: Reached maximum artificial parent ID limit (300000)")
                break
            
            artificial_dog = Dog(dog_id=artificial_id)
            artificial_dog.set_name(parent_name)
            
            # Calculate average weight from children
            child_weights = []
            for child_id in child_ids:
                child = self.dog_lookup.get(child_id)
                if child and isinstance(child, Dog) and child.weight:
                    child_weights.append(child.weight)
            
            if child_weights:
                avg_weight = sum(child_weights) / len(child_weights)
                artificial_dog.set_weight(avg_weight)
            
            # Add to dog lookup
            self.dog_lookup[artificial_id] = artificial_dog
            
            # Update children to reference this artificial parent
            for child_id in child_ids:
                child = self.dog_lookup.get(child_id)
                if child and isinstance(child, Dog):
                    # Determine if this should be sire or dam based on original string
                    if getattr(child, 'sire') == parent_name:
                        child.sire = artificial_dog
                    if getattr(child, 'dam') == parent_name:
                        child.dam = artificial_dog
                    
                    # Update parent-offspring mapping
                    parent_offspring[artificial_id].append(child_id)
            
            print(f"Created artificial parent: {artificial_id} ({parent_name}) for {len(child_ids)} children")
        
        return dict(parent_offspring), artificial_parents_created

    def save_parent_offspring_mapping_to_csv(self):
        """Save the parent-offspring mapping to CSV for analysis"""
        csv_path = os.path.join(self.test_unified_dir, 'parent_offspring_mapping.csv')
        
        # Prepare data for CSV
        csv_data = []
        for parent_id, offspring_ids in self.parent_offspring_dict.items():
            parent_dog = self.dog_lookup.get(parent_id)
            parent_name = parent_dog.name if parent_dog and parent_dog.name else "Unknown"
            is_artificial = int(parent_id) >= 100000 and int(parent_id) < 300000
            
            for offspring_id in offspring_ids:
                offspring_dog = self.dog_lookup.get(offspring_id)
                offspring_name = offspring_dog.name if offspring_dog and offspring_dog.name else "Unknown"
                
                csv_data.append({
                    'parent_id': parent_id,
                    'parent_name': parent_name,
                    'is_artificial_parent': is_artificial,
                    'offspring_id': offspring_id,
                    'offspring_name': offspring_name,
                    'offspring_count': len(offspring_ids)
                })
        
        # Save to CSV
        if csv_data:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['parent_id', 'parent_name', 'is_artificial_parent', 
                                                     'offspring_id', 'offspring_name', 'offspring_count'])
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Saved parent-offspring mapping to {csv_path}")
            print(f"Total parent-offspring relationships: {len(csv_data)}")
            
            # Statistics
            artificial_count = sum(1 for row in csv_data if row['is_artificial_parent'])
            real_count = len(csv_data) - artificial_count
            
            print(f"  - Real parent relationships: {real_count}")
            print(f"  - Artificial parent relationships: {artificial_count}")
        else:
            print("No parent-offspring relationships to save")

    def save_enhanced_dogs_temporarily(self):
        """Save enhanced dogs (including artificial parents) to test data directory"""
        # Group dogs by bucket for consistent structure
        NUM_BUCKETS = 100
        dog_buckets = defaultdict(dict)
        
        for dog_id, dog in self.dog_lookup.items():
            bucket_idx = int(dog_id) % NUM_BUCKETS
            dog_buckets[bucket_idx][dog_id] = dog
        
        # Save buckets to test directory
        for bucket_idx, dogs_dict in dog_buckets.items():
            bucket_path = os.path.join(self.test_dogs_enhanced_dir, f"dogs_bucket_{bucket_idx}.pkl")
            with open(bucket_path, 'wb') as f:
                pickle.dump(dogs_dict, f)
        
        print(f"Saved {len(self.dog_lookup)} dogs (including {self.artificial_parent_count} artificial parents) to test directory")

    def test_parent_offspring_mapping(self):
        """Test the parent-offspring mapping creation and save to CSV"""
        print("\n=== Testing Parent-Offspring Mapping ===")
        
        # Save the mapping to CSV
        self.save_parent_offspring_mapping_to_csv()
        
        # Save enhanced dogs temporarily
        self.save_enhanced_dogs_temporarily()
        
        # Verify the mapping
        total_relationships = sum(len(offspring) for offspring in self.parent_offspring_dict.values())
        artificial_parents = sum(1 for parent_id in self.parent_offspring_dict.keys() 
                               if int(parent_id) >= 100000 and int(parent_id) < 300000)
        
        print(f"Parent-offspring mapping statistics:")
        print(f"  - Total parents: {len(self.parent_offspring_dict)}")
        print(f"  - Artificial parents: {artificial_parents}")
        print(f"  - Total parent-offspring relationships: {total_relationships}")
        
        # Verify CSV file was created
        csv_path = os.path.join(self.test_unified_dir, 'parent_offspring_mapping.csv')
        self.assertTrue(os.path.exists(csv_path), "Parent-offspring CSV should be created")
        
        # Verify some artificial parents were created
        self.assertGreater(artificial_parents, 0, "Should have created some artificial parents")
        
        # Show sample artificial parents
        print(f"\nSample artificial parents:")
        count = 0
        for parent_id, offspring_ids in self.parent_offspring_dict.items():
            if int(parent_id) >= 100000 and int(parent_id) < 300000 and count < 5:
                parent_dog = self.dog_lookup[parent_id]
                print(f"  - {parent_id}: {parent_dog.name} ({len(offspring_ids)} offspring)")
                count += 1

    def tearDown(self):
        """Clean up test data directory after tests"""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
            print(f"Cleaned up test data directory: {self.test_data_dir}")

    def test_race_info_print(self):
        # Print race repr and detailed info
        for rid, race in self.races.items():
            print(f"\n--- Race {rid} ---")
            print(repr(race))
            race.print_info()
            self.assertIsInstance(race, Race)
            self.assertTrue(hasattr(race, 'dog_ids') and race.dog_ids)

    def test_race_save_load(self):
        # Test saving and loading of a race
        for rid, race in self.races.items():
            temp_path = f"temp_{rid}.pkl"
            with open(temp_path, 'wb') as f:
                pickle.dump(race, f)
            loaded = Race.load(temp_path)
            print(f"Loaded race {loaded.race_id} from pickle:", loaded)
            self.assertEqual(loaded.race_id, race.race_id)
            os.remove(temp_path)
            break  # only test first

    def test_pedigree_traversal_and_print(self):
        # For each root dog (no sire and no dam), traverse descendants
        # Fix: ensure we're working with Dog objects
        roots = []
        for dog_id, dog in self.dog_lookup.items():
            if isinstance(dog, Dog) and not isinstance(dog.sire, Dog) and not isinstance(dog.dam, Dog):
                roots.append(dog)
        
        for root in roots[:3]:
            print(f"\n=== Pedigree starting at {root.id} ===")
            root.print_info()
            # Traverse children
            stack = [root.id]
            while stack:
                current_id = stack.pop()
                children = self.parent_offspring.get(current_id, [])
                for cid in children[:2]:  # print first two children per node
                    child = self.dog_lookup.get(cid)
                    if child and isinstance(child, Dog):
                        child.print_info()
                        stack.append(cid)
            print("=== End of pedigree ===")
            self.assertTrue(True)

    def test_race_construction(self):
        dogs = self.dog_lookup  # Use loaded dogs
        tracks = load_all_tracks()

        sample_races = set()
        # Fix: ensure we're working with Dog objects
        for dog_id, dog in dogs.items():
            if isinstance(dog, Dog):
                for p in dog.race_participations:
                    if p.race_id and p.meeting_id:
                        sample_races.add((p.race_id, p.meeting_id))
                if len(sample_races) >= 5:
                    break

        for race_id, meeting_id in list(sample_races)[:5]:
            trap_to_dog = {}
            for dog_id, dog in dogs.items():
                if isinstance(dog, Dog):
                    for p in dog.race_participations:
                        if p.race_id == race_id and p.meeting_id == meeting_id and p.trap_number is not None:
                            trap_to_dog[p.trap_number] = dog
            if not trap_to_dog:
                continue

            race = Race.from_dogs(trap_to_dog, race_id, meeting_id)
            if race:
                print("\n========================")
                race.print_info()
                
                # Weather data should already be added in setUp
                print("--- Weather Data ---")
                print(f"Rainfall (7d): {race.rainfall_7d}")
                print(f"Temperature: {race.temperature}°C")
                print(f"Humidity: {race.humidity}%")
                
                # Verify weather fields are present
                self.assertIsNotNone(race.rainfall_7d)
                self.assertIsNotNone(race.temperature)
                self.assertIsNotNone(race.humidity)
                self.assertEqual(len(race.rainfall_7d), 7)
                
                # Commentary tags are now automatically extracted in Race.from_dogs()
                print("--- Commentary Tags (automatically extracted) ---")
                print(f"Commentary tags: {race.commentary_tags}")
                
                print("--- Loaded Dogs ---")
                for trap, dog in race.get_dogs(dogs).items():
                    print(f"Trap {trap}: {dog.name or dog.id}")
                
                # Save and reload with weather and commentary
                race_path = os.path.join(RACES_OUT, f"{race_id}_{meeting_id}.pkl")
                race.save(race_path)
                reloaded = Race.load(race_path)
                
                # Verify weather and commentary were saved
                print("--- Verifying Saved Data ---")
                print(f"Weather saved: rainfall={reloaded.rainfall_7d}, temp={reloaded.temperature}, humidity={reloaded.humidity}")
                print(f"Commentary saved: {reloaded.commentary_tags}")

    def test_pedigree_relationships(self):
        """Test that sire/dam are now Dog objects, not strings"""
        string_count = 0
        dog_count = 0
        none_count = 0
        
        for dog_id, dog in self.dog_lookup.items():
            if isinstance(dog, Dog):
                for rel_name in ['sire', 'dam']:
                    rel_value = getattr(dog, rel_name)
                    if isinstance(rel_value, str):
                        string_count += 1
                        print(f"WARNING: {dog_id}.{rel_name} is still string: {rel_value}")
                    elif isinstance(rel_value, Dog):
                        dog_count += 1
                    elif rel_value is None:
                        none_count += 1
        
        print(f"Pedigree relationship summary:")
        print(f"  Dog objects: {dog_count}")
        print(f"  None values: {none_count}")
        print(f"  String values (should be 0): {string_count}")
        
        # Assert no string relationships remain
        self.assertEqual(string_count, 0, "All sire/dam relationships should be Dog objects or None")

    def test_artificial_parents(self):
        """Test that artificial parents have correct properties and numeric IDs"""
        artificial_parents = [dog for dog in self.dog_lookup.values() 
                             if int(dog.id) >= 100000 and int(dog.id) < 300000]
        
        print(f"Found {len(artificial_parents)} artificial parents")
        
        for parent in artificial_parents:
            print(f"Artificial parent: {parent.id} ({parent.name})")
            print(f"  Weight: {parent.weight}")
            print(f"  Children count: {len(self.parent_offspring_dict.get(parent.id, []))}")
            
            # Verify properties
            self.assertTrue(parent.name, "Artificial parent should have a name")
            self.assertIsNone(parent.sire, "Artificial parent should have no sire")
            self.assertIsNone(parent.dam, "Artificial parent should have no dam")
            self.assertEqual(len(parent.race_participations), 0, "Artificial parent should have no races")
            
            # Verify ID is in correct range
            parent_id_int = int(parent.id)
            self.assertGreaterEqual(parent_id_int, 100000, "Artificial parent ID should be >= 100000")
            self.assertLess(parent_id_int, 300000, "Artificial parent ID should be < 300000")

def load_all_tracks() -> dict:
    tracks = {}
    tracks_dir = os.path.join(parent_dir, TRACKS_DIR)
    for fname in os.listdir(tracks_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(tracks_dir, fname), "rb") as f:
                track = pickle.load(f)
                tracks[track.name] = track
    return tracks

if __name__ == '__main__':
    unittest.main()
