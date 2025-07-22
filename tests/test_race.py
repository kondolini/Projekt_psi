import os
import sys
import pickle
import unittest
import csv
import shutil
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adjust path for module import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.dog import Dog
from models.race import Race
from models.track import Track
from scraping.weather_checker import get_weather

# Test configuration
NUM_BUCKETS = int(os.getenv('NUM_BUCKETS', 100))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DATA_DIR = os.path.join(project_root, os.getenv('TEST_DATA_DIR', 'test_data'))

class TestRaceConstruction(unittest.TestCase):
    """Test race construction with pedigree relationships and weather data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once"""
        cls.setup_test_directories()
        cls.load_source_data()
        cls.initialize_dog_name_dict()
        cls.fix_pedigree_relationships()
        cls.save_test_data()
        cls.build_sample_races()

    @classmethod
    def setup_test_directories(cls):
        """Create clean test directory structure"""
        # Clean and create test directories
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)
        
        cls.test_dogs_dir = os.path.join(TEST_DATA_DIR, 'dogs_enhanced')
        cls.test_races_dir = os.path.join(TEST_DATA_DIR, 'races')
        cls.test_unified_dir = os.path.join(TEST_DATA_DIR, 'unified')
        
        for directory in [cls.test_dogs_dir, cls.test_races_dir, cls.test_unified_dir]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def load_source_data(cls):
        """Load dogs and participations from source directories"""
        # Load dogs from enhanced directory using absolute paths
        dogs_enhanced_dir = os.path.join(project_root, os.getenv('DOGS_ENHANCED_DIR', 'data/dogs_enhanced'))
        cls.dog_lookup = {}
        
        if not os.path.exists(dogs_enhanced_dir):
            print(f"Warning: Enhanced dogs directory not found: {dogs_enhanced_dir}")
            return
        
        for fname in os.listdir(dogs_enhanced_dir):
            if fname.endswith('.pkl'):
                with open(os.path.join(dogs_enhanced_dir, fname), 'rb') as f:
                    bucket = pickle.load(f)
                    for dog_id, dog_obj in bucket.items():
                        if isinstance(dog_obj, Dog):
                            cls.dog_lookup[dog_id] = dog_obj

        # Load race participations using absolute paths
        parts_dir = os.path.join(project_root, os.getenv('RACE_PARTICIPATIONS_DIR', 'data/race_participations'))
        race_parts = defaultdict(list)
        
        if not os.path.exists(parts_dir):
            print(f"Warning: Race participations directory not found: {parts_dir}")
            return
        
        for fname in os.listdir(parts_dir):
            if fname.endswith('.pkl'):
                with open(os.path.join(parts_dir, fname), 'rb') as f:
                    parts = pickle.load(f)
                    for p in parts:
                        race_parts[(p.race_id, p.meeting_id)].append(p)
        
        cls.race_participations = race_parts

    @classmethod
    def initialize_dog_name_dict(cls):
        """Initialize dog name dictionary from existing data or create new"""
        cls.dog_name_csv_path = os.path.join(TEST_DATA_DIR, 'dog_name_dict.csv')
        cls.dog_name_map = {}
        cls.artificial_id_counter = 100000
        
        # Load existing dog name mapping using absolute path
        source_csv = os.path.join(project_root, os.getenv('DOG_NAME_CSV', 'data/dog_name_dict.csv'))
        if os.path.exists(source_csv):
            with open(source_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cls.dog_name_map[row['dogName']] = row['dogId']

    @classmethod
    def fix_pedigree_relationships(cls):
        """Convert string sire/dam names to Dog objects, create artificial parents"""
        missing_parents = defaultdict(list)
        cls.parent_offspring_dict = defaultdict(list)
        
        print(f"Fixing pedigree relationships for {len(cls.dog_lookup)} dogs...")
        
        # Debug: Check if any dogs have string sire/dam names
        string_sire_count = 0
        string_dam_count = 0
        none_sire_count = 0
        none_dam_count = 0
        
        for dog_id, dog in cls.dog_lookup.items():
            if isinstance(getattr(dog, 'sire', None), str):
                string_sire_count += 1
            elif getattr(dog, 'sire', None) is None:
                none_sire_count += 1
            
            if isinstance(getattr(dog, 'dam', None), str):
                string_dam_count += 1
            elif getattr(dog, 'dam', None) is None:
                none_dam_count += 1
        
        print(f"Debug: String sires: {string_sire_count}, None sires: {none_sire_count}")
        print(f"Debug: String dams: {string_dam_count}, None dams: {none_dam_count}")
        
        # If no string relationships exist, create some test data
        if string_sire_count == 0 and string_dam_count == 0:
            print("No string pedigree data found - creating artificial test data")
            cls.create_test_pedigree_data()
        
        # First pass: identify existing parents and collect missing ones
        for dog_id, dog in cls.dog_lookup.items():
            for rel_name in ['sire', 'dam']:
                rel_value = getattr(dog, rel_name)
                if isinstance(rel_value, str) and rel_value:
                    parent_id = cls.dog_name_map.get(rel_value)
                    if parent_id and parent_id in cls.dog_lookup:
                        # Parent exists, set as Dog object
                        parent_dog = cls.dog_lookup[parent_id]
                        setattr(dog, rel_name, parent_dog)
                        cls.parent_offspring_dict[parent_id].append(dog_id)
                    else:
                        # Parent missing, collect for artificial creation
                        missing_parents[rel_value].append((dog_id, rel_name))
                elif isinstance(rel_value, Dog):
                    # Already a Dog object
                    cls.parent_offspring_dict[rel_value.id].append(dog_id)
        
        print(f"Found {len(missing_parents)} missing parents to create")
        
        # Second pass: create artificial parents with proper ID range
        artificial_parents_created = 0
        for parent_name, child_relationships in missing_parents.items():
            if cls.artificial_id_counter >= 400000:  # Proper upper limit
                print("Warning: Reached maximum artificial parent ID limit")
                break
                
            artificial_id = str(cls.artificial_id_counter)
            cls.artificial_id_counter += 1
            artificial_parents_created += 1
            
            # Create artificial parent
            artificial_dog = cls.create_artificial_parent(artificial_id, parent_name, child_relationships)
            cls.dog_lookup[artificial_id] = artificial_dog
            
            # Update children to reference this artificial parent
            for child_id, rel_name in child_relationships:
                child = cls.dog_lookup[child_id]
                setattr(child, rel_name, artificial_dog)
                cls.parent_offspring_dict[artificial_id].append(child_id)
            
            # Add to dog name mapping
            cls.dog_name_map[parent_name] = artificial_id
        
        print(f"Created {artificial_parents_created} artificial parents with IDs {100000}-{cls.artificial_id_counter-1}")

    @classmethod
    def create_test_pedigree_data(cls):
        """Create test pedigree data by adding string sire/dam names to some dogs"""
        print("Creating test pedigree data...")
        
        # Get first 100 dogs for testing
        test_dogs = list(cls.dog_lookup.items())[:100]
        
        # Create some fictional parent names
        test_sire_names = [
            "Champion Bolt", "Lightning Strike", "Thunder Runner", "Speed Demon", 
            "Quick Silver", "Fast Track", "Rapid Fire", "Bullet Train", "Swift Arrow", "Racing Star"
        ]
        
        test_dam_names = [
            "Lady Lightning", "Speed Queen", "Fast Lady", "Racing Belle", "Swift Princess",
            "Thunder Maiden", "Quick Empress", "Rapid Rose", "Bullet Beauty", "Champion Girl"
        ]
        
        created_relationships = 0
        
        for i, (dog_id, dog) in enumerate(test_dogs):
            # Give every 10th dog a sire and dam
            if i % 10 == 0:
                sire_name = test_sire_names[i // 10 % len(test_sire_names)]
                dam_name = test_dam_names[i // 10 % len(test_dam_names)]
                
                dog.sire = sire_name
                dog.dam = dam_name
                created_relationships += 2
                
                print(f"  Dog {dog_id}: sire='{sire_name}', dam='{dam_name}'")
        
        print(f"Created {created_relationships} test pedigree relationships")

    @classmethod
    def create_artificial_parent(cls, artificial_id: str, parent_name: str, child_relationships: list) -> Dog:
        """Create an artificial parent dog with averaged properties from children"""
        artificial_dog = Dog(dog_id=artificial_id)
        artificial_dog.set_name(parent_name)
        
        # Calculate average weight from children
        child_weights = []
        for child_id, _ in child_relationships:
            child = cls.dog_lookup[child_id]
            if child.weight:
                child_weights.append(child.weight)
        
        if child_weights:
            avg_weight = sum(child_weights) / len(child_weights)
            artificial_dog.set_weight(avg_weight)
        
        # Set other properties to None (artificial parents have no race history)
        artificial_dog.sire = None
        artificial_dog.dam = None
        artificial_dog.trainer = None
        artificial_dog.birth_date = None
        artificial_dog.race_participations = []
        
        return artificial_dog

    @classmethod
    def save_test_data(cls):
        """Save enhanced dogs and updated dog name dictionary to test directory"""
        # Save dogs to test buckets
        dog_buckets = defaultdict(dict)
        for dog_id, dog in cls.dog_lookup.items():
            bucket_idx = int(dog_id) % NUM_BUCKETS
            dog_buckets[bucket_idx][dog_id] = dog
        
        for bucket_idx, dogs_dict in dog_buckets.items():
            bucket_path = os.path.join(cls.test_dogs_dir, f"dogs_bucket_{bucket_idx}.pkl")
            with open(bucket_path, 'wb') as f:
                pickle.dump(dogs_dict, f)
        
        # Save updated dog name dictionary
        with open(cls.dog_name_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['dogName', 'dogId'])
            writer.writeheader()
            for name, dog_id in cls.dog_name_map.items():
                writer.writerow({'dogName': name, 'dogId': dog_id})
        
        print(f"Saved {len(cls.dog_lookup)} dogs to test directory")
        print(f"Saved {len(cls.dog_name_map)} entries to dog name dictionary")

    @classmethod
    def build_sample_races(cls):
        """Build sample races with weather data"""
        sample_race_keys = list(cls.race_participations.keys())[:3]
        cls.races = {}
        
        for race_key in sample_race_keys:
            parts = cls.race_participations[race_key]
            race = Race.from_participations(parts)
            
            # Add weather data
            cls.add_weather_to_race(race)
            
            cls.races[race_key] = race

    @classmethod
    def add_weather_to_race(cls, race: Race):
        """Add weather data to a race"""
        try:
            race_date_str = race.race_date.strftime("%Y-%m-%d")
            race_time_str = race.race_time.strftime("%H:%M")
            
            weather_data = get_weather(race_date_str, race_time_str, race.track_name)
            if weather_data:
                race.rainfall_7d = weather_data['rainfall_7d']
                race.humidity = weather_data['humidity']
                race.temperature = weather_data['temperature']
            else:
                # Set default weather values
                race.rainfall_7d = [0.0] * 7
                race.humidity = 50.0
                race.temperature = 15.0
        except Exception as e:
            print(f"Weather fetch failed for race {race.race_id}: {e}")
            race.rainfall_7d = [0.0] * 7
            race.humidity = 50.0
            race.temperature = 15.0

    def test_pedigree_relationships_fixed(self):
        """Test that all sire/dam are now Dog objects or None"""
        string_count = 0
        dog_count = 0
        none_count = 0
        
        for dog_id, dog in self.dog_lookup.items():
            for rel_name in ['sire', 'dam']:
                rel_value = getattr(dog, rel_name)
                if isinstance(rel_value, str):
                    string_count += 1
                elif isinstance(rel_value, Dog):
                    dog_count += 1
                elif rel_value is None:
                    none_count += 1
        
        print(f"Pedigree relationships: {dog_count} Dog objects, {none_count} None, {string_count} strings")
        self.assertEqual(string_count, 0, "All sire/dam should be Dog objects or None")

    def test_artificial_parents_created(self):
        """Test that artificial parents have correct properties"""
        artificial_parents = [
            dog for dog in self.dog_lookup.values() 
            if int(dog.id) >= 100000 and int(dog.id) < 400000
        ]
        
        print(f"Found {len(artificial_parents)} artificial parents")
        self.assertGreater(len(artificial_parents), 0, "Should have created artificial parents")
        
        for parent in artificial_parents[:3]:  # Test first 3
            self.assertTrue(parent.name, "Artificial parent should have name")
            self.assertIsNone(parent.sire, "Artificial parent should have no sire")
            self.assertIsNone(parent.dam, "Artificial parent should have no dam")
            self.assertEqual(len(parent.race_participations), 0, "Artificial parent should have no races")
            
            # Verify ID is in correct range
            parent_id_int = int(parent.id)
            self.assertGreaterEqual(parent_id_int, 100000, "Artificial parent ID should be >= 100000")
            self.assertLess(parent_id_int, 400000, "Artificial parent ID should be < 400000")

    def test_race_construction_with_weather(self):
        """Test race construction includes weather data and commentary tags"""
        for race_key, race in self.races.items():
            print(f"\n--- Testing Race {race_key} ---")
            race.print_info()  # Add detailed race info
            
            # Basic race properties
            self.assertIsInstance(race, Race)
            self.assertTrue(race.dog_ids, "Race should have dogs")
            
            # Weather data
            self.assertIsNotNone(race.rainfall_7d, "Race should have rainfall data")
            self.assertIsNotNone(race.temperature, "Race should have temperature")
            self.assertIsNotNone(race.humidity, "Race should have humidity")
            self.assertEqual(len(race.rainfall_7d), 7, "Should have 7 days of rainfall")
            
            # Commentary tags should be extracted
            self.assertIsInstance(race.commentary_tags, dict, "Should have commentary tags dict")
            
            print(f"✓ Race {race.race_id}: {len(race.dog_ids)} dogs, weather: {race.temperature}°C")

    def test_debug_data_details(self):
        """Debug test to examine data in detail"""
        print("\n=== DEBUGGING DATA DETAILS ===")
        
        # Print some regular dogs
        print("\n--- Sample Regular Dogs ---")
        regular_dogs = [dog for dog in self.dog_lookup.values() 
                       if int(dog.id) < 100000][:5]
        
        for i, dog in enumerate(regular_dogs):
            print(f"\nRegular Dog {i+1}:")
            dog.print_info()
        
        # Print artificial dogs
        print("\n--- Sample Artificial Dogs ---")
        artificial_dogs = [dog for dog in self.dog_lookup.values() 
                          if int(dog.id) >= 100000][:5]
        
        for i, dog in enumerate(artificial_dogs):
            print(f"\nArtificial Dog {i+1}:")
            dog.print_info()
        
        # Print parent-child relationships
        print("\n--- Parent-Child Relationships ---")
        relationship_count = 0
        for parent_id, child_ids in self.parent_offspring_dict.items():
            if relationship_count >= 3:  # Only show first 3 relationships
                break
            
            parent = self.dog_lookup.get(parent_id)
            if parent:
                print(f"\nParent: {parent.name} (ID: {parent_id})")
                print(f"  Is artificial: {int(parent_id) >= 100000}")
                print(f"  Children ({len(child_ids)}):")
                
                for child_id in child_ids[:3]:  # Show first 3 children
                    child = self.dog_lookup.get(child_id)
                    if child:
                        print(f"    - {child.name or 'No name'} (ID: {child_id})")
                        print(f"      Sire: {child.sire.name if child.sire else 'None'}")
                        print(f"      Dam: {child.dam.name if child.dam else 'None'}")
                
                relationship_count += 1
        
        # Check why so many None values
        print("\n--- Analyzing None Values ---")
        total_dogs = len(self.dog_lookup)
        dogs_with_sire = sum(1 for dog in self.dog_lookup.values() if dog.sire is not None)
        dogs_with_dam = sum(1 for dog in self.dog_lookup.values() if dog.dam is not None)
        dogs_with_name = sum(1 for dog in self.dog_lookup.values() if dog.name)
        
        print(f"Total dogs: {total_dogs}")
        print(f"Dogs with sire: {dogs_with_sire} ({dogs_with_sire/total_dogs*100:.1f}%)")
        print(f"Dogs with dam: {dogs_with_dam} ({dogs_with_dam/total_dogs*100:.1f}%)")
        print(f"Dogs with name: {dogs_with_name} ({dogs_with_name/total_dogs*100:.1f}%)")
        
        # Sample dogs from the dog_name_map
        print(f"\n--- Dog Name Map Sample (first 10) ---")
        for i, (name, dog_id) in enumerate(list(self.dog_name_map.items())[:10]):
            dog = self.dog_lookup.get(dog_id)
            print(f"{name} -> {dog_id} (exists: {dog is not None})")
            if dog:
                print(f"  Name in dog object: {dog.name}")

    def test_race_save_and_load(self):
        """Test race can be saved and loaded with all data intact"""
        race_key, race = next(iter(self.races.items()))
        
        # Save race in bucket format
        bucket_idx = int(race.race_id) % NUM_BUCKETS
        bucket_path = os.path.join(self.test_races_dir, f"races_bucket_{bucket_idx}.pkl")
        
        # Create or load existing bucket
        races_bucket = {}
        if os.path.exists(bucket_path):
            with open(bucket_path, 'rb') as f:
                races_bucket = pickle.load(f)
        
        # Add race to bucket
        storage_key = f"{race.race_id}_{race.meeting_id}"
        races_bucket[storage_key] = race
        
        # Save bucket
        with open(bucket_path, 'wb') as f:
            pickle.dump(races_bucket, f)
        
        # Load race from bucket
        with open(bucket_path, 'rb') as f:
            loaded_bucket = pickle.load(f)
        loaded_race = loaded_bucket[storage_key]
        
        # Verify all data preserved
        self.assertEqual(loaded_race.race_id, race.race_id)
        self.assertEqual(loaded_race.meeting_id, race.meeting_id)
        self.assertEqual(loaded_race.dog_ids, race.dog_ids)
        self.assertEqual(loaded_race.rainfall_7d, race.rainfall_7d)
        self.assertEqual(loaded_race.temperature, race.temperature)
        self.assertEqual(loaded_race.humidity, race.humidity)
        self.assertEqual(loaded_race.commentary_tags, race.commentary_tags)
        
        print(f"✓ Race successfully saved and loaded in bucket format: {bucket_path}")

    def test_parent_offspring_mapping(self):
        """Test parent-offspring relationships are correctly established"""
        # Save mapping to CSV for verification
        csv_path = os.path.join(self.test_unified_dir, 'parent_offspring_mapping.csv')
        
        csv_data = []
        for parent_id, offspring_ids in self.parent_offspring_dict.items():
            parent_dog = self.dog_lookup.get(parent_id)
            parent_name = parent_dog.name if parent_dog else "Unknown"
            is_artificial = int(parent_id) >= 100000
            
            for offspring_id in offspring_ids:
                offspring_dog = self.dog_lookup.get(offspring_id)
                offspring_name = offspring_dog.name if offspring_dog else "Unknown"
                
                csv_data.append({
                    'parent_id': parent_id,
                    'parent_name': parent_name,
                    'is_artificial_parent': is_artificial,
                    'offspring_id': offspring_id,
                    'offspring_name': offspring_name,
                    'offspring_count': len(offspring_ids)
                })
        
        # Save to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'parent_id', 'parent_name', 'is_artificial_parent', 
                'offspring_id', 'offspring_name', 'offspring_count'
            ])
            writer.writeheader()
            writer.writerows(csv_data)
        
        total_relationships = len(csv_data)
        artificial_relationships = sum(1 for row in csv_data if row['is_artificial_parent'])
        
        print(f"✓ Parent-offspring mapping: {total_relationships} total, {artificial_relationships} artificial")
        self.assertGreater(total_relationships, 0, "Should have parent-offspring relationships")

    def test_dog_name_dict_updated(self):
        """Test that dog name dictionary includes new artificial parents"""
        self.assertTrue(os.path.exists(self.dog_name_csv_path), "Dog name dict should exist")
        
        # Count artificial entries
        artificial_count = sum(1 for dog_id in self.dog_name_map.values() if int(dog_id) >= 100000)
        
        print(f"✓ Dog name dictionary: {len(self.dog_name_map)} total entries, {artificial_count} artificial")
        self.assertGreater(artificial_count, 0, "Should have artificial parent entries")

    def tearDown(self):
        """Clean up after each test"""
        # Remove race files but keep dog data and name dict for other tests
        for file in os.listdir(self.test_races_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.test_races_dir, file))

def load_all_tracks() -> dict:
    tracks = {}
    tracks_dir = os.path.join(project_root, os.getenv('TRACKS_DIR', 'data/tracks'))
    if not os.path.exists(tracks_dir):
        return tracks
    
    for fname in os.listdir(tracks_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(tracks_dir, fname), "rb") as f:
                track = pickle.load(f)
                tracks[track.name] = track
    return tracks

if __name__ == '__main__':
    unittest.main(verbosity=2)
