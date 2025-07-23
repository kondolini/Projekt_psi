"""
Example script to test the ML pipeline on a small dataset
"""

import os
import sys
import pickle
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog
from machine_learning.data_processor import RaceDataProcessor, create_dataset


def test_data_processing():
    """Test data processing pipeline on sample data"""
    
    print("🧪 Testing ML Data Processing Pipeline")
    print("="*50)
    
    # Try to load some sample data
    data_dir = os.path.join(parent_dir, "data")
    
    # Load dogs
    dogs = {}
    dogs_dir = os.path.join(data_dir, "dogs_enhanced")
    
    if os.path.exists(dogs_dir):
        print(f"📁 Loading dogs from: {dogs_dir}")
        file_count = 0
        
        for fname in os.listdir(dogs_dir):
            if fname.endswith('.pkl') and file_count < 3:  # Only first 3 files for testing
                try:
                    with open(os.path.join(dogs_dir, fname), 'rb') as f:
                        bucket = pickle.load(f)
                        for dog_id, dog_obj in bucket.items():
                            if isinstance(dog_obj, Dog):
                                dogs[dog_id] = dog_obj
                    file_count += 1
                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")
        
        print(f"✅ Loaded {len(dogs)} dogs from {file_count} files")
    else:
        print(f"❌ Dogs directory not found: {dogs_dir}")
        return False
    
    # Load pre-built race objects from buckets
    races = []
    races_dir = os.path.join(data_dir, "races")
    
    if os.path.exists(races_dir):
        print(f"📁 Loading pre-built races from: {races_dir}")
        file_count = 0
        
        for fname in os.listdir(races_dir):
            if fname.startswith('races_bucket_') and fname.endswith('.pkl') and file_count < 2:  # Only first 2 files
                try:
                    bucket_path = os.path.join(races_dir, fname)
                    with open(bucket_path, 'rb') as f:
                        races_bucket = pickle.load(f)
                    
                    for storage_key, race in races_bucket.items():
                        if isinstance(race, Race) and len(race.dog_ids) >= 3:  # At least 3 dogs
                            races.append(race)
                            if len(races) >= 50:  # Limit to 50 races for testing
                                break
                    file_count += 1
                    
                    if len(races) >= 50:
                        break
                        
                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")
        
        print(f"✅ Loaded {len(races)} race objects from {file_count} bucket files")
    else:
        print(f"❌ Races directory not found: {races_dir}")
        return False

    if not races:
        print("❌ No valid races found")
        return False
    
    # Test data processor
    print("🔧 Testing data processor...")
    processor = RaceDataProcessor()
    
    try:
        # Fit encoders on subset of data
        processor.fit_encoders(races[:20], dogs)
        print("✅ Encoders fitted successfully")
        
        # Process a few races
        dataset = create_dataset(races[:10], dogs, processor)
        print(f"✅ Created dataset with {len(dataset)} samples")
        
        if dataset:
            sample = dataset[0]
            print("\n📊 Sample data structure:")
            print(f"  Race features keys: {list(sample['race_features'].keys())}")
            print(f"  Number of dogs: {len(sample['dog_features'])}")
            print(f"  Target labels: {sample['targets']}")
            print(f"  Race datetime: {sample['race_datetime']}")
            
            # Check dog features structure
            if sample['dog_features']:
                dog_features = sample['dog_features'][0]
                print(f"  Dog features keys: {list(dog_features.keys())}")
                if 'history' in dog_features:
                    history = dog_features['history']
                    print(f"  History features keys: {list(history.keys())}")
                    print(f"  History length: {len(history['positions'])}")
        
        print("\n🎯 Encoder sizes:")
        print(f"  Tracks: {len(processor.track_encoder)}")
        print(f"  Classes: {len(processor.class_encoder)}")
        print(f"  Categories: {len(processor.category_encoder)}")
        print(f"  Trainers: {len(processor.trainer_encoder)}")
        print(f"  Commentary vocab: {processor.commentary_processor.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🏁 Greyhound Racing ML Pipeline Test")
    print("="*50)
    
    success = test_data_processing()
    
    if success:
        print("\n🎉 All tests passed! Pipeline is ready for training.")
        print("\n📝 Next steps:")
        print("  1. Run full training: python train.py --data_dir ../data")
        print("  2. Evaluate model: python evaluate.py --data_dir ../data")
        print("  3. Check outputs in: machine_learning/outputs/")
    else:
        print("\n❌ Tests failed. Please check your data setup.")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure data directory exists: ../data/")
        print("  2. Check dogs_enhanced/ and race_participations/ folders")
        print("  3. Verify pickle files are not corrupted")


if __name__ == '__main__':
    main()
