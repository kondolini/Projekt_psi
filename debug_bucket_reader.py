import os
import sys
import pickle
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from models.dog import Dog
from models.race_participation import RaceParticipation

def read_bucket_file(bucket_path):
    """Read and analyze a bucket file"""
    print(f"📂 Reading bucket file: {bucket_path}")
    print("=" * 60)
    
    if not os.path.exists(bucket_path):
        print(f"❌ File does not exist: {bucket_path}")
        return
    
    try:
        with open(bucket_path, "rb") as f:
            dogs_dict = pickle.load(f)
        
        print(f"✅ Successfully loaded bucket file")
        print(f"📊 Total dogs in bucket: {len(dogs_dict)}")
        print(f"📁 File size: {os.path.getsize(bucket_path) / 1024:.1f} KB")
        print()
        
        # Show data types
        print("🔍 Data structure analysis:")
        print(f"  - Dictionary keys type: {type(list(dogs_dict.keys())[0]) if dogs_dict else 'None'}")
        print(f"  - Dictionary values type: {type(list(dogs_dict.values())[0]) if dogs_dict else 'None'}")
        print()
        
        # Show sample keys
        print("🗂️ Sample dog IDs (first 10):")
        sample_keys = list(dogs_dict.keys())[:10]
        for i, key in enumerate(sample_keys):
            print(f"  {i+1}. {key} (type: {type(key)})")
        print()
        
        # Analyze first dog in detail
        if dogs_dict:
            first_dog_id, first_dog = next(iter(dogs_dict.items()))
            print(f"🐕 Detailed analysis of first dog: {first_dog_id}")
            print(f"  - Dog ID: {first_dog.id}")
            print(f"  - Dog Name: '{first_dog.name}' (type: {type(first_dog.name)})")
            print(f"  - Trainer: '{first_dog.trainer}' (type: {type(first_dog.trainer)})")
            print(f"  - Birth Date: {first_dog.birth_date}")
            print(f"  - Color: '{first_dog.color}'")
            print(f"  - Weight: {first_dog.weight}")
            print(f"  - Race Participations: {len(first_dog.race_participations)}")
            print()
            
            # Analyze race participations
            if first_dog.race_participations:
                print("🏁 Race participation analysis:")
                participation = first_dog.race_participations[0]
                print(f"  - Participation type: {type(participation)}")
                print(f"  - Race ID: {participation.race_id}")
                print(f"  - Dog ID: {participation.dog_id}")
                print(f"  - Race DateTime: {participation.race_datetime}")
                print(f"  - Track Name: {participation.track_name}")
                print(f"  - Position: {participation.position}")
                print(f"  - Run Time: {participation.run_time}")
                print(f"  - Trap Number: {participation.trap_number}")
                print(f"  - Meeting ID: {getattr(participation, 'meeting_id', 'Not available')}")
                print(f"  - Race Class: {participation.race_class}")
                print(f"  - Distance: {participation.distance}")
                print(f"  - Going: {participation.going}")
                print(f"  - SP: {participation.sp}")
                print(f"  - Comment: '{participation.comment}'")
                print()
                
                # Show all attributes of the participation object
                print("📋 All participation attributes:")
                for attr in dir(participation):
                    if not attr.startswith('_') and not callable(getattr(participation, attr)):
                        value = getattr(participation, attr)
                        print(f"  - {attr}: {value} (type: {type(value)})")
                print()
        
        # Statistics about dog names and trainers
        dogs_with_names = sum(1 for dog in dogs_dict.values() if dog.name and dog.name.strip())
        dogs_with_trainers = sum(1 for dog in dogs_dict.values() if dog.trainer and dog.trainer.strip())
        
        print("📈 Content statistics:")
        print(f"  - Dogs with names: {dogs_with_names}/{len(dogs_dict)} ({dogs_with_names/len(dogs_dict)*100:.1f}%)")
        print(f"  - Dogs with trainers: {dogs_with_trainers}/{len(dogs_dict)} ({dogs_with_trainers/len(dogs_dict)*100:.1f}%)")
        print()
        
        # Show dogs that have names (enhanced dogs)
        dogs_with_good_names = [
            (dog_id, dog.name, dog.trainer) 
            for dog_id, dog in dogs_dict.items() 
            if dog.name and dog.name.strip() and len(dog.name.strip()) > 2
        ]
        
        if dogs_with_good_names:
            print(f"🏆 Dogs with names (first 5 of {len(dogs_with_good_names)}):")
            for i, (dog_id, name, trainer) in enumerate(dogs_with_good_names[:5]):
                print(f"  {i+1}. Dog {dog_id}: '{name}' (Trainer: '{trainer or 'Unknown'}')")
        else:
            print("❌ No dogs found with meaningful names")
        print()
        
        # Check date range of races
        all_dates = []
        for dog in dogs_dict.values():
            for participation in dog.race_participations:
                if participation.race_datetime:
                    all_dates.append(participation.race_datetime)
        
        if all_dates:
            earliest = min(all_dates)
            latest = max(all_dates)
            print(f"📅 Race date range:")
            print(f"  - Earliest race: {earliest}")
            print(f"  - Latest race: {latest}")
            print(f"  - Total races: {len(all_dates)}")
        
    except Exception as e:
        print(f"❌ Error reading bucket file: {e}")
        import traceback
        traceback.print_exc()

def find_bucket_files():
    """Find all available bucket files"""
    print("🔍 Searching for bucket files...")
    print("=" * 40)
    
    possible_dirs = [
        "data/dogs_enhanced",
        #"data/dogs", 
        "../data/dogs_enhanced",
        #"../data/dogs"
    ]
    
    found_files = []
    
    for directory in possible_dirs:
        if os.path.exists(directory):
            try:
                files = os.listdir(directory)
                bucket_files = [f for f in files if f.startswith('dogs_bucket_') and f.endswith('.pkl')]
                if bucket_files:
                    print(f"📁 {directory}: {len(bucket_files)} bucket files")
                    for file in sorted(bucket_files)[:5]:  # Show first 5
                        full_path = os.path.join(directory, file)
                        size_kb = os.path.getsize(full_path) / 1024
                        found_files.append(full_path)
                        print(f"  - {file} ({size_kb:.1f} KB)")
                    if len(bucket_files) > 5:
                        print(f"  ... and {len(bucket_files) - 5} more files")
                else:
                    print(f"📁 {directory}: No bucket files found")
            except Exception as e:
                print(f"📁 {directory}: Error reading - {e}")
        else:
            print(f"📁 {directory}: Directory not found")
    
    print()
    return found_files

if __name__ == "__main__":
    print("🐕 Dog Bucket File Reader")
    print("Analyzes the structure and content of dog bucket files")
    print()
    
    # Find available bucket files
    bucket_files = find_bucket_files()
    
    if not bucket_files:
        print("❌ No bucket files found!")
        sys.exit(1)
    
    # Read the first available bucket file
    bucket_to_read = bucket_files[0]
    print(f"📖 Reading first available bucket file: {bucket_to_read}")
    print()
    
    read_bucket_file(bucket_to_read)
    
    # Ask if user wants to read a specific bucket
    print("\n" + "=" * 60)
    choice = input("Enter a specific bucket file path to read (or press Enter to exit): ").strip()
    
    if choice and os.path.exists(choice):
        print()
        read_bucket_file(choice)
    elif choice:
        print(f"❌ File not found: {choice}")
