import os
import pickle
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.insert(0, parent_dir)

from models.dog import Dog

def build_name_to_id_dict_and_csv(
    buckets_dir="../data/dogs_enhanced",
    output_pickle="name_to_id_dict.pkl",
    output_csv="name_to_id.csv"
):
    """
    Build a dictionary mapping dog names to dog IDs from all enhanced buckets,
    including artificial parents with IDs 100000-300000.
    """
    name_to_id = {}
    total_dogs = 0
    dogs_with_names = 0
    artificial_parents = 0

    for bucket_idx in range(100):
        bucket_path = os.path.join(buckets_dir, f"dogs_bucket_{bucket_idx}.pkl")
        if not os.path.exists(bucket_path):
            continue
        with open(bucket_path, "rb") as f:
            dogs_dict = pickle.load(f)
        for dog_id, dog in dogs_dict.items():
            total_dogs += 1
            if dog.name and dog.name.strip():
                name_to_id[dog.name.strip()] = dog_id
                dogs_with_names += 1
                
                # Count artificial parents
                if int(dog_id) >= 100000 and int(dog_id) < 300000:
                    artificial_parents += 1

    print(f"✅ Built name-to-id dictionary for {dogs_with_names} dogs with names (out of {total_dogs} total)")
    print(f"   - Real dogs: {dogs_with_names - artificial_parents}")
    print(f"   - Artificial parents: {artificial_parents}")

    # Save as pickle
    with open(output_pickle, "wb") as f:
        pickle.dump(name_to_id, f)
    print(f"💾 Saved dictionary to {output_pickle}")

    # Save as CSV
    df = pd.DataFrame(list(name_to_id.items()), columns=["dogName", "dogId"])
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved CSV to {output_csv}")

    return name_to_id

# Example usage:
if __name__ == "__main__":
    build_name_to_id_dict_and_csv()