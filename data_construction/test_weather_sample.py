import os
import sys
import pickle
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.race import Race

def test_weather_on_sample():
    """Test weather on a small sample of races"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    race_output_dir = os.path.join(project_root, "data/races")
    
    # Load first race bucket
    first_bucket = os.path.join(race_output_dir, "races_bucket_0.pkl")
    
    if not os.path.exists(first_bucket):
        print("No race buckets found!")
        return
    
    with open(first_bucket, 'rb') as f:
        races_bucket = pickle.load(f)
    
    print(f"Found {len(races_bucket)} races in first bucket")
    
    # Test weather on first 3 races
    for i, (storage_key, race) in enumerate(list(races_bucket.items())[:3]):
        print(f"\n--- Race {i+1}: {race.race_id} ---")
        print(f"Date: {race.race_date}, Track: {race.track_name}")
        print(f"Current weather: temp={race.temperature}, humidity={race.humidity}")
        
        # Try to get weather
        try:
            from scraping.weather_checker import get_weather
            date_str = race.race_date.strftime('%Y-%m-%d')
            weather_data = get_weather(date_str, "12:00", race.track_name)
            
            if weather_data:
                print(f"New weather: temp={weather_data['temperature']}, humidity={weather_data['humidity']}")
                print(f"Rainfall: {weather_data['rainfall_7d']}")
            else:
                print("No weather data available")
                
        except Exception as e:
            print(f"Weather error: {e}")

if __name__ == '__main__':
    test_weather_on_sample()
