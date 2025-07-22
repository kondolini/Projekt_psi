import argparse
from datetime import datetime, timedelta
import requests
from geopy.geocoders import Nominatim
import time
import threading

# Enhanced rate limiting - reduced for speed
_last_request_time = 0
_request_lock = threading.Lock()
MIN_REQUEST_INTERVAL = 0.5  # Reduced to 0.5 seconds (2 req/sec)
_request_count = 0
_start_time = time.time()

def get_weather(date: str, time_str: str, place: str):
    """
    Fetches weather data with optimized rate limiting
    """
    global _last_request_time, _request_count, _start_time
    
    # Optimized rate limiting
    with _request_lock:
        now = time.time()
        time_since_last = now - _last_request_time
        
        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last
            time.sleep(wait_time)
        
        _last_request_time = time.time()
        _request_count += 1
        
        # Log progress every 20 requests
        if _request_count % 20 == 0:
            elapsed = time.time() - _start_time
            rate = _request_count / elapsed * 60  # requests per minute
            print(f"Weather API: {_request_count} requests, {rate:.1f} req/min")
    
    try:
        # Step 1: Geocode with longer timeout and retry
        geolocator = Nominatim(user_agent="weather_checker_v2", timeout=15)
        
        # Try geocoding with retry
        location = None
        for attempt in range(2):
            try:
                location = geolocator.geocode(place)
                if location:
                    break
                time.sleep(1)  # Brief pause between attempts
            except Exception as e:
                if attempt == 0:
                    print(f"Geocoding attempt 1 failed for {place}: {e}")
                    time.sleep(2)
                else:
                    print(f"Geocoding failed for {place}: {e}")
                    return None
        
        if not location:
            print(f"Could not geocode location: {place}")
            return None

        latitude = location.latitude
        longitude = location.longitude

        # Step 2: Date range for rainfall history
        end_date = datetime.fromisoformat(date)
        start_date = end_date - timedelta(days=6)

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Step 3: Request weather data with timeout
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "daily": "precipitation_sum",
            "hourly": "temperature_2m,relativehumidity_2m",
            "start_date": start_str,
            "end_date": end_str
        }

        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Step 4: Extract rainfall
        rainfall_7d = data.get("daily", {}).get("precipitation_sum", [])
        if len(rainfall_7d) != 7:
            print("Warning: Rainfall data missing or incomplete.")

        # Step 5: Extract hourly humidity and temperature for target time
        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])
        target_timestamp = f"{date}T{time_str}"

        try:
            index = timestamps.index(target_timestamp)
            temperature = hourly.get("temperature_2m", [])[index]
            humidity = hourly.get("relativehumidity_2m", [])[index]
        except (ValueError, IndexError):
            print(f"Error: No data available for {target_timestamp}. Using defaults.")
            temperature = 15.0
            humidity = 50.0

        # Step 6: Return results
        return {
            "rainfall_7d": rainfall_7d,
            "humidity": humidity,
            "temperature": temperature
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limited - backing off for {place}")
            time.sleep(10)  # Longer backoff for rate limits
        return None
    except Exception as e:
        print(f"Weather API error for {place} on {date}: {str(e)[:100]}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch weather data for a date, time, and place.")
    parser.add_argument("date", help="Date in YYYY-MM-DD format")
    parser.add_argument("time", help="Time in HH:MM format (e.g. 14:00)")
    parser.add_argument("place", help="Location name (e.g., 'Brisbane')")

    args = parser.parse_args()
    result = get_weather(args.date, args.time, args.place)

    if result:
        print("\nWeather Data:")
        print("-" * 40)
        print(f"Rainfall (last 7 days including {args.date}): {result['rainfall_7d']} mm")
        print(f"Humidity at {args.time}: {result['humidity']}%")
        print(f"Temperature at {args.time}: {result['temperature']}°C")
        print("-" * 40)
    parser.add_argument("place", help="Location name (e.g., 'Brisbane')")

    args = parser.parse_args()
    result = get_weather_cached(args.date, args.time, args.place)

    if result:
        print("\nWeather Data:")
        print("-" * 40)
        print(f"Rainfall (last 7 days including {args.date}): {result['rainfall_7d']} mm")
        print(f"Humidity at {args.time}: {result['humidity']}%")
        print(f"Temperature at {args.time}: {result['temperature']}°C")
        print("-" * 40)
        print("-" * 40)
