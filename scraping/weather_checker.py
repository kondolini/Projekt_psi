import argparse
from datetime import datetime, timedelta
import requests
from geopy.geocoders import Nominatim


def get_weather(date: str, time: str, place: str):
    """
    Fetches weather data (past 7 days rainfall, humidity and temperature) for a specific date, time, and location.

    Args:
        date (str): Date in YYYY-MM-DD format.
        time (str): Time in HH:MM format (must be on the hour).
        place (str): Location name.
    
    Returns:
        dict: {
            'rainfall_7d': [float],  # mm rainfall for each of past 7 days (including the date)
            'humidity': float,       # % humidity at that hour
            'temperature': float     # °C at that hour
        }
    """
    try:
        # Step 1: Geocode the location
        geolocator = Nominatim(user_agent="weather_checker")
        location = geolocator.geocode(place)
        if not location:
            print(f"Error: Could not find the location '{place}'.")
            return None

        latitude = location.latitude
        longitude = location.longitude

        # Step 2: Date range for rainfall history
        end_date = datetime.fromisoformat(date)
        start_date = end_date - timedelta(days=6)

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Step 3: Request both daily and hourly data
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

        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Step 4: Extract rainfall
        rainfall_7d = data.get("daily", {}).get("precipitation_sum", [])
        if len(rainfall_7d) != 7:
            print("Warning: Rainfall data missing or incomplete.")

        # Step 5: Extract hourly humidity and temperature for target time
        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])
        target_timestamp = f"{date}T{time}"

        try:
            index = timestamps.index(target_timestamp)
            temperature = hourly.get("temperature_2m", [])[index]
            humidity = hourly.get("relativehumidity_2m", [])[index]
        except (ValueError, IndexError):
            print(f"Error: No data available for {target_timestamp}. Check time format (e.g., '14:00').")
            return None

        # Step 6: Return results
        return {
            "rainfall_7d": rainfall_7d,
            "humidity": humidity,
            "temperature": temperature
        }

    except Exception as e:
        print(f"An error occurred: {e}")
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
