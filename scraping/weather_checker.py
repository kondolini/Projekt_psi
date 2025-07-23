


# --- Meteostat Implementation ---
from datetime import datetime, timedelta
from meteostat import Point, Hourly, Daily
from geopy.geocoders import Nominatim
import time

_geocode_cache = {}

# --- Open-Meteo ERA5 Archive API Implementation ---
import os
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import time

def geocode_place(place):
    geolocator = Nominatim(user_agent="openmeteo_weather_checker", timeout=10)
    for attempt in range(2):
        try:
            location = geolocator.geocode(place)
            if location:
                return location.latitude, location.longitude
            time.sleep(1)
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
    return None, None

def get_weather(date: str, time_str: str, place: str):
    """
    Fetch historical weather for a given date, time, and place using Open-Meteo ERA5 API.
    Returns dict with rainfall_7d, temperature, humidity or None if not found.
    """
    try:
        # Geocode place to lat/lon
        lat, lon = geocode_place(place)
        if lat is None or lon is None:
            print(f"Could not geocode place: {place}")
            return None

        # Get 7 days ending at date
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=6)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        url = f"https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m,relativehumidity_2m,precipitation",
            "timezone": "Europe/London"
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Rainfall 7d: sum precipitation for each day
        rainfall_7d = []
        times = data.get("hourly", {}).get("time", [])
        precip = data.get("hourly", {}).get("precipitation", [])
        temp = data.get("hourly", {}).get("temperature_2m", [])
        humidity = data.get("hourly", {}).get("relativehumidity_2m", [])

        # Group precipitation by day
        day_precip = {}
        for t, p in zip(times, precip):
            d = t.split('T')[0]
            day_precip.setdefault(d, 0.0)
            if p is not None:
                day_precip[d] += p
        for i in range(7):
            d = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            rainfall_7d.append(round(day_precip.get(d, 0.0), 2))

        # Round input time to the nearest hour
        input_dt = datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M")
        # If minutes >= 30, round up, else round down
        if input_dt.minute >= 30:
            input_dt = input_dt.replace(minute=0) + timedelta(hours=1)
        else:
            input_dt = input_dt.replace(minute=0)

        # Find the hour closest to the requested (rounded) time on the requested date
        target_dt = input_dt
        best_idx = None
        min_diff = None
        for idx, t in enumerate(times):
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
            if dt.date() == target_dt.date():
                diff = abs((dt - target_dt).total_seconds())
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    best_idx = idx
        if best_idx is not None:
            temperature = temp[best_idx]
            hum = humidity[best_idx]
        else:
            temperature = None
            hum = None
        return {
            "rainfall_7d": rainfall_7d,
            "temperature": temperature,
            "humidity": hum
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None
        choice = input("Select an option: ").strip()
        if choice == "1":
            date = input("Enter date (YYYY-MM-DD): ").strip()
            time_str = input("Enter time (HH:MM, 24h): ").strip()
            place = input("Enter place (track/location name): ").strip()
            result = get_weather(date, time_str, place)
            if result:
                print("\nWeather Data:")
                print("-" * 40)
                print(f"Rainfall (last 7 days including {date}): {result['rainfall_7d']} mm")
                print(f"Temperature at {time_str}: {result['temperature']}°C")
                print(f"Humidity at {time_str}: {result['humidity']}%")
                print("-" * 40)
            else:
                print("No weather data found for the given input.")
        elif choice == "0":
            print("Exiting.")
        else:
            print("Invalid option.")
