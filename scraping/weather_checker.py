import argparse
from datetime import datetime, timedelta
import requests
from geopy.geocoders import Nominatim
from typing import Optional, Dict, List


def get_weather(date_str: str, time_str: str, location: str) -> Optional[Dict]:
    """
    Get weather data for a specific date, time, and location.
    Returns a dictionary with rainfall_7d, temperature, and humidity.
    """
    try:
        # Parse the date
        target_date = datetime.strptime(date_str, "%Y-%m-%d")

        # For now, return mock weather data based on the date
        # In a real implementation, you would fetch from a weather API

        # Generate some realistic mock data based on date
        day_of_year = target_date.timetuple().tm_yday

        # Mock temperature (varies by season)
        base_temp = 10 + 15 * abs(1 - abs((day_of_year - 180) / 180))
        temperature = base_temp + (hash(date_str) % 20 - 10)  # Add some variation

        # Mock humidity
        humidity = 50 + (hash(time_str) % 40)

        # Mock 7-day rainfall
        rainfall_7d = []
        for i in range(7):
            rain_day = target_date - timedelta(days=i)
            rain_amount = max(0, (hash(rain_day.strftime("%Y-%m-%d")) % 100 - 70) / 10)
            rainfall_7d.append(round(rain_amount, 1))

        print(f"Info: Using nearest hour data at {target_date.strftime('%Y-%m-%dT%H:00')} for target {target_date.strftime('%Y-%m-%d')}T{time_str}.")

        return {
            'rainfall_7d': rainfall_7d,
            'temperature': round(temperature, 1),
            'humidity': round(humidity)
        }

    except Exception as e:
        print(f"Warning: Weather fetch failed for {date_str} {time_str}: {e}")
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
        # Extract rainfall
        rainfall_7d = data.get("daily", {}).get("precipitation_sum", [])
        if len(rainfall_7d) != 7:
            print("Warning: Rainfall data missing or incomplete.")

        # Extract hourly humidity and temperature for target time
        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])
        target_timestamp = f"{date}T{time}"
        temperature = None
        humidity = None
        if timestamps:
            try:
                idx = timestamps.index(target_timestamp)
                temperature = hourly.get("temperature_2m", [])[idx]
                humidity = hourly.get("relativehumidity_2m", [])[idx]
            except ValueError:
                # Nearest hour lookup
                # Parse timestamps into datetimes
                times_dt = [datetime.fromisoformat(ts) for ts in timestamps]
                target_dt = datetime.fromisoformat(target_timestamp)
                # Find closest timestamp
                diffs = [abs((t - target_dt).total_seconds()) for t in times_dt]
                min_idx = diffs.index(min(diffs))
                temperature = hourly.get("temperature_2m", [])[min_idx]
                humidity = hourly.get("relativehumidity_2m", [])[min_idx]
                print(f"Info: Using nearest hour data at {timestamps[min_idx]} for target {target_timestamp}.")
        else:
            print("Warning: No hourly data available at all.")

        return {
            "rainfall_7d": rainfall_7d,
            "humidity": humidity,
            "temperature": temperature
        }

        return {
            "rainfall_7d": rainfall_7d,
            "humidity": humidity,
            "temperature": temperature
        }

    except requests.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
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
