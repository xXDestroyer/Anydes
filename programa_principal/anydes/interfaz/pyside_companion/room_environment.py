from __future__ import annotations

import os
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

from anydes.nucleo.config import Config, config as app_config
from anydes.nucleo.consent import external_access_decision


ROOM_ASSET_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class RoomEnvironment:
    """Estado visual publico de la habitacion de Anydes."""

    time_of_day: str
    weather: str
    warm_light: float
    cool_light: float
    particle_density: float
    window_glow: float
    lamp_glow: float
    sunrise: str = ""
    sunset: str = ""
    season: str = ""


def room_background_path() -> Path:
    cinematic = ROOM_ASSET_DIR / "anydes_room_cinematic_warm_cool.png"
    return cinematic if cinematic.exists() else ROOM_ASSET_DIR / "anydes_room_night_rain.png"


def synced_environment(
    moment: datetime | None = None,
    *,
    config: Config | None = None,
    opener=None,
) -> RoomEnvironment:
    """Sincroniza la habitacion con PC, fecha y clima configurable.

    Si `ANYDES_WEATHER_PROVIDER=open_meteo` y existen coordenadas, consulta
    Open-Meteo. Si no, usa hora real del PC y `ANYDES_ROOM_WEATHER`.
    """

    current = moment or datetime.now().astimezone()
    selected_config = config or app_config
    weather = os.getenv("ANYDES_ROOM_WEATHER")
    sunrise = _parse_hhmm(os.getenv("ANYDES_ROOM_SUNRISE", ""))
    sunset = _parse_hhmm(os.getenv("ANYDES_ROOM_SUNSET", ""))

    if (
        os.getenv("ANYDES_WEATHER_PROVIDER", "").strip().lower() == "open_meteo"
        and external_access_decision(selected_config, "weather").allowed
    ):
        fetched = _fetch_open_meteo_environment(opener=opener)
        if fetched:
            weather = fetched.get("weather") or weather
            sunrise = _parse_iso_time(fetched.get("sunrise", "")) or sunrise
            sunset = _parse_iso_time(fetched.get("sunset", "")) or sunset

    return environment_for_datetime(current, weather=weather, sunrise=sunrise, sunset=sunset)


def environment_for_datetime(
    moment: datetime,
    *,
    weather: str | None = None,
    sunrise: time | None = None,
    sunset: time | None = None,
) -> RoomEnvironment:
    selected_weather = _normalize_weather(weather or os.getenv("ANYDES_ROOM_WEATHER", "rain"))
    season = _season_for_month(moment.month)
    tod = _time_of_day(moment, sunrise=sunrise, sunset=sunset)
    if tod == "dawn":
        return RoomEnvironment("dawn", selected_weather, 0.78, 0.34, 0.44, 0.60, 0.34, _fmt(sunrise), _fmt(sunset), season)
    if tod == "day":
        return RoomEnvironment("day", selected_weather, 0.38, 0.52, 0.26, 0.68, 0.12, _fmt(sunrise), _fmt(sunset), season)
    if tod == "sunset":
        return RoomEnvironment("sunset", selected_weather, 0.90, 0.24, 0.48, 0.48, 0.50, _fmt(sunrise), _fmt(sunset), season)
    return RoomEnvironment("night", selected_weather, 0.58, 0.78, 0.54, 0.42, 0.82, _fmt(sunrise), _fmt(sunset), season)


def _fetch_open_meteo_environment(*, opener=None) -> dict[str, str] | None:
    latitude = os.getenv("ANYDES_WEATHER_LATITUDE", "").strip()
    longitude = os.getenv("ANYDES_WEATHER_LONGITUDE", "").strip()
    if not latitude or not longitude:
        return None
    timezone = os.getenv("ANYDES_WEATHER_TIMEZONE", "auto").strip() or "auto"
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "weather_code,precipitation,rain,showers,snowfall,cloud_cover,is_day",
            "daily": "sunrise,sunset",
            "timezone": timezone,
            "forecast_days": "1",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(url, timeout=float(os.getenv("ANYDES_WEATHER_TIMEOUT", "2.5"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    weather = _weather_from_open_meteo(
        int(current.get("weather_code", 0) or 0),
        float(current.get("precipitation", 0.0) or 0.0),
        float(current.get("snowfall", 0.0) or 0.0),
        float(current.get("cloud_cover", 0.0) or 0.0),
    )
    sunrise = _first(daily.get("sunrise"))
    sunset = _first(daily.get("sunset"))
    return {"weather": weather, "sunrise": sunrise, "sunset": sunset}


def _weather_from_open_meteo(code: int, precipitation: float, snowfall: float, cloud_cover: float) -> str:
    if snowfall > 0 or code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if code in {95, 96, 99}:
        return "storm"
    if precipitation > 0 or code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "rain"
    if cloud_cover >= 62 or code in {2, 3, 45, 48}:
        return "cloudy"
    return "sunny"


def _normalize_weather(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"sun", "sunny", "soleado"}:
        return "sunny"
    if normalized in {"cloud", "cloudy", "nublado"}:
        return "cloudy"
    if normalized in {"storm", "tormenta"}:
        return "storm"
    if normalized in {"snow", "nieve"}:
        return "snow"
    if normalized in {"rain", "rainy", "lluvia"}:
        return "rain"
    return "rain"


def _time_of_day(moment: datetime, *, sunrise: time | None, sunset: time | None) -> str:
    current = moment.time()
    if sunrise and sunset:
        sunrise_minutes = _minutes(sunrise)
        sunset_minutes = _minutes(sunset)
        current_minutes = _minutes(current)
        if sunrise_minutes - 35 <= current_minutes < sunrise_minutes + 115:
            return "dawn"
        if sunset_minutes - 85 <= current_minutes < sunset_minutes + 55:
            return "sunset"
        if sunrise_minutes + 115 <= current_minutes < sunset_minutes - 85:
            return "day"
        return "night"

    hour = moment.hour
    if 5 <= hour < 9:
        return "dawn"
    if 9 <= hour < 17:
        return "day"
    if 17 <= hour < 20:
        return "sunset"
    return "night"


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _parse_hhmm(value: str) -> time | None:
    if not value.strip():
        return None
    try:
        hour, minute = value.strip().split(":", 1)
        return time(int(hour), int(minute[:2]))
    except Exception:
        return None


def _parse_iso_time(value: str) -> time | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).time()
    except Exception:
        return _parse_hhmm(value[-5:])


def _first(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _fmt(value: time | None) -> str:
    return value.strftime("%H:%M") if value else ""
