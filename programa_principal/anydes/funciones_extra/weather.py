from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from anydes.nucleo.config import Config
from anydes.nucleo.consent import external_access_decision


def handler_for_config(config: Config):
    def current_weather(_: str = "") -> dict[str, object]:
        access = external_access_decision(config, "weather")
        if not access.allowed:
            return {
                "available": False,
                "reason": access.reason,
                "provider": "open_meteo",
            }
        latitude = os.getenv("ANYDES_WEATHER_LATITUDE", "").strip()
        longitude = os.getenv("ANYDES_WEATHER_LONGITUDE", "").strip()
        if not latitude or not longitude:
            return {
                "available": False,
                "reason": "weather_location_not_configured",
                "provider": "open_meteo",
            }
        timezone_name = (
            os.getenv("ANYDES_WEATHER_TIMEZONE", "America/Guayaquil").strip()
            or "America/Guayaquil"
        )
        params = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,precipitation,"
                    "rain,showers,snowfall,weather_code,cloud_cover,"
                    "wind_speed_10m,is_day"
                ),
                "timezone": timezone_name,
                "forecast_days": "1",
            }
        )
        url = f"https://api.open-meteo.com/v1/forecast?{params}"
        timeout = float(os.getenv("ANYDES_WEATHER_TIMEOUT", "2.5"))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {
                "available": False,
                "reason": "weather_request_failed",
                "error_type": type(exc).__name__,
                "provider": "open_meteo",
            }
        current = dict(payload.get("current") or {})
        code = int(current.get("weather_code", 0) or 0)
        precipitation = float(current.get("precipitation", 0.0) or 0.0)
        snowfall = float(current.get("snowfall", 0.0) or 0.0)
        cloud_cover = float(current.get("cloud_cover", 0.0) or 0.0)
        return {
            "available": True,
            "provider": "open_meteo",
            "location_label": (
                os.getenv("ANYDES_WEATHER_LOCATION_LABEL", "").strip()
                or "tu ubicación configurada"
            ),
            "timezone": str(payload.get("timezone") or timezone_name),
            "observed_at": str(current.get("time") or datetime.now(timezone.utc).isoformat()),
            "weather": _weather_kind(code, precipitation, snowfall, cloud_cover),
            "weather_code": code,
            "temperature_c": _number(current.get("temperature_2m")),
            "apparent_temperature_c": _number(current.get("apparent_temperature")),
            "precipitation_mm": precipitation,
            "cloud_cover_percent": cloud_cover,
            "wind_speed_kmh": _number(current.get("wind_speed_10m")),
            "is_day": bool(current.get("is_day", 0)),
        }

    return current_weather


def _weather_kind(
    code: int,
    precipitation: float,
    snowfall: float,
    cloud_cover: float,
) -> str:
    if snowfall > 0 or code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if code in {95, 96, 99}:
        return "storm"
    if precipitation > 0 or code in {
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        80,
        81,
        82,
    }:
        return "rain"
    if cloud_cover >= 62 or code in {2, 3, 45, 48}:
        return "cloudy"
    return "sunny"


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None
