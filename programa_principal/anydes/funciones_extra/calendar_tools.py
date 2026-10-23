from __future__ import annotations

import importlib.util
import json
import os
import webbrowser
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlencode


@lru_cache(maxsize=1)
def outlook_calendar_available() -> bool:
    try:
        if importlib.util.find_spec("win32com.client") is None:
            return False
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    if os.name != "nt":
        return False
    try:
        import winreg

        paths = (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\OUTLOOK.EXE",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\OUTLOOK.EXE",
        )
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for path in paths:
                try:
                    with winreg.OpenKey(root, path):
                        break
                except OSError:
                    continue
            else:
                continue
            break
        else:
            return False
    except ImportError:
        return False
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        return int(namespace.Stores.Count) > 0
    except Exception:
        return False


def microsoft365_web_available() -> bool:
    return os.name == "nt"


def create_calendar_event(argument: str) -> dict[str, object]:
    if outlook_calendar_available():
        result = create_outlook_event(argument)
        if result.get("created"):
            return result
        if result.get("error") not in {"calendar_creation_failed"}:
            return result
    if microsoft365_web_available():
        return prepare_microsoft365_event(argument)
    return {
        "created": False,
        "prepared": False,
        "error": "calendar_integration_unavailable",
        "backend": "unavailable",
    }


def create_outlook_event(argument: str) -> dict[str, object]:
    try:
        payload = json.loads(argument)
    except json.JSONDecodeError:
        return {"created": False, "error": "invalid_calendar_payload"}
    required = ("title", "start_iso", "end_iso", "timezone")
    if any(not str(payload.get(key) or "").strip() for key in required):
        return {"created": False, "error": "incomplete_calendar_payload"}
    if not outlook_calendar_available():
        return {
            "created": False,
            "error": "calendar_integration_unavailable",
            "backend": "outlook",
        }
    try:
        start = datetime.fromisoformat(str(payload["start_iso"]))
        end = datetime.fromisoformat(str(payload["end_iso"]))
    except ValueError:
        return {"created": False, "error": "invalid_calendar_datetime"}
    if end <= start:
        return {"created": False, "error": "calendar_end_before_start"}
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        appointment = outlook.CreateItem(1)
        appointment.Subject = str(payload["title"]).strip()
        appointment.Start = start.strftime("%Y-%m-%d %H:%M")
        appointment.End = end.strftime("%Y-%m-%d %H:%M")
        appointment.Location = str(payload.get("location") or "").strip()
        appointment.Body = str(payload.get("notes") or "").strip()
        appointment.Save()
        entry_id = str(getattr(appointment, "EntryID", "") or "")
    except Exception as exc:
        return {
            "created": False,
            "error": "calendar_creation_failed",
            "error_type": type(exc).__name__,
            "backend": "outlook",
        }
    return {
        "created": bool(entry_id),
        "backend": "outlook",
        "entry_id": entry_id,
        "title": str(payload["title"]).strip(),
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
        "timezone": str(payload["timezone"]),
        "location": str(payload.get("location") or "").strip(),
    }


def prepare_microsoft365_event(argument: str) -> dict[str, object]:
    try:
        payload = json.loads(argument)
    except json.JSONDecodeError:
        return {"created": False, "prepared": False, "error": "invalid_calendar_payload"}
    required = ("title", "start_iso", "end_iso", "timezone")
    if any(not str(payload.get(key) or "").strip() for key in required):
        return {
            "created": False,
            "prepared": False,
            "error": "incomplete_calendar_payload",
        }
    try:
        start = datetime.fromisoformat(str(payload["start_iso"]))
        end = datetime.fromisoformat(str(payload["end_iso"]))
    except ValueError:
        return {
            "created": False,
            "prepared": False,
            "error": "invalid_calendar_datetime",
        }
    if end <= start:
        return {
            "created": False,
            "prepared": False,
            "error": "calendar_end_before_start",
        }
    query = {
        "subject": str(payload["title"]).strip(),
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
    }
    notes = str(payload.get("notes") or "").strip()
    location = str(payload.get("location") or "").strip()
    if notes:
        query["body"] = notes
    if location:
        query["location"] = location
    compose_url = (
        "https://outlook.office.com/calendar/0/deeplink/compose?"
        + urlencode(query)
    )
    try:
        opened = bool(webbrowser.open(compose_url, new=2))
    except webbrowser.Error:
        opened = False
    if not opened:
        return {
            "created": False,
            "prepared": False,
            "error": "calendar_form_open_failed",
            "backend": "microsoft365_web",
        }
    return {
        "created": False,
        "prepared": True,
        "requires_user_save": True,
        "backend": "microsoft365_web",
        "title": str(payload["title"]).strip(),
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
        "timezone": str(payload["timezone"]),
        "location": location,
    }
