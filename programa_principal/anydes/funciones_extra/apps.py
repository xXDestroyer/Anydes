from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from urllib.parse import quote_plus

from anydes.funciones_extra.application_resolver import ApplicationResolver


_APPLICATION_RESOLVER = ApplicationResolver()

# Compatibility vocabulary for speech transcript normalization only.
# Application discovery and execution are handled by ApplicationResolver.
APP_ALIASES = {
    "chrome": (),
    "google chrome": (),
    "edge": (),
    "microsoft edge": (),
    "spotify": (),
    "notepad": (),
    "bloc de notas": (),
    "calculadora": (),
    "calculator": (),
    "explorador": (),
    "explorador de archivos": (),
    "vscode": (),
    "visual studio code": (),
    "codigo": (),
    "whatsapp": (),
    "word": (),
    "microsoft word": (),
    "excel": (),
    "powerpoint": (),
    "outlook": (),
}


def app_availability(name: str) -> bool | None:
    """Return installed, unavailable, or unknown without launching anything."""
    resolution = _APPLICATION_RESOLVER.resolve(name)
    if resolution.status == "resolved":
        if resolution.candidate is not None and resolution.candidate.method == "shell":
            return None
        return True
    if resolution.status == "ambiguous":
        return None
    return False


def resolve_application(name: str):
    return _APPLICATION_RESOLVER.resolve(name)


def open_url(url: str) -> dict[str, object]:
    clean = url.strip()
    if not clean:
        return {"opened": False, "error": "missing_url"}
    if not clean.startswith(("http://", "https://")):
        clean = "https://" + clean
    requested = _open_with_browser(clean)
    return {
        "opened": requested,
        "launch_requested": requested,
        "url_loaded_verified": False,
        "url": clean,
        "target": "browser",
        "browser": _running_browser_name(),
    }


def open_browser_tab(url: str = "") -> dict[str, object]:
    target = url.strip() or "about:blank"
    requested = _open_with_browser(target)
    return {
        "opened": requested,
        "launch_requested": requested,
        "new_tab_requested": requested,
        "url": target,
        "url_loaded_verified": False,
        "target": "browser",
        "browser": _running_browser_name(),
    }


def chrome_search(query: str) -> dict[str, object]:
    clean = query.strip()
    if not clean:
        return {"opened": False, "error": "missing_query", "engine": "google"}
    url = f"https://www.google.com/search?q={quote_plus(clean)}"
    requested = _open_with_browser(url)
    return {
        "opened": requested,
        "launch_requested": requested,
        "query": clean,
        "url": url,
        "engine": "google",
    }


def open_youtube(_: str = "") -> dict[str, object]:
    url = "https://www.youtube.com"
    requested = _open_with_browser(url)
    return {
        "opened": requested,
        "launch_requested": requested,
        "url": url,
        "service": "youtube",
    }


def youtube_search(query: str) -> dict[str, object]:
    clean = query.strip()
    if not clean:
        return open_youtube()
    url = f"https://www.youtube.com/results?search_query={quote_plus(clean)}"
    requested = _open_with_browser(url)
    return {
        "opened": requested,
        "launch_requested": requested,
        "query": clean,
        "url": url,
        "service": "youtube",
    }


def open_spotify(_: str = "") -> dict[str, object]:
    try:
        os.startfile("spotify:")
        time.sleep(0.12)
        verified = _process_running("Spotify.exe")
        return {
            "opened": True,
            "opened_verified": verified,
            "launch_requested": True,
            "app": "spotify",
            "method": "uri",
        }
    except OSError:
        return open_app("spotify")


def spotify_search(argument: str) -> dict[str, object]:
    clean = argument.strip()
    if not clean:
        return {
            "opened": False,
            "playback_confirmed": False,
            "error": "missing_media_query",
        }
    try:
        os.startfile(f"spotify:search:{quote_plus(clean)}")
        time.sleep(0.12)
        verified = _process_running("Spotify.exe")
        return {
            "opened": True,
            "opened_verified": verified,
            "launch_requested": True,
            "search_opened": True,
            "playback_confirmed": False,
            "query": clean,
            "service": "spotify",
            "method": "uri_search",
        }
    except OSError:
        return {
            "opened": False,
            "playback_confirmed": False,
            "query": clean,
            "service": "spotify",
            "error": "spotify_local_launch_failed",
        }


def spotify_web_search(argument: str) -> dict[str, object]:
    clean = argument.strip()
    if not clean:
        return {
            "opened": False,
            "playback_confirmed": False,
            "error": "missing_media_query",
        }
    url = f"https://open.spotify.com/search/{quote_plus(clean)}"
    requested = _open_with_browser(url)
    return {
        "opened": requested,
        "launch_requested": requested,
        "search_opened": requested,
        "playback_confirmed": False,
        "query": clean,
        "url": url,
        "service": "spotify_web",
    }


def open_app(name: str) -> dict[str, object]:
    clean = name.strip()
    if not clean:
        return {"opened": False, "error": "missing_app_name"}
    return _APPLICATION_RESOLVER.launch(clean)


def _open_with_browser(url: str) -> bool:
    return bool(webbrowser.open(url, new=2, autoraise=True))


def _running_browser_name() -> str:
    if os.name != "nt":
        return ""
    names = {
        "chrome.exe": "Google Chrome",
        "msedge.exe": "Microsoft Edge",
        "firefox.exe": "Mozilla Firefox",
        "brave.exe": "Brave",
        "opera.exe": "Opera",
    }
    try:
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
            creationflags=_no_window_flag(),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    output = completed.stdout.casefold()
    for executable, label in names.items():
        if executable.casefold() in output:
            return label
    return ""


def _process_running(image_name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
            creationflags=_no_window_flag(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return image_name.casefold() in completed.stdout.casefold()


def _no_window_flag() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
