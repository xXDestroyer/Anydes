from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Literal


ResolutionStatus = Literal["resolved", "ambiguous", "not_found"]


@dataclass(frozen=True)
class ApplicationCandidate:
    display_name: str
    launch_target: str
    method: str
    aliases: tuple[str, ...] = ()
    process_name: str = ""


@dataclass(frozen=True)
class ApplicationResolution:
    requested_name: str
    status: ResolutionStatus
    candidate: ApplicationCandidate | None = None
    alternatives: tuple[ApplicationCandidate, ...] = ()
    confidence: float = 0.0
    method: str = ""


_EXECUTABLE_ALIASES = {
    "chrome": ("google chrome",),
    "msedge": ("edge", "microsoft edge"),
    "spotify": ("spotify",),
    "notepad": ("bloc de notas", "notepad"),
    "calc": ("calculadora", "calculator"),
    "explorer": ("explorador", "explorador de archivos"),
    "code": ("visual studio code", "vscode", "codigo"),
    "winword": ("word", "microsoft word"),
    "excel": ("excel", "microsoft excel"),
    "powerpnt": ("powerpoint", "microsoft powerpoint"),
    "outlook": ("outlook", "microsoft outlook"),
}

_SAFE_BUILTINS = (
    ApplicationCandidate("Bloc de notas", "notepad", "command", ("notepad",)),
    ApplicationCandidate("Calculadora", "calc", "command", ("calculator",)),
    ApplicationCandidate(
        "Explorador de archivos",
        "explorer",
        "command",
        ("explorador", "file explorer"),
    ),
    ApplicationCandidate(
        "Spotify",
        r"shell:AppsFolder\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify",
        "shell",
        ("spotify",),
        "Spotify.exe",
    ),
    ApplicationCandidate(
        "WhatsApp",
        r"shell:AppsFolder\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App",
        "shell",
        ("whatsapp",),
    ),
)


class ApplicationResolver:
    """Indexes launchable Windows applications without executing them."""

    def __init__(
        self,
        *,
        candidates: Iterable[ApplicationCandidate] | None = None,
        cache_seconds: float = 300.0,
    ) -> None:
        self._provided = tuple(candidates) if candidates is not None else None
        self._cache_seconds = max(1.0, cache_seconds)
        self._lock = threading.RLock()
        self._cached_at = 0.0
        self._cached: tuple[ApplicationCandidate, ...] = ()

    def candidates(self, *, refresh: bool = False) -> tuple[ApplicationCandidate, ...]:
        with self._lock:
            if (
                not refresh
                and self._cached
                and time.monotonic() - self._cached_at < self._cache_seconds
            ):
                return self._cached
            values = (
                self._provided
                if self._provided is not None
                else tuple(self._discover_windows())
            )
            self._cached = _deduplicate(values)
            self._cached_at = time.monotonic()
            return self._cached

    def resolve(self, requested_name: str) -> ApplicationResolution:
        requested = _normalize(requested_name)
        if not requested or _looks_like_command(requested_name):
            return ApplicationResolution(requested_name, "not_found")
        ranked: list[tuple[float, ApplicationCandidate]] = []
        for candidate in self.candidates():
            score = _candidate_score(requested, candidate)
            if score >= 0.68:
                ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], _normalize(item[1].display_name)))
        if not ranked:
            return ApplicationResolution(requested_name, "not_found")
        top_score, top = ranked[0]
        alternatives = tuple(item[1] for item in ranked[:5])
        if len(ranked) > 1 and top_score < 0.96 and top_score - ranked[1][0] < 0.06:
            return ApplicationResolution(
                requested_name,
                "ambiguous",
                alternatives=alternatives,
                confidence=top_score,
                method="ranked_index",
            )
        return ApplicationResolution(
            requested_name,
            "resolved",
            candidate=top,
            alternatives=alternatives,
            confidence=top_score,
            method=top.method,
        )

    def launch(self, requested_name: str) -> dict[str, object]:
        resolution = self.resolve(requested_name)
        if resolution.status != "resolved" or resolution.candidate is None:
            return {
                "opened": False,
                "opened_verified": False,
                "app": requested_name.strip(),
                "resolution_status": resolution.status,
                "alternatives": [
                    item.display_name for item in resolution.alternatives
                ],
                "error": (
                    "application_ambiguous"
                    if resolution.status == "ambiguous"
                    else "unknown_or_unapproved_app"
                ),
            }
        candidate = resolution.candidate
        try:
            if candidate.method in {"shortcut", "shell"}:
                os.startfile(candidate.launch_target)
                process = None
            else:
                process = subprocess.Popen(
                    [candidate.launch_target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_no_window_flag(),
                )
            time.sleep(0.12)
            verified = _verify_launch(candidate, process)
            return {
                "opened": True,
                "opened_verified": verified,
                "launch_requested": True,
                "app": candidate.display_name,
                "requested_app": requested_name.strip(),
                "candidate": candidate.launch_target,
                "resolution_method": candidate.method,
                "resolution_confidence": resolution.confidence,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "opened": False,
                "opened_verified": False,
                "app": candidate.display_name,
                "requested_app": requested_name.strip(),
                "candidate": candidate.launch_target,
                "error": "application_launch_failed",
                "error_type": type(exc).__name__,
            }

    def _discover_windows(self) -> Iterable[ApplicationCandidate]:
        yield from _SAFE_BUILTINS
        yield from _discover_start_menu()
        yield from _discover_app_paths()
        yield from _discover_path_aliases()


def _discover_start_menu() -> Iterable[ApplicationCandidate]:
    roots = (
        Path(os.environ.get("ProgramData", ""))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs",
        Path(os.environ.get("APPDATA", ""))
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs",
    )
    for root in roots:
        if not root.is_dir():
            continue
        try:
            shortcuts = root.rglob("*.lnk")
            for shortcut in shortcuts:
                name = shortcut.stem.strip()
                if name:
                    yield ApplicationCandidate(
                        display_name=name,
                        launch_target=str(shortcut),
                        method="shortcut",
                    )
        except OSError:
            continue


def _discover_app_paths() -> Iterable[ApplicationCandidate]:
    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:
        return
    roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    access_modes = (
        winreg.KEY_READ,
        winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
        winreg.KEY_READ | getattr(winreg, "KEY_WOW64_32KEY", 0),
    )
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for access in access_modes:
            try:
                key = winreg.OpenKey(root, base, 0, access)
            except OSError:
                continue
            with key:
                index = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            target = str(winreg.QueryValue(subkey, None) or "").strip('"')
                    except OSError:
                        continue
                    if not target or not Path(target).is_file():
                        continue
                    executable = Path(target).stem
                    aliases = _EXECUTABLE_ALIASES.get(_normalize(executable), ())
                    identity = (_normalize(executable), target.casefold())
                    if identity in seen:
                        continue
                    seen.add(identity)
                    yield ApplicationCandidate(
                        display_name=_display_name(executable),
                        launch_target=target,
                        method="app_paths",
                        aliases=aliases,
                        process_name=Path(target).name,
                    )


def _discover_path_aliases() -> Iterable[ApplicationCandidate]:
    for executable, aliases in _EXECUTABLE_ALIASES.items():
        target = shutil.which(executable)
        if not target:
            continue
        yield ApplicationCandidate(
            display_name=_display_name(executable),
            launch_target=target,
            method="path",
            aliases=aliases,
            process_name=Path(target).name,
        )


def _candidate_score(requested: str, candidate: ApplicationCandidate) -> float:
    names = {
        _normalize(candidate.display_name),
        _normalize(Path(candidate.launch_target).stem),
        *(_normalize(alias) for alias in candidate.aliases),
    }
    names.discard("")
    if requested in names:
        return 1.0
    requested_tokens = set(requested.split())
    best = 0.0
    for name in names:
        if requested in name or name in requested:
            best = max(best, 0.9)
        name_tokens = set(name.split())
        if requested_tokens and requested_tokens <= name_tokens:
            best = max(best, 0.88)
        best = max(best, SequenceMatcher(None, requested, name).ratio())
    return best


def _deduplicate(
    candidates: Iterable[ApplicationCandidate],
) -> tuple[ApplicationCandidate, ...]:
    selected: dict[tuple[str, str], ApplicationCandidate] = {}
    for candidate in candidates:
        key = (
            _normalize(candidate.display_name),
            candidate.launch_target.casefold(),
        )
        selected.setdefault(key, candidate)
    return tuple(selected.values())


def _display_name(executable: str) -> str:
    canonical = {
        "winword": "Microsoft Word",
        "powerpnt": "Microsoft PowerPoint",
        "msedge": "Microsoft Edge",
        "code": "Visual Studio Code",
    }
    normalized = _normalize(executable)
    return canonical.get(normalized, executable.replace("_", " ").title())


def _verify_launch(
    candidate: ApplicationCandidate,
    process: subprocess.Popen | None,
) -> bool:
    if process is not None and process.poll() is None:
        return True
    if candidate.process_name:
        return _process_running(candidate.process_name)
    return False


def _process_running(image_name: str) -> bool:
    if os.name != "nt" or not image_name:
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


def _looks_like_command(value: str) -> bool:
    return bool(re.search(r"[;&|<>]|\b(?:cmd|powershell)\s+/", value, re.IGNORECASE))


def _normalize(value: str) -> str:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


def _no_window_flag() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
