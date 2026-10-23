def fetch_url(_: str = "") -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "network_disabled_in_clean_base",
    }


def open_browser(_: str = "") -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "automatic_browser_opening_disabled",
    }
