"""Open the latest generalization session in Windows Explorer."""
from __future__ import annotations

import os

from common import OUT, latest_session


def main() -> None:
    target = latest_session(complete_only=False) or OUT
    target.mkdir(parents=True, exist_ok=True)
    os.startfile(str(target))


if __name__ == "__main__":
    main()
