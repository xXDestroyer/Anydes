"""Best-effort telemetry samples, not an atomic filesystem inventory."""
import os
from pathlib import Path
import stat
import traceback


TRANSIENT = (FileNotFoundError, NotADirectoryError)


def scan_sizes(root, record):
    root = Path(root)
    record('observer_scan_start')
    pending, count, size, skipped = [root], 0, 0, 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as scan:
                entries = list(scan)
        except TRANSIENT as error:
            skipped += 1
            record('observer_transient_race', operation='enumerate', error_type=type(error).__name__)
            continue
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except TRANSIENT as error:
                skipped += 1
                record('observer_transient_race', operation='stat', error_type=type(error).__name__)
                continue
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                count += 1
                size += metadata.st_size
                record('observer_item', bytes=metadata.st_size)
    return {'data_files': count, 'data_bytes': size, 'skipped_items': skipped}


def read_metadata_bytes(path, record):
    try:
        return Path(path).read_bytes()
    except TRANSIENT as error:
        record('observer_transient_race', operation='read', error_type=type(error).__name__)
        return None


class ObserverBoundary:
    """Unexpected errors quarantine observation, never the application's work.

    BaseException (process exit/interrupt) is deliberately not swallowed.
    This policy applies only to telemetry callbacks, not runtime operations.
    """
    def __init__(self, record):
        self.record = record
        self.disabled = False

    def call(self, phase, callback):
        if self.disabled:
            return None
        try:
            return callback()
        except Exception as error:
            self.disabled = True
            self.record('observer_unexpected_error', phase=phase,
                        error_type=type(error).__name__, errno=getattr(error, 'errno', None),
                        frames=[{'module':Path(frame.filename).name, 'line':frame.lineno,
                                 'function':frame.name} for frame in traceback.extract_tb(error.__traceback__)],
                        policy='DISABLE_OBSERVER_NOT_RUNTIME')
            return None
