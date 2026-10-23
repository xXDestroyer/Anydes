"""Technical source names only; document names are not normalized here."""
from pathlib import Path
import re
import unicodedata


def canonical_source_name(value, format_id):
    raw = str(value)
    if not raw or len(raw) > 180 or any(c in raw for c in '/\\:<>|?*') or raw in {'.', '..'}:
        raise ValueError('source_filename_invalid')
    name = unicodedata.normalize('NFKC', raw).strip()
    reserved = {'CON','PRN','AUX','NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}
    if Path(name).stem.upper() in reserved or any(unicodedata.category(c).startswith('C') for c in name):
        raise ValueError('source_filename_invalid')
    explicit = bool(Path(name).suffix)
    name = re.sub(r'\s+', '_', name)
    if not explicit:
        name = name.lower() + '.' + format_id
    if Path(name).suffix.casefold() != '.' + format_id or name.endswith((' ', '.')):
        raise ValueError('source_filename_format_mismatch')
    return name
