"""Read-only math/TTS surface adapter. No sentinel substitution or core authority."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LiteralSpan:
    start: int
    end: int
    kind: str


_TECHNICAL = (
    ('CODE',re.compile(r'```[\s\S]*?```|`[^`\n]+`')),
    ('URL',re.compile(r'\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s`<>"\']+')),
    ('WINDOWS_PATH',re.compile(r'(?:[A-Za-z]:[\\/]|\\\\)[^\r\n`<>"|?*]*?\.[A-Za-z0-9]{1,12}(?=$|[\s`,;.!?])|(?:[A-Za-z]:[\\/]|\\\\)[^\s`<>"|?*]+')),
    ('POSIX_PATH',re.compile(r'(?<![\w/])(?:/|\./|\.\./)[\w.~-]+(?:/[\w.~-]+)*')),
    ('IDENTIFIER',re.compile(r'\b[A-Za-z][\w.-]{1,24}:[A-Za-z0-9][\w:.-]{3,}')),
    ('HASH',re.compile(r'(?<!\w)[a-fA-F0-9]{32,128}(?!\w)')),
    ('COMMAND',re.compile(r'(?m)^\s*[\w.-]+(?:\s+--?[\w-]+(?:=[^\s]+)?|\s+[\w./\\-]+\.[A-Za-z0-9]{1,12})+')),
    ('FILENAME',re.compile(r'(?<![\w.])(?=[\w.-]*[A-Za-z_])[\w-]+(?:\.[A-Za-z0-9]{1,12})+(?!\w)')),
    ('CODE_IDENTIFIER',re.compile(r'\b(?:[A-Za-z]\w*_[A-Za-z]\w*|[A-Za-z_]\w*::[A-Za-z_]\w*)\b')),
)
_MATH = re.compile(
    r'\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)|\$\$[\s\S]*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$)'
    r'|(?:\\(?:sqrt|frac|vec)\{[^{}]*\}(?:\{[^{}]*\})?|sqrt\([^()\n]+\))'
    r'|[A-Za-z]\s*=\s*[A-Za-z0-9]+(?:\s*[+*/^-]\s*[A-Za-z0-9]+)+'
    r'|[A-Za-z]_[A-Za-z0-9]+\s*[+*/=-]\s*[A-Za-z]_[A-Za-z0-9]+'
    r'|∫[^\n,.;!?]{1,100}\bd[A-Za-z]\b')
_ADAPTER_MARKER = re.compile(r'\b(?:ANYDESPATHPLACEHOLDER\d+TOKEN|TECH_LITERAL_\d+|PATH_TOKEN(?:_\d+)?|TOKEN)\b')


def literal_spans(text):
    spans=[]
    for kind,pattern in _TECHNICAL:
        for match in pattern.finditer(text):
            start,end=match.span()
            if kind in {'URL','WINDOWS_PATH','POSIX_PATH','IDENTIFIER'}:
                while end>start and text[end-1] in '.,;!?)':end-=1
            if end>start and not any(start<s.end and end>s.start for s in spans):
                spans.append(LiteralSpan(start,end,kind))
    return tuple(sorted(spans,key=lambda s:s.start))


def _introduced_marker(source,result):
    return any(source.count(m.group())<result.count(m.group()) for m in _ADAPTER_MARKER.finditer(result))


def _present_technical(text, *, transform=None):
    from anydes.continuity_v2.math_presentation import MathPresentation,normalize_speech,verbalize_formula
    source=str(text or '')
    if transform is None:
        def transform(formula):
            if formula.startswith(('\\[','\\(','$')):
                return normalize_speech(formula)
            return verbalize_formula(formula),1
    spans=literal_spans(source)
    speech=[]
    regions=0
    fallback=False

    def prose(fragment):
        nonlocal regions,fallback
        cursor=0
        for match in _MATH.finditer(fragment):
            speech.append(fragment[cursor:match.start()])
            formula=match.group()
            try:
                adapted,count=transform(formula)
                if not isinstance(adapted,str) or not adapted.strip() or _introduced_marker(source,adapted):
                    raise ValueError('INVALID_MATH_ADAPTATION')
                speech.append(adapted)
                regions+=max(1,count)
            except Exception:
                speech.append(formula)
                fallback=True
            cursor=match.end()
        speech.append(fragment[cursor:])

    cursor=0
    preserved=[]
    for span in spans:
        prose(source[cursor:span.start])
        literal=source[span.start:span.end]
        # Each technical segment is appended directly, outside every transform.
        speech.append(literal)
        preserved.append((literal,speech[-1]))
        cursor=span.end
    prose(source[cursor:])
    spoken=''.join(speech)
    if (any(original!=rendered for original,rendered in preserved)
            or _introduced_marker(source,spoken) or source.strip() and not spoken.strip()):
        spoken=source
        regions=0
        fallback=True
    return MathPresentation(source,source,spoken,regions,fallback)


def present_technical(text, *, transform=None):
    from anydes.continuity_v2.math_presentation import MathPresentation
    source=str(text or '')
    try:
        return _present_technical(source,transform=transform)
    except Exception:
        return MathPresentation(source,source,source,0,True)


def install():
    from anydes.continuity_v2 import math_runtime
    # The existing UI and voice hooks both dereference this function at call time.
    # Their lifecycle, manager, engine, playback and identity are unchanged.
    math_runtime.present_math=present_technical
