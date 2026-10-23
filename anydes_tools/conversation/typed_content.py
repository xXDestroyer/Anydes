"""Approved presentation data, not executable markup or model protocol."""
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
import html
from threading import RLock


class ContentKind(str, Enum):
    TEXT = 'TEXT'
    CODE = 'CODE'
    MATH = 'MATH'
    DOCUMENT_PREVIEW = 'DOCUMENT_PREVIEW'
    ERROR_EXPLANATION = 'ERROR_EXPLANATION'


@dataclass(frozen=True)
class ContentSegment:
    kind: ContentKind
    text: str
    language: str = ''


_APPROVED = OrderedDict()
_LOCK = RLock()


def approve_literal_publication(final):
    if not any(s.kind == ContentKind.CODE for s in final.content_segments):
        return
    key = hashlib.sha256(final.visible_text.encode()).hexdigest()
    with _LOCK:
        _APPROVED[key] = (final.visible_text,final.content_segments)
        _APPROVED.move_to_end(key)
        while len(_APPROVED) > 32:
            _APPROVED.popitem(last=False)


def literal_segments(text):
    with _LOCK:
        entry = _APPROVED.get(hashlib.sha256(str(text).encode()).hexdigest())
        return entry[1] if entry and entry[0] == text else ()


def code_html(segments):
    rows = []
    for segment in segments:
        escaped = html.escape(segment.text)
        rows.append('<pre style="white-space:pre; font-family:Consolas">'+escaped+'</pre>'+('<p></p>' if segment.text.endswith('\n') else '')
            if segment.kind == ContentKind.CODE else '<span>'+escaped.replace('\n','<br>')+'</span>')
    return ''.join(rows)


def install_literal_presentation():
    """Staging adapters only; B4 identity/audio implementation is untouched."""
    from anydes.continuity_v2 import math_runtime, math_presentation
    if getattr(math_runtime,'_candidate_literal_content',False):
        return
    original = math_runtime.present_math
    def present(text):
        if literal_segments(text):
            return math_presentation.MathPresentation(text,text,text,0,False)
        return original(text)
    math_runtime.present_math = present
    math_runtime._candidate_literal_content = True

    import anydes.voice_input.integration as integration
    install = integration.install
    def install_ui(module, *args, **kwargs):
        result = install(module,*args,**kwargs)
        panel_class = getattr(module,'ChatPanel',None)
        if panel_class and not getattr(panel_class,'_candidate_literal_content',False):
            render = panel_class._render
            palette = render.__globals__['PALETTE']
            def render_typed(panel):
                if not any(literal_segments(message) for _,message,user in panel._messages if not user):
                    return render(panel)
                rows = []
                for speaker,message,user in panel._messages:
                    color = palette.pink if user else palette.text_secondary
                    label = '<span style="color:'+color+'; font-weight:600">'+html.escape(speaker)+'</span>  '
                    segments = literal_segments(message) if not user else ()
                    body = code_html(segments) if segments else '<span style="color:'+palette.text+'">'+html.escape(message)+'</span>'
                    rows.append(label+body)
                panel.text.setHtml('<br>'.join(rows))
                panel.text.verticalScrollBar().setValue(panel.text.verticalScrollBar().maximum())
            panel_class._render = render_typed
            panel_class._candidate_literal_content = True
        return result
    integration.install = install_ui
