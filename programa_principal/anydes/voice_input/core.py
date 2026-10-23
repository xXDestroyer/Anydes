"""CPU-only speech input: bounded capture, VAD endpointing and latest-only STT."""
from __future__ import annotations

from collections import Counter,deque
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import queue
import re
import threading
import time
import uuid

from .environment import HERE,ROOT,MODELS,setup
setup()
import numpy as np
from .capture import Microphone
from .diagnostics import SpeechRcaStore
from .telemetry import mark


BASELINE_CONFIG='base_int8_es_beam3_clean_v1'
BASELINE_LANGUAGE='es'
BASELINE_CONTEXT='none'


@dataclass(frozen=True)
class UserInputEvent:
    source: str
    text: str
    timestamp: str
    confidence: float | None = None
    audio_id: str | None = None

    def __post_init__(self):
        if self.source not in {'keyboard','voice'}:
            raise ValueError('Unknown input source')

    @classmethod
    def keyboard(cls,text):
        return cls('keyboard',text,datetime.now(timezone.utc).isoformat())


class InputText(str):
    """Existing LatestTurn keeps the event without changing its queue contract."""
    def __new__(cls,event):
        result=super().__new__(cls,event.text)
        result.event=event
        return result


class Segmenter:
    """20 ms PCM frames; retain pre-roll and internal pauses, reject pure silence."""
    def __init__(self,mode='vad',start_frames=4,end_frames=40,max_frames=1500,
                 pre_roll_frames=15,post_roll_frames=15):
        self.mode=mode;self.start_frames=start_frames;self.end_frames=end_frames
        self.max_frames=max_frames;self.pre_roll_frames=max(1,int(pre_roll_frames))
        self.post_roll_frames=max(0,int(post_roll_frames));self.last_metrics={}
        self.reset()

    def reset(self):
        self.pre=deque(maxlen=self.pre_roll_frames);self.frames=[];self.flags=[];self.active=False
        self.on=0;self.off=0;self.voiced=0

    def push(self,pcm,voiced):
        if not self.active:
            self.pre.append((pcm,voiced))
            self.on=self.on+1 if voiced else 0
            if self.on < self.start_frames:
                return None,None
            self.active=True
            self.frames=[item[0] for item in self.pre]
            self.flags=[bool(item[1]) for item in self.pre]
            self.voiced=sum(self.flags)
            return 'start',None
        self.frames.append(pcm);self.flags.append(bool(voiced))
        self.voiced+=int(voiced)
        self.off=0 if voiced else self.off+1
        if len(self.frames)>=self.max_frames:
            return 'limit',self.finish()
        if self.mode=='vad' and self.off>=self.end_frames:
            return 'end',self.finish()
        return None,None

    def finish(self):
        frames=list(self.frames);flags=list(self.flags)
        removed_tail_frames=max(0,self.off-self.post_roll_frames)
        if removed_tail_frames and len(frames)>removed_tail_frames:
            frames=frames[:-removed_tail_frames];flags=flags[:-removed_tail_frames]
        voiced_indices=[index for index,value in enumerate(flags) if value]
        audio=np.concatenate(frames) if len(voiced_indices)>=6 and frames else None
        if audio is not None:
            first=voiced_indices[0];last=voiced_indices[-1]
            voiced_pcm=np.concatenate([frame for frame,flag in zip(frames,flags) if flag])
            tail_pcm=np.concatenate(frames[last+1:]) if last+1<len(frames) else np.empty(0,np.float32)
            self.last_metrics=dict(frame_ms=20,total_frames=len(frames),voiced_frames=len(voiced_indices),
                voice_span_frames=last-first+1,pre_voice_frames=first,
                post_voice_frames=len(frames)-last-1,trimmed_tail_frames=removed_tail_frames,
                voice_detected_duration_s=len(voiced_indices)*.02,
                voice_span_duration_s=(last-first+1)*.02,
                silence_before_voice_s=first*.02,
                silence_after_voice_s=(len(frames)-last-1)*.02,
                speech_rms=float(np.sqrt(np.mean(voiced_pcm.astype(np.float64)**2))),
                tail_rms=float(np.sqrt(np.mean(tail_pcm.astype(np.float64)**2))) if len(tail_pcm) else 0.0)
        else:
            self.last_metrics=dict(frame_ms=20,total_frames=len(frames),voiced_frames=len(voiced_indices),
                voice_span_frames=0,pre_voice_frames=0,post_voice_frames=0,
                trimmed_tail_frames=removed_tail_frames,voice_detected_duration_s=0,
                voice_span_duration_s=0,silence_before_voice_s=0,silence_after_voice_s=0,
                speech_rms=0,tail_rms=0)
        self.reset()
        return audio


class PttPreRoll:
    """Bounded physical PCM history captured before the logical PTT press."""
    def __init__(self,max_ms=500,frame_ms=20):
        self.frame_ms=frame_ms
        self.frames=deque(maxlen=max(1,round(max_ms/frame_ms)))

    def append(self,pcm,at,voiced):
        self.frames.append((pcm.copy(),at,voiced))

    def snapshot(self,requested_ms,press_at=None):
        history=[item for item in self.frames if press_at is None or item[1]<=press_at]
        wanted=max(1,(int(requested_ms)+self.frame_ms-1)//self.frame_ms)
        return history,history[-wanted:]


class WebRtcDetector:
    def __init__(self):
        import webrtcvad
        self.vad=webrtcvad.Vad(2)

    def __call__(self,pcm):
        pcm16=(np.clip(pcm,-1,1)*32767).astype('<i2').tobytes()
        return self.vad.is_speech(pcm16,16000) and float(np.sqrt(np.mean(pcm*pcm)))>.0006


class SileroDetector:
    """Stateful CPU ONNX inference using the pinned faster-whisper v6 asset.

    Unlike its offline wrapper, recurrent state/context survive between blocks.
    20ms capture is buffered into the model's required 512-sample windows.
    """
    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.session=get_vad_model().session
        self.h=np.zeros((1,1,128),np.float32)
        self.c=np.zeros((1,1,128),np.float32)
        self.context=np.zeros(64,np.float32)
        self.pending=np.empty(0,np.float32);self.probability=0.0

    def __call__(self,pcm):
        self.pending=np.concatenate((self.pending,pcm))
        while len(self.pending)>=512:
            chunk=self.pending[:512];self.pending=self.pending[512:]
            data=np.concatenate((self.context,chunk))[None,:]
            out,self.h,self.c=self.session.run(None,{'input':data,'h':self.h,'c':self.c})
            self.context=chunk[-64:].copy()
            self.probability=float(out.reshape(-1)[-1])
        return self.probability>=.6 and float(np.sqrt(np.mean(pcm*pcm)))>.0006


def hallucination_diagnostics(text,audio_seconds,segments):
    words=re.findall(r"[\w+]+",text.lower(),flags=re.UNICODE)
    reasons=[];signature=None;count=0
    if len(words)>=20 and len(words)>max(24,audio_seconds*8):
        reasons.append('word_rate_implausible')
    for width in range(1,min(6,len(words)//4)+1):
        grams=[' '.join(words[index:index+width]) for index in range(len(words)-width+1)]
        if not grams:continue
        candidate,repetitions=Counter(grams).most_common(1)[0]
        coverage=repetitions*width/max(1,len(words))
        if repetitions>=4 and coverage>=.5 and repetitions>count:
            signature=candidate;count=repetitions
    if signature:
        reasons.append('pathological_repetition')
    compression=max((float(row.get('compression_ratio') or 0) for row in segments),default=0)
    if compression>2.4:reasons.append('high_compression_ratio')
    return dict(suspected=bool(reasons),reasons=reasons,word_count=len(words),
        words_per_second=len(words)/max(audio_seconds,.001),repeated_phrase=signature,
        repeated_phrase_count=count,max_compression_ratio=compression)


class Recognizer:
    def __init__(self,name='base',threads=2):
        self.name=name
        if name not in {'base','vosk'}:
            raise ValueError('Baseline congelada: solo Whisper Base INT8.')
        started=time.perf_counter()
        if name=='vosk':
            from vosk import Model,SetLogLevel
            SetLogLevel(-1)
            self.model=Model(str(ROOT/'programa_principal/modelos/vosk/vosk-model-small-es-0.42'))
        else:
            from faster_whisper import WhisperModel
            if not (MODELS/name/'model.bin').is_file():
                raise FileNotFoundError(f'STT {name} no preparado: ejecuta prepare.py.')
            self.model=WhisperModel(str(MODELS/name),device='cpu',compute_type='int8',
                cpu_threads=threads,num_workers=1,local_files_only=True)
        self.load_ms=(time.perf_counter()-started)*1000

    def transcribe(self,audio):
        started=time.perf_counter()
        if self.name=='vosk':
            from vosk import KaldiRecognizer
            rec=KaldiRecognizer(self.model,16000)
            rec.SetWords(True)
            rows=[]
            pcm=(np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
            for offset in range(0,len(pcm),8000):
                if rec.AcceptWaveform(pcm[offset:offset+8000]):
                    rows.append(json.loads(rec.Result()))
            rows.append(json.loads(rec.FinalResult()))
            text=' '.join(r.get('text','') for r in rows).strip()
            scores=[w['conf'] for r in rows for w in r.get('result',[])]
            confidence=float(np.mean(scores)) if scores else None
        else:
            segments,info=self.model.transcribe(audio,language=BASELINE_LANGUAGE,beam_size=3,
                temperature=0,condition_on_previous_text=False,vad_filter=False,
                initial_prompt=None,hotwords=None,word_timestamps=False)
            rows=list(segments)
            accepted=[s for s in rows if float(getattr(s,'no_speech_prob',0))<.6]
            text=' '.join(s.text.strip() for s in accepted).strip()
            segment_details=[dict(index=index,text=s.text.strip(),
                start=float(getattr(s,'start',0)),end=float(getattr(s,'end',0)),
                no_speech_prob=float(getattr(s,'no_speech_prob',0)),
                avg_logprob=float(getattr(s,'avg_logprob',0)),
                compression_ratio=float(getattr(s,'compression_ratio',0)))
                for index,s in enumerate(rows)]
            # Decoder log-probability is not a calibrated confidence score.
            confidence=None
        elapsed=(time.perf_counter()-started)*1000
        audio_seconds=len(audio)/16000
        if self.name=='vosk':
            segment_details=[];language='es';language_probability=None
        else:
            language=str(getattr(info,'language',BASELINE_LANGUAGE))
            language_probability=getattr(info,'language_probability',None)
        hallucination=hallucination_diagnostics(text,audio_seconds,segment_details)
        return dict(text=text,confidence=confidence,stt_ms=elapsed,rtf=elapsed/max(1,len(audio)/16),
            audio_seconds=audio_seconds,backend=self.name,model='Whisper Base INT8' if self.name=='base' else 'Vosk',
            config=BASELINE_CONFIG,language_requested=BASELINE_LANGUAGE,
            language_backend=language,language_probability=language_probability,
            context=BASELINE_CONTEXT,condition_on_previous_text=False,
            temperature=0,beam_size=3,initial_prompt=None,hotwords=None,
            whisper_segment_count=len(segment_details),whisper_segments=segment_details,
            no_speech_probability_max=max((row['no_speech_prob'] for row in segment_details),default=None),
            no_speech_probability_mean=float(np.mean([row['no_speech_prob'] for row in segment_details]))
                if segment_details else None,hallucination=hallucination)


class VoiceInput:
    """No cognition/TTS/model ownership. All outbound events carry a revision."""
    def __init__(self,emit,recognizer_factory=Recognizer,mic_factory=Microphone):
        self.emit=emit;self.recognizer_factory=recognizer_factory;self.mic_factory=mic_factory
        self.lock=threading.RLock();self.revision=0
        self.stop_event=threading.Event();self.finish_event=threading.Event();self.reset_event=threading.Event()
        self.ptt_begin_event=threading.Event()
        self.jobs=queue.Queue(maxsize=1)
        self.capture_thread=None;self.stt_thread=None;self.mic=None;self.recognizer=None
        self.mode='ptt';self.backend='base';self.alive=True
        self.model_label='Whisper Base INT8';self.config_label=BASELINE_CONFIG
        self.language_label=BASELINE_LANGUAGE;self.context_label=BASELINE_CONTEXT
        self.pre_roll_ms=400;self.pending_ptt_revision=0;self.pending_ptt_at=0.0
        self.stale_discarded=0;self.audio_id=None
        self.diagnostic_enabled=False;self.diagnostic_reference='';self.diagnostic_store=None

    def configure_diagnostics(self,enabled,reference='',directory=None):
        with self.lock:
            self.diagnostic_enabled=bool(enabled);self.diagnostic_reference=reference.strip()
            if enabled and self.diagnostic_store is None:
                from .telemetry import get_monitor
                monitor=get_monitor();root=directory or (monitor.directory if monitor else ROOT/'datos'/'telemetria'/'voice_input')
                self.diagnostic_store=SpeechRcaStore(root)

    def _emit(self,kind,rev=None,**fields):
        if kind in {'mic_ready','capture_closed','voice_detected','transcribing',
                    'stt_loading','stt_ready','transcription_started','transcript',
                    'hallucination_suspected','stale_discarded','error'}:
            # Capture stage memory before Qt dispatch, without storing recognized text.
            metric_keys=('model','config','language','language_requested','language_backend',
                'stt_ms','rtf','audio_seconds','endpoint_to_text_ms','voice_end_to_text_ms',
                'language_drift','english_markers','load_ms','decoder_state')
            metrics={key:fields[key] for key in metric_keys if key in fields}
            hallucination=fields.get('hallucination')
            if isinstance(hallucination,dict):
                metrics['hallucination_suspected']=bool(hallucination.get('suspected'))
                metrics['hallucination_reasons']=hallucination.get('reasons',[])
            mark('input_'+kind,revision=self.revision if rev is None else rev,
                audio_id=fields.get('audio_id') or fields.get('input',{}).get('audio_id'),
                backend=self.backend,message=fields.get('message'),**metrics)
        self.emit(dict(kind=kind,revision=self.revision if rev is None else rev,
                       at=time.monotonic(),**fields))

    def invalidate(self):
        with self.lock:
            self.revision+=1
            self.reset_event.set()
            try:
                self.jobs.get_nowait()
                self.stale_discarded+=1
            except queue.Empty:
                pass
            return self.revision

    def start(self,device=None,mode='ptt',backend='base'):
        if self.capture_thread and self.capture_thread.is_alive():
            raise RuntimeError('Ya hay una captura activa.')
        if not self.alive:
            raise RuntimeError('La sesion de entrada esta cerrada.')
        self.mode=mode;self.backend=backend
        self.invalidate();self.reset_event.clear()
        self.stop_event.clear();self.finish_event.clear();self.ptt_begin_event.clear()
        if self.stt_thread is None:
            self.stt_thread=threading.Thread(target=self._decode_loop,name='anydes-stt',daemon=True)
            self.stt_thread.start()
        self.capture_thread=threading.Thread(target=self._capture,args=(device,),name='anydes-mic',daemon=True)
        self.capture_thread.start()

    def arm_ptt(self,device=None,backend='base',pre_roll_ms=400):
        """Open one explicit mic stream and keep only a bounded in-memory history."""
        if pre_roll_ms not in {250,400,500}:
            raise ValueError('Pre-roll PTT permitido: 250, 400 o 500 ms.')
        self.pre_roll_ms=pre_roll_ms
        self.start(device,'armed_ptt',backend)

    def begin_ptt(self,pre_roll_ms=None):
        if self.mode!='armed_ptt' or not self.capture_thread or not self.capture_thread.is_alive():
            raise RuntimeError('Primero prepara el microfono PTT.')
        if pre_roll_ms is not None:
            if pre_roll_ms not in {250,400,500}:
                raise ValueError('Pre-roll PTT permitido: 250, 400 o 500 ms.')
            self.pre_roll_ms=pre_roll_ms
        with self.lock:
            self.revision+=1
            self.pending_ptt_revision=self.revision
            self.pending_ptt_at=time.monotonic()
            try:
                self.jobs.get_nowait();self.stale_discarded+=1
            except queue.Empty:
                pass
        self.finish_event.clear();self.ptt_begin_event.set()
        return self.pending_ptt_revision

    def finish(self):
        self.finish_event.set()

    def stop(self):
        self.invalidate();self.stop_event.set()
        if self.capture_thread and self.capture_thread is not threading.current_thread():
            self.capture_thread.join(timeout=3)
            if self.capture_thread.is_alive():
                self._emit('error',message='La captura no se cerro a tiempo.')
                return
        self._emit('mic_off')

    def close(self):
        self.stop();self.alive=False
        if self.stt_thread:
            self.stt_thread.join(timeout=10)
        return not any(thread and thread.is_alive() for thread in (self.capture_thread,self.stt_thread))

    def _capture(self,device):
        rev=self.revision
        mic=None
        armed=self.mode=='armed_ptt'
        segment=None if armed else Segmenter(self.mode)
        last_meter=0;audio_id=None;last_voice_at=None
        diagnostic_enabled=self.diagnostic_enabled
        diagnostic_ring=deque(maxlen=100);diagnostic_raw=[];raw_start_at=None;segment_offset=0
        physical_ring=PttPreRoll()
        requested_pre_roll_ms=0;available_pre_roll_ms=0;ptt_press_offset_samples=0
        try:
            mic=self.mic_factory(device);self.mic=mic
            mic.start()
            detector=None if armed else SileroDetector()
            if diagnostic_enabled and self.mode=='ptt':audio_id=str(uuid.uuid4())
            self._emit('mic_ready',rev,stats=asdict(mic.stats))
            self._emit('ptt_armed' if armed else 'listening',rev,pre_roll_ms=self.pre_roll_ms)
            while not self.stop_event.is_set():
                if self.reset_event.is_set():
                    if segment is not None:segment.reset()
                    segment=None if armed else Segmenter(self.mode)
                    detector=None if armed else SileroDetector()
                    audio_id=None;last_voice_at=None
                    diagnostic_ring.clear();diagnostic_raw=[];raw_start_at=None;segment_offset=0
                    rev=self.revision;self.reset_event.clear()
                    # Keyboard supersession discards already captured old PCM.
                    while not mic.queue.empty():
                        try: mic.queue.get_nowait()
                        except queue.Empty: break
                if self.finish_event.is_set() and segment is not None:
                    self.finish_event.clear()
                    final_audio=segment.finish();segment_metrics=segment.last_metrics
                    self._submit(final_audio,rev,audio_id,last_voice_at,
                        diagnostic_raw,raw_start_at,segment_offset,mic,
                        requested_pre_roll_ms,available_pre_roll_ms,ptt_press_offset_samples,
                        segment_metrics)
                    if armed:
                        segment=None;audio_id=None;last_voice_at=None
                        detector=None
                        diagnostic_raw=[];raw_start_at=None;segment_offset=0
                        continue
                    break
                packet=mic.read()
                if packet is None: continue
                pcm,at=packet
                if diagnostic_enabled:diagnostic_ring.append(pcm)
                if diagnostic_enabled and self.mode=='ptt':
                    if not diagnostic_raw:raw_start_at=at-len(pcm)/mic.stats.sample_rate
                    diagnostic_raw.append(pcm)
                elif diagnostic_enabled and diagnostic_raw and not armed:
                    diagnostic_raw.append(pcm)
                if at-last_meter>=.08:
                    self._emit('levels',rev,stats=asdict(mic.stats));last_meter=at
                if armed:
                    physical_ring.append(pcm,at,False)
                    if self.ptt_begin_event.is_set() and segment is None:
                        self.ptt_begin_event.clear();rev=self.pending_ptt_revision
                        requested_pre_roll_ms=self.pre_roll_ms
                        all_buffered,selected=physical_ring.snapshot(
                            requested_pre_roll_ms,self.pending_ptt_at)
                        wanted=max(1,(requested_pre_roll_ms+19)//20)
                        segment=Segmenter('ptt',pre_roll_frames=wanted)
                        detector=SileroDetector()
                        audio_id=str(uuid.uuid4());last_voice_at=None
                        available_pre_roll_ms=len(all_buffered)*20
                        diagnostic_raw=[item[0] for item in all_buffered] if diagnostic_enabled else []
                        ptt_press_offset_samples=sum(len(item[0]) for item in all_buffered)
                        raw_start_at=(all_buffered[0][1]-len(all_buffered[0][0])/mic.stats.sample_rate) if all_buffered else at
                        for buffered_pcm,buffered_at,_ in selected:
                            buffered_voiced=detector(buffered_pcm)
                            if buffered_voiced:last_voice_at=buffered_at
                            event,audio=segment.push(buffered_pcm,buffered_voiced)
                            if event=='start':
                                if diagnostic_enabled:
                                    segment_offset=max(0,sum(map(len,diagnostic_raw))-sum(map(len,segment.frames)))
                                self._emit('voice_detected',rev,audio_id=audio_id)
                        self._emit('listening',rev,pre_roll_ms=requested_pre_roll_ms,
                            available_pre_roll_ms=available_pre_roll_ms)
                        if at<=self.pending_ptt_at:
                            continue
                    if segment is None:
                        continue
                    if diagnostic_enabled:diagnostic_raw.append(pcm)
                    voiced=detector(pcm)
                else:
                    voiced=detector(pcm)
                if voiced:last_voice_at=at
                event,audio=segment.push(pcm,voiced)
                if event=='start':
                    if not armed:
                        with self.lock:
                            self.revision+=1;rev=self.revision
                    audio_id=audio_id or str(uuid.uuid4())
                    if diagnostic_enabled and self.mode=='vad':
                        diagnostic_raw=list(diagnostic_ring)
                        raw_start_at=at-(len(diagnostic_raw)-1)*len(pcm)/mic.stats.sample_rate
                    if diagnostic_enabled:
                        segment_offset=max(0,sum(map(len,diagnostic_raw))-sum(map(len,segment.frames)))
                    self._emit('voice_detected',rev,audio_id=audio_id)
                if event in {'end','limit'}:
                    self._submit(audio,rev,audio_id,last_voice_at,
                        diagnostic_raw,raw_start_at,segment_offset,mic,
                        requested_pre_roll_ms,available_pre_roll_ms,ptt_press_offset_samples,
                        segment.last_metrics)
                    diagnostic_raw=[];raw_start_at=None;segment_offset=0
                    last_voice_at=None
                    if event=='limit':
                        self._emit('error',rev,message='Frase excede 30 segundos; usa pausas. Segmento limitado explicitamente.')
                    if armed:
                        segment=None;audio_id=None;detector=None
                    elif self.mode=='ptt': break
            if not self.stop_event.is_set() and not audio_id:
                self._emit('no_speech',rev)
        except Exception as exc:
            self._emit('error',rev,message=f'{type(exc).__name__}: {exc}')
        finally:
            closed=True
            if mic is not None:
                try:mic.close()
                except Exception as exc:
                    closed=False
                    self._emit('error',rev,message=f'Cierre de microfono: {type(exc).__name__}: {exc}')
            self._emit('capture_closed',rev,stats=asdict(mic.stats) if mic is not None else {},closed=closed)

    def _submit(self,audio,rev,audio_id,last_voice_at=None,raw_frames=None,
                raw_start_at=None,segment_offset=0,mic=None,
                requested_pre_roll_ms=0,available_pre_roll_ms=0,
                ptt_press_offset_samples=0,segment_metrics=None):
        if not audio_id:
            self._emit('no_speech',rev)
            return
        endpoint=time.monotonic();diagnostic=None
        with self.lock:
            if rev!=self.revision: return
            diagnostic_enabled=self.diagnostic_enabled
            reference=self.diagnostic_reference
        if diagnostic_enabled and self.diagnostic_store is not None and mic is not None and raw_frames:
            try:
                raw=np.concatenate(raw_frames) if raw_frames else audio.copy()
                segment=audio if audio is not None else np.empty(0,np.float32)
                diagnostic=self.diagnostic_store.capture(audio_id,raw,segment,reference,mic.stats,
                    self.mode,raw_start_at,segment_offset,last_voice_at,endpoint,
                    requested_pre_roll_ms,available_pre_roll_ms,ptt_press_offset_samples,
                    segment_metrics or {})
                self._emit('diagnostic_saved',rev,audio_id=audio_id,diagnostic=diagnostic)
            except Exception as exc:
                self._emit('error',rev,message=f'No se guardo audio RCA: {type(exc).__name__}: {exc}')
        if audio is None:
            self._emit('no_speech',rev,audio_id=audio_id,diagnostic=diagnostic)
            return
        with self.lock:
            if rev!=self.revision:return
            try: self.jobs.get_nowait();self.stale_discarded+=1
            except queue.Empty: pass
            self.jobs.put_nowait((audio,rev,audio_id,self.backend,endpoint,last_voice_at))
        self._emit('transcribing',rev,audio_id=audio_id,model=self.model_label,
            config=self.config_label,language=self.language_label,context=self.context_label)

    def _decode_loop(self):
        while self.alive:
            try: audio,rev,audio_id,backend,endpoint,last_voice_at=self.jobs.get(timeout=.2)
            except queue.Empty: continue
            try:
                if rev!=self.revision:
                    self.stale_discarded+=1;continue
                if self.recognizer is None or self.recognizer.name!=backend:
                    self.recognizer=None
                    self._emit('stt_loading',rev)
                    self.recognizer=self.recognizer_factory(backend)
                    self._emit('stt_ready',rev,load_ms=self.recognizer.load_ms)
                self._emit('transcription_started',rev,audio_id=audio_id,
                    model=self.model_label,config=self.config_label,
                    language=self.language_label,context=self.context_label,
                    decoder_state='fresh_call_no_previous_text')
                result=self.recognizer.transcribe(audio)
                text_ready_at=time.monotonic()
                if rev!=self.revision or not self.alive:
                    self.stale_discarded+=1
                    self._emit('stale_discarded',rev,audio_id=audio_id)
                    continue
                text=result.pop('text');confidence=result.pop('confidence')
                event=UserInputEvent('voice',text,datetime.now(timezone.utc).isoformat(),confidence,audio_id)
                if self.diagnostic_store is not None:
                    self.diagnostic_store.transcript(audio_id,{**result,'confidence':event.confidence},event.text)
                common=dict(input=asdict(event),
                    endpoint_to_text_ms=(text_ready_at-endpoint)*1000,
                    voice_end_to_text_ms=(text_ready_at-last_voice_at)*1000 if last_voice_at is not None else None,
                    last_voice_at=last_voice_at,endpoint_at=endpoint,text_ready_at=text_ready_at,
                    voice_end_clock_source='last_silero_voiced_block_callback_monotonic',**result)
                if result.get('hallucination',{}).get('suspected',False):
                    self._emit('hallucination_suspected',rev,**common)
                else:
                    self._emit('transcript',rev,**common)
            except Exception as exc:
                self._emit('error',rev,message=f'STT no disponible: {type(exc).__name__}: {exc}')
