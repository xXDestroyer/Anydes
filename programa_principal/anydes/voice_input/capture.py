"""Bounded, observable PCM capture. No recording or network access by default."""
from __future__ import annotations

import argparse
import hashlib
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd

PROGRAM = Path(__file__).resolve().parents[2]
OUT = PROGRAM / 'datos' / 'telemetria' / 'voice_input'


def permission_state():
    import winreg
    result = {}
    key = r'Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone'
    for name, hive, suffix in [('machine',winreg.HKEY_LOCAL_MACHINE,''),
                               ('user',winreg.HKEY_CURRENT_USER,''),
                               ('desktop',winreg.HKEY_CURRENT_USER,r'\NonPackaged')]:
        try:
            with winreg.OpenKey(hive,key+suffix) as handle:
                result[name] = winreg.QueryValueEx(handle,'Value')[0]
        except OSError:
            result[name] = 'unknown'
    return result


def devices():
    apis=sd.query_hostapis()
    return [dict(d,hostapi_name=apis[d['hostapi']]['name'],
                 default=d['index']==sd.default.device[0])
            for d in sd.query_devices() if d['max_input_channels'] > 0]


@dataclass
class CaptureStats:
    device_index: int | None = None
    device_name: str = ''
    hostapi_name: str = ''
    device_default_sample_rate: float = 0
    device_max_input_channels: int = 0
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 320
    rms: float = 0
    peak: float = 0
    frames_received: int = 0
    dropped_frames: int = 0
    overflow_events: int = 0
    stream_errors: int = 0
    callback_count: int = 0
    last_callback_at: float = 0
    stream_latency_ms: float = 0


class Microphone:
    def __init__(self, device=None, sample_rate=16000, block_ms=20):
        self.device=device
        self.stats=CaptureStats(sample_rate=sample_rate,buffer_size=round(sample_rate*block_ms/1000))
        self.queue=queue.Queue(maxsize=100)
        self.stream=None
        self.running=False
        self.fault=None
        self.opened_at=0.0

    def start(self):
        if self.running:
            raise RuntimeError('El microfono ya esta abierto.')
        if 'Deny' in permission_state().values():
            raise PermissionError('Windows bloquea el microfono. Revisa Privacidad > Microfono > aplicaciones de escritorio.')
        selected=sd.query_devices(self.device,'input')
        apis=sd.query_hostapis()
        resolved=int(self.device if self.device is not None else sd.default.device[0])
        self.stats.device_index=resolved
        self.stats.device_name=selected['name']
        self.stats.hostapi_name=apis[selected['hostapi']]['name']
        self.stats.device_default_sample_rate=float(selected['default_samplerate'])
        self.stats.device_max_input_channels=int(selected['max_input_channels'])
        try:
            sd.check_input_settings(device=self.device,channels=1,dtype='float32',samplerate=self.stats.sample_rate)
            self.stream=sd.InputStream(device=self.device,samplerate=self.stats.sample_rate,
                channels=1,dtype='float32',blocksize=self.stats.buffer_size,latency='low',callback=self._callback)
            self.opened_at=time.monotonic()
            self.stream.start()
            self.stats.sample_rate=round(float(self.stream.samplerate))
            self.stats.stream_latency_ms=float(self.stream.latency)*1000
            self.running=True
        except Exception:
            self.stats.stream_errors+=1
            self.close()
            raise

    def _callback(self, data, frames, clock, status):
        # Never run VAD/STT or emit UI calls on PortAudio's callback thread.
        try:
            pcm=data[:,0].copy()
            self.stats.callback_count+=1
            self.stats.frames_received+=frames
            self.stats.last_callback_at=time.monotonic()
            self.stats.rms=float(np.sqrt(np.mean(pcm*pcm)))
            self.stats.peak=float(np.max(np.abs(pcm)))
            if status:
                self.stats.stream_errors+=1
                self.stats.overflow_events+=int(bool(status.input_overflow))
            try:
                self.queue.put_nowait((pcm,self.stats.last_callback_at))
            except queue.Full:
                self.stats.dropped_frames+=frames
                self.fault='capture_queue_overflow'
        except Exception as exc:
            self.fault=f'capture_callback_failed: {exc}'
            self.stats.stream_errors+=1

    def read(self, timeout=.25):
        if self.fault:
            raise RuntimeError(self.fault)
        if not self.stream or not self.stream.active:
            raise RuntimeError('El stream del microfono se detuvo.')
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            if time.monotonic()-max(self.stats.last_callback_at,self.opened_at)>2:
                raise RuntimeError('El microfono no entrega muestras PCM.')
            return None

    def close(self):
        self.running=False
        if self.stream is not None:
            stream,self.stream=self.stream,None
            try:
                stream.abort()
            finally:
                stream.close()


def probe(device=None,seconds=8,wav_path=None):
    mic=Microphone(device)
    started=time.monotonic()
    levels=[]; chunks=[]; error=None
    try:
        mic.start()
        while time.monotonic()-started<seconds:
            packet=mic.read()
            if packet:
                pcm,at=packet
                levels.append(dict(at_s=at-started,rms=float(np.sqrt(np.mean(pcm*pcm))),peak=float(np.abs(pcm).max())))
                if wav_path:
                    chunks.append(pcm)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        mic.close()
    if wav_path and chunks:
        with wave.open(str(wav_path),'wb') as handle:
            handle.setparams((1,2,16000,0,'NONE','not compressed'))
            handle.writeframes((np.clip(np.concatenate(chunks),-1,1)*32767).astype('<i2').tobytes())
    return dict(stats=asdict(mic.stats),permissions=permission_state(),error=error,
                elapsed_s=time.monotonic()-started,levels=levels,closed=mic.stream is None,
                overflow_lost_frame_count='unknown' if mic.stats.overflow_events else 0,
                audio_saved=bool(wav_path and chunks),
                note='RMS/peak are levels, not proof of intelligible speech. Driver overflow loss is not measurable from this API.')


def probe_device(device,seconds=6,wav_path=None,cancel_event=None):
    """Relative device check: 1.5s ambient, then natural speech. Native reported rate."""
    selected=sd.query_devices(device,'input');rate=round(float(selected['default_samplerate']))
    mic=Microphone(device,sample_rate=rate);chunks=[];error=None;started=time.monotonic()
    try:
        mic.start()
        while time.monotonic()-started<seconds and not (cancel_event and cancel_event.is_set()):
            packet=mic.read()
            if packet:chunks.append(packet[0])
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        mic.close()
    pcm=np.concatenate(chunks) if chunks else np.empty(0,np.float32)
    files={}
    if wav_path and len(pcm):
        wav_path=Path(wav_path);wav_path.parent.mkdir(parents=True,exist_ok=True)
        exact=wav_path.with_suffix('.float32.npy');np.save(exact,pcm,allow_pickle=False)
        with wave.open(str(wav_path),'wb') as handle:
            handle.setparams((1,2,rate,0,'NONE','not compressed'))
            handle.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
        files={'float32_npy':str(exact),'listening_wav':str(wav_path),
            'float32_sha256':hashlib.sha256(exact.read_bytes()).hexdigest(),
            'wav_sha256':hashlib.sha256(wav_path.read_bytes()).hexdigest()}
    from .diagnostics import comparative_metrics
    return dict(stats=asdict(mic.stats),metrics=comparative_metrics(pcm,rate),
        elapsed_s=time.monotonic()-started,error=error,cancelled=bool(cancel_event and cancel_event.is_set()),
        audio_saved=bool(wav_path and len(pcm)),files=files)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--device',type=int)
    parser.add_argument('--seconds',type=float,default=8)
    parser.add_argument('--wav',type=Path)
    parser.add_argument('--output',type=Path,default=OUT/'capture.json')
    args=parser.parse_args()
    result=dict(devices=devices(),default=list(sd.default.device),permissions=permission_state()) if args.list else probe(args.device,args.seconds,args.wav)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=True))
    print(json.dumps({k:v for k,v in result.items() if k!='levels'},indent=2))
    return int(bool(result.get('error')))


if __name__=='__main__':
    raise SystemExit(main())
