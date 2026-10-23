"""Opt-in speech RCA artifacts. Audio is local and never enabled implicitly."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
import wave

import numpy as np


def audio_metrics(pcm,rate=16000):
    pcm=np.asarray(pcm,dtype=np.float32)
    if not len(pcm):
        return dict(samples=0,duration_s=0,rms=0,peak=0,clipping_percent=0)
    return dict(samples=len(pcm),duration_s=len(pcm)/rate,
        rms=float(np.sqrt(np.mean(np.square(pcm,dtype=np.float64)))),
        peak=float(np.max(np.abs(pcm))),
        clipping_percent=float(np.mean(np.abs(pcm)>=.999)*100))


def comparative_metrics(pcm,rate,noise_seconds=1.5):
    result=audio_metrics(pcm,rate)
    boundary=min(len(pcm),round(noise_seconds*rate))
    noise=np.asarray(pcm[:boundary],np.float32)
    voice=np.asarray(pcm[boundary:],np.float32)
    noise_rms=float(np.sqrt(np.mean(noise.astype(np.float64)**2))) if len(noise) else 0
    voice_rms=float(np.sqrt(np.mean(voice.astype(np.float64)**2))) if len(voice) else 0
    result.update(noise_window_s=boundary/rate,noise_rms=noise_rms,
        voice_window_rms=voice_rms,
        snr_estimate_db=20*np.log10(max(voice_rms,1e-12)/max(noise_rms,1e-12)))
    result['snr_note']='First 1.5s ambient vs remaining window; useful for relative device comparison, not calibrated SNR.'
    return result


def write_listening_wav(path,pcm,rate):
    encoded=(np.clip(pcm,-1,1)*32767).astype('<i2')
    with wave.open(str(path),'wb') as handle:
        handle.setparams((1,2,rate,0,'NONE','not compressed'))
        handle.writeframes(encoded.tobytes())


def atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


class SpeechRcaStore:
    def __init__(self,directory):
        self.root=Path(directory)/'speech-rca';self.root.mkdir(parents=True,exist_ok=True)
        self.records={};self.lock=threading.Lock()

    def capture(self,audio_id,raw,segment,reference,capture_stats,mode,
                raw_start_at,segment_start_offset_samples,last_voice_at,endpoint_at,
                requested_pre_roll_ms=0,available_pre_roll_ms=0,
                ptt_press_offset_samples=0,segment_metrics=None):
        started=time.perf_counter();folder=self.root/audio_id;folder.mkdir(exist_ok=False)
        raw=np.asarray(raw,np.float32);segment=np.asarray(segment,np.float32)
        np.save(folder/'raw_input_float32.npy',raw,allow_pickle=False)
        np.save(folder/'vad_segment_float32.npy',segment,allow_pickle=False)
        np.save(folder/'stt_segment_float32.npy',segment,allow_pickle=False)
        write_listening_wav(folder/'raw_input_listen.wav',raw,capture_stats.sample_rate)
        write_listening_wav(folder/'vad_segment_listen.wav',segment,capture_stats.sample_rate)
        write_listening_wav(folder/'stt_segment_listen.wav',segment,capture_stats.sample_rate)
        segment_end_offset=segment_start_offset_samples+len(segment)
        files=dict(raw_float32='raw_input_float32.npy',raw_listening='raw_input_listen.wav',
            vad_segment_float32='vad_segment_float32.npy',vad_segment_listening='vad_segment_listen.wav',
            segment_float32='stt_segment_float32.npy',segment_listening='stt_segment_listen.wav')
        pre_roll_variants={}
        if ptt_press_offset_samples>0:
            for milliseconds in (250,400,500):
                retained=round(milliseconds*capture_stats.sample_rate/1000)
                start=max(0,ptt_press_offset_samples-retained)
                variant=raw[start:]
                npy_name=f'preroll_{milliseconds}ms_float32.npy'
                wav_name=f'preroll_{milliseconds}ms_listen.wav'
                np.save(folder/npy_name,variant,allow_pickle=False)
                write_listening_wav(folder/wav_name,variant,capture_stats.sample_rate)
                files[f'preroll_{milliseconds}ms_float32']=npy_name
                files[f'preroll_{milliseconds}ms_listening']=wav_name
                pre_roll_variants[str(milliseconds)]=dict(start_offset_samples=start,
                    available_before_press_ms=ptt_press_offset_samples/capture_stats.sample_rate*1000,
                    metrics=audio_metrics(variant,capture_stats.sample_rate),
                    float32=npy_name,listening=wav_name)
        segment_metrics=dict(segment_metrics or {})
        segment_start_at=(raw_start_at+segment_start_offset_samples/capture_stats.sample_rate
            if raw_start_at is not None else None)
        if segment_start_at is not None:
            segment_metrics['voice_start_at']=segment_start_at+segment_metrics.get('silence_before_voice_s',0)
            segment_metrics['voice_end_at']=segment_metrics['voice_start_at']+segment_metrics.get('voice_span_duration_s',0)
        segment_metrics['last_voiced_callback_at']=last_voice_at
        segment_metrics['endpoint_at']=endpoint_at
        segment_metrics['callback_tail_s']=max(0,endpoint_at-last_voice_at) if last_voice_at is not None else None
        metadata=dict(schema='anydes.speech_rca.v2',audio_id=audio_id,utterance_id=audio_id,mode=mode,
            reference_text=reference,created_monotonic=time.monotonic(),
            capture=asdict(capture_stats),raw=audio_metrics(raw,capture_stats.sample_rate),
            stt_segment=audio_metrics(segment,capture_stats.sample_rate),
            vad=dict(backend='silero_v6_onnx_stateful',threshold=.6,rms_gate=.0006,
                confirmation_ms=80,pre_roll_ms=300,hangover_ms=800,min_voiced_ms=120,
                physical_pre_roll_requested_ms=requested_pre_roll_ms,
                physical_pre_roll_available_ms=available_pre_roll_ms,
                raw_start_at=raw_start_at,last_voice_at=last_voice_at,endpoint_at=endpoint_at,
                segment_start_offset_samples=segment_start_offset_samples,
                segment_start_offset_s=segment_start_offset_samples/capture_stats.sample_rate,
                trailing_raw_outside_segment_samples=max(0,len(raw)-segment_end_offset),
                trailing_raw_outside_segment_s=max(0,len(raw)-segment_end_offset)/capture_stats.sample_rate),
            endpointing=dict(strategy='silero_v6_then_trim',post_roll_target_ms=300,
                stt_input_transform='identity_from_vad_segment',metrics=segment_metrics),
            ptt_press_offset_samples=ptt_press_offset_samples,
            pre_roll_variants=pre_roll_variants,files=files,
            sha256={})
        for key,name in metadata['files'].items():
            metadata['sha256'][key]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        metadata['artifact_write_ms']=(time.perf_counter()-started)*1000
        atomic_json(folder/'metadata.json',metadata)
        with self.lock:self.records[audio_id]=folder
        return dict(folder=str(folder),metadata=str(folder/'metadata.json'),
            artifact_write_ms=metadata['artifact_write_ms'])

    def transcript(self,audio_id,result,text):
        with self.lock:folder=self.records.get(audio_id)
        if not folder:return
        payload=dict(result);payload.update(audio_id=audio_id,utterance_id=audio_id,text=text,
            created_monotonic=time.monotonic())
        atomic_json(folder/'transcription.json',payload)
