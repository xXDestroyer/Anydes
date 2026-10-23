"""Experimental microphone panel. Never claims listening before the stream opens."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time

from .environment import OUT,setup
setup()
from PySide6.QtCore import QObject,QEvent,QUrl,Qt,QTimer,Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication,QCheckBox,QComboBox,QDialog,QHBoxLayout,
    QLabel,QPlainTextEdit,QProgressBar,QPushButton,QVBoxLayout,QFormLayout,QStyle,QLineEdit)
from .capture import devices,probe_device
from .core import UserInputEvent,VoiceInput
from .telemetry import get_monitor


class EventBridge(QObject):
    received=Signal(object)


class VoicePanel(QDialog):
    def __init__(self,parent=None,on_input=None,on_barge=None,baseline_lab=False):
        super().__init__(parent)
        self.setWindowTitle('Anydes | Entrada de voz')
        self.setMinimumSize(520,690)
        self.resize(630,850)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose,False)
        self.on_input=on_input;self.on_barge=on_barge;self.shutting_down=False
        self.baseline_lab=bool(baseline_lab)
        self.bridge=EventBridge(self);self.bridge.received.connect(self.on_event)
        self.engine=VoiceInput(self.bridge.received.emit)
        self.last_audio_id=None;self.busy=False;self.ptt_down=False;self.ptt_armed=False
        self.state_code='mic_off';self.stream_state='Cerrado';self.have_devices=False
        self.state_history=[];self.devices_by_id={};self.underrun_count=0
        self.device_test_thread=None;self.device_test_cancel=threading.Event()
        self.rca_folder=None
        self.setStyleSheet('QDialog{background:#171b20;color:#ecedef} QLabel,QCheckBox{color:#ecedef} '
            'QComboBox,QPlainTextEdit{background:#242b31;color:#eef0f0;border:1px solid #49535a;padding:7px;} '
            'QPushButton{background:#344640;color:#ffffff;padding:9px;border:1px solid #597469;border-radius:6px;} '
            'QPushButton:pressed{background:#426558;} QProgressBar{background:#242b31;color:#eee;border:0;height:16px;} '
            'QProgressBar::chunk{background:#91c6ad;}')
        layout=QVBoxLayout(self)
        title=QLabel('Entrada de voz');title.setStyleSheet('font-size:20px;font-weight:600;');layout.addWidget(title)
        self.status=QLabel('mic_off | Microfono apagado');self.status.setObjectName('micStatus');layout.addWidget(self.status)
        self.history=QLabel('');self.history.setWordWrap(True)
        self.history.setStyleSheet('color:#aeb8c1;font-size:11px;');layout.addWidget(self.history)
        device_row=QHBoxLayout()
        self.device=QComboBox();self.device.setToolTip('Dispositivo de entrada')
        device_row.addWidget(self.device,1)
        self.refresh=QPushButton();self.refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh.setFixedSize(34,34);self.refresh.setToolTip('Actualizar dispositivos de entrada')
        self.refresh.setAccessibleName('Actualizar dispositivos de entrada')
        self.refresh.clicked.connect(self.reload_devices);device_row.addWidget(self.refresh);layout.addLayout(device_row)
        self.device_info=QLabel('');self.device_info.setWordWrap(True);layout.addWidget(self.device_info)
        self.device.currentIndexChanged.connect(self.show_selected_device)
        details=QFormLayout()
        self.format_info=QLabel('16000 Hz | mono | 320 frames / 20 ms (solicitado)')
        self.stream_info=QLabel('Cerrado | 0 muestras PCM')
        details.addRow('Formato PCM',self.format_info);details.addRow('Stream real',self.stream_info)
        layout.addLayout(details)
        self.diagnostic=QCheckBox('Guardar audio RCA local (raw + segmento STT)');layout.addWidget(self.diagnostic)
        if self.baseline_lab:
            self.diagnostic.setChecked(True)
            self.diagnostic.setToolTip('Obligatorio en la baseline RCA; guarda solo frases PTT explicitas.')
        self.reference=QLineEdit();self.reference.setPlaceholderText('Texto que realmente vas a decir (ground truth)')
        layout.addWidget(self.reference)
        self.device_test=QPushButton('Comparar este dispositivo: 1.5 s silencio + habla')
        self.device_test.clicked.connect(self.start_device_test);layout.addWidget(self.device_test)
        self.backend=QComboBox();self.backend.setToolTip('Reconocimiento local, CPU')
        self.backend.addItem('Whisper Base / CPU INT8','base')
        if not self.baseline_lab:
            self.backend.addItem('Vosk espanol / control actual','vosk')
        layout.addWidget(self.backend)
        self.decoder_info=QLabel('MODEL: Whisper Base INT8 | CONFIG: base_int8_es_beam3_clean_v1 | '
            'LANGUAGE: es | CONTEXT: none')
        self.decoder_info.setWordWrap(True);self.decoder_info.setStyleSheet('color:#bcd7ca;')
        layout.addWidget(self.decoder_info)
        self.headphones=QCheckBox('Auriculares conectados (necesarios para VAD)');layout.addWidget(self.headphones)
        self.headphones.toggled.connect(self._headphones_changed)
        pre_roll_row=QHBoxLayout()
        self.pre_roll=QComboBox();self.pre_roll.setToolTip('Audio real conservado antes de pulsar PTT')
        for value in (250,400,500):self.pre_roll.addItem(f'Pre-roll {value} ms',value)
        self.pre_roll.setCurrentIndex(self.pre_roll.findData(400));pre_roll_row.addWidget(self.pre_roll,1)
        self.prepare_ptt=QPushButton('Preparar microfono PTT')
        self.prepare_ptt.setToolTip('Abre el stream y mantiene hasta 500 ms solo en memoria')
        self.prepare_ptt.clicked.connect(self.arm_ptt);pre_roll_row.addWidget(self.prepare_ptt)
        layout.addLayout(pre_roll_row)
        meters=QFormLayout()
        self.meter=QProgressBar();self.meter.setRange(0,60);self.meter.setTextVisible(False)
        self.peak_meter=QProgressBar();self.peak_meter.setRange(0,60);self.peak_meter.setTextVisible(False)
        meters.addRow('RMS',self.meter);meters.addRow('Peak',self.peak_meter);layout.addLayout(meters)
        self.levels=QLabel('Sin muestras PCM');layout.addWidget(self.levels)
        row=QHBoxLayout()
        self.ptt=QPushButton('Mantener para hablar');self.ptt.setToolTip('Pulsar para hablar (F9 dentro de esta ventana)')
        self.ptt.pressed.connect(self.begin_ptt);self.ptt.released.connect(self.end_ptt);row.addWidget(self.ptt)
        self.vad=QPushButton('Escucha VAD');self.vad.setCheckable(True);self.vad.toggled.connect(self.toggle_vad);row.addWidget(self.vad)
        if self.baseline_lab:self.vad.setToolTip('Desactivado durante la baseline controlada PTT.')
        layout.addLayout(row)
        stop=QPushButton('Detener microfono');stop.clicked.connect(self.stop);layout.addWidget(stop)
        layout.addWidget(QLabel('Texto final reconocido'))
        self.transcript=QPlainTextEdit();self.transcript.setReadOnly(True);self.transcript.setMinimumHeight(90)
        self.transcript.setPlaceholderText('Sin transcripciones');layout.addWidget(self.transcript,1)
        self.metrics=QLabel('STT sin cargar');self.metrics.setWordWrap(True);layout.addWidget(self.metrics)
        timing=QFormLayout()
        self.voice_latency=QLabel('Sin medicion');self.voice_latency.setWordWrap(True)
        self.endpoint_latency=QLabel('Sin medicion')
        self.confidence=QLabel('No disponible en Whisper (no calibrada)');self.confidence.setWordWrap(True)
        timing.addRow('Fin de voz -> texto',self.voice_latency)
        timing.addRow('Cierre de captura -> texto',self.endpoint_latency)
        timing.addRow('Confianza STT',self.confidence);layout.addLayout(timing)
        self.ram=QLabel('');self.ram.setWordWrap(True);layout.addWidget(self.ram)
        self.underruns=QLabel('');self.underruns.setWordWrap(True);layout.addWidget(self.underruns)
        self.resource_path=QLabel('');self.resource_path.setWordWrap(True)
        self.resource_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.resource_path.setStyleSheet('color:#aeb8c1;font-size:11px;');layout.addWidget(self.resource_path)
        rca_actions=QHBoxLayout()
        self.open_rca=QPushButton('Abrir carpeta RCA');self.open_rca.clicked.connect(self.open_rca_folder)
        self.play_raw=QPushButton('Reproducir RAW');self.play_raw.clicked.connect(lambda:self.play_rca('raw_input_listen.wav'))
        self.play_segment=QPushButton('Reproducir segmento STT');self.play_segment.clicked.connect(
            lambda:self.play_rca('stt_segment_listen.wav'))
        for button in (self.open_rca,self.play_raw,self.play_segment):
            button.setEnabled(False);rca_actions.addWidget(button)
        layout.addLayout(rca_actions)
        self.error=QLabel('');self.error.setWordWrap(True);self.error.setStyleSheet('color:#f5b9ab;');layout.addWidget(self.error)
        self.timer=QTimer(self);self.timer.setInterval(250);self.timer.timeout.connect(self.refresh_controls);self.timer.start()
        QApplication.instance().installEventFilter(self)
        monitor=get_monitor()
        OUT.mkdir(parents=True,exist_ok=True)
        self.log_path=monitor.directory/'voice-events.jsonl' if monitor else OUT/f'session-{time.time_ns()}.jsonl'
        self.resource_path.setText('Registro local: '+str(self.log_path.parent))
        self.reload_devices()

    def configure_diagnostics(self):
        self.engine.configure_diagnostics(self.baseline_lab or self.diagnostic.isChecked(),
            self.reference.text(),self.log_path.parent)

    def rca_capture_active(self):
        return bool(self.engine.capture_thread and self.engine.capture_thread.is_alive())

    def open_rca_folder(self):
        if self.rca_folder and self.rca_folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.rca_folder)))

    def play_rca(self,name):
        if self.rca_capture_active():
            self.error.setText('Deten el microfono antes de reproducir audio RCA para no contaminar la captura.')
            return
        path=self.rca_folder/name if self.rca_folder else None
        if not path or not path.is_file():
            self.error.setText(f'Archivo RCA no disponible: {name}')
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def start_device_test(self):
        if self.device.currentData() is None:
            selected=next((d for d in self.devices_by_id.values() if d['default']),None)
            device=selected['index'] if selected else None
        else:device=self.device.currentData()
        if device is None:
            self.error.setText('Selecciona un dispositivo concreto para compararlo.');return
        if self.engine.capture_thread and self.engine.capture_thread.is_alive():
            self.error.setText('Deten el microfono antes de comparar el dispositivo.');return
        if self.device_test_thread and self.device_test_thread.is_alive():return
        self.error.setText('');self.set_state('device_test','Silencio 1.5 s; despues habla normal hasta completar 6 s')
        self.device_test_cancel.clear();self.device_test.setEnabled(False)
        monitor=get_monitor();root=(monitor.directory if monitor else OUT)/'device-tests';root.mkdir(parents=True,exist_ok=True)
        target=root/f'device-{device}-{time.time_ns()}'
        save_audio=self.diagnostic.isChecked();reference=self.reference.text().strip()
        def run():
            try:
                wav=target.with_suffix('.wav') if save_audio else None
                result=probe_device(device,6,wav,self.device_test_cancel)
                result['reference_text']=reference
                result['result_file']=str(target.with_suffix('.json'))
                Path(result['result_file']).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
            except Exception as exc:
                result=dict(error=f'{type(exc).__name__}: {exc}',metrics={},stats={},result_file=str(target.with_suffix('.json')))
            result['kind']='device_test_result';result['revision']=self.engine.revision
            self.bridge.received.emit(result)
        self.device_test_thread=threading.Thread(target=run,name='anydes-device-test',daemon=True)
        self.device_test_thread.start()

    def set_state(self,code,label):
        self.state_code=code;self.status.setText(f'{code} | {label}')
        if not self.state_history or self.state_history[-1]!=code:
            self.state_history.append(code);self.state_history=self.state_history[-5:]
            self.history.setText(' > '.join(self.state_history))

    def reload_devices(self):
        if self.engine.capture_thread and self.engine.capture_thread.is_alive():return
        previous=self.device.currentData()
        try:
            rows=devices()
            self.devices_by_id={d['index']:d for d in rows};self.have_devices=bool(rows)
            self.device.blockSignals(True);self.device.clear()
            default=next((d for d in rows if d['default']),None)
            self.device.addItem('Windows: '+(default['name'] if default else 'predeterminado no disponible'),None)
            for d in rows:self.device.addItem(f"{d['name']} | {d['hostapi_name']} | #{d['index']}",d['index'])
            self.device.setCurrentIndex(max(0,self.device.findData(previous)));self.device.blockSignals(False)
            if not rows:raise RuntimeError('No hay dispositivos de entrada disponibles.')
            self.error.setText('')
            if self.state_code=='error':self.set_state('mic_off','Microfono apagado')
            self.show_selected_device()
        except Exception as exc:
            self.device.blockSignals(False);self.have_devices=False
            self.error.setText(f'Enumeracion de microfonos: {exc}');self.set_state('error','Dispositivo no disponible')
        self.refresh_controls()

    def show_selected_device(self):
        index=self.device.currentData()
        selected=self.devices_by_id.get(index) if index is not None else next(
            (d for d in self.devices_by_id.values() if d['default']),None)
        self.device_info.setText('Seleccionado: '+(f"{selected['name']} ({selected['hostapi_name']})"
            if selected else 'predeterminado de Windows sin resolver'))
        self.format_info.setText(('Fuente reporta '+str(round(selected['default_samplerate']))+' Hz | '
            if selected else '')+'stream STT solicitado: 16000 Hz mono')

    def log(self,event):
        # Store metrics/IDs, not transcripts or raw PCM. No ambient audio on disk.
        row=dict(event)
        if event['kind']=='output_queue_underrun':
            self.underrun_count+=1
            source=event.get('source_at');monitor=get_monitor()
            stamp=f'T+{source-monitor.started:.3f} s' if source is not None and monitor else str(source)
            self.underruns.setText(f'Underruns de salida: {self.underrun_count} | ultimo: {stamp}')
        if 'input' in row:
            row['input']=dict(row['input']);row['input']['text_chars']=len(row['input'].pop('text',''))
        try:
            with self.log_path.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=True)+'\n')
        except OSError as exc:
            self.error.setText(f'No se pudo guardar telemetria: {exc}')

    def refresh_controls(self):
        active=bool(self.engine.capture_thread and self.engine.capture_thread.is_alive())
        testing=bool(self.device_test_thread and self.device_test_thread.is_alive())
        self.device.setEnabled(not active);self.refresh.setEnabled(not active)
        self.backend.setEnabled(not self.baseline_lab and not active and not self.busy)
        self.pre_roll.setEnabled(not self.baseline_lab and not active)
        self.prepare_ptt.setEnabled(self.have_devices and not testing and not active and not self.vad.isChecked())
        self.diagnostic.setEnabled(not self.baseline_lab and not active);self.reference.setEnabled(not active)
        self.ptt.setEnabled(self.have_devices and self.ptt_armed and not testing and not self.vad.isChecked())
        self.vad.setEnabled(not self.baseline_lab and self.have_devices and not testing and (not active or self.vad.isChecked()))
        self.device_test.setEnabled(self.have_devices and not active and not testing)
        monitor=get_monitor()
        if monitor:
            with monitor.lock:sample=monitor.latest
            memory=sample.get('memory',{})
            if memory:
                mib=1024**2
                self.ram.setText(f"RAM libre: {memory['system_available_bytes']/mib:.0f} MiB | "
                    f"Este proceso: {memory['process_rss_bytes']/mib:.0f} MiB | "
                    f"Pico: {monitor.peaks.get('process_rss_bytes',0)/mib:.0f} MiB")
            if monitor.error:self.error.setText(monitor.error)

    def _start(self,mode):
        self.error.setText('');self.set_state('opening','Abriendo microfono...')
        self.stream_state='Abriendo';self.stream_info.setText('Abriendo; esperando confirmacion del backend')
        try:
            self.configure_diagnostics()
            self.engine.start(self.device.currentData(),mode,self.backend.currentData())
        except Exception as exc:
            self.ptt_down=False;self.error.setText(str(exc));self.set_state('error','No se pudo iniciar')
            self.stream_state='Cerrado';self.stream_info.setText('Cerrado')

    def arm_ptt(self):
        if self.engine.capture_thread and self.engine.capture_thread.is_alive():return
        self.error.setText('');self.set_state('opening','Preparando microfono PTT...')
        self.stream_state='Abriendo';self.stream_info.setText('Abriendo; preparando pre-roll fisico')
        try:
            self.configure_diagnostics()
            self.engine.arm_ptt(self.device.currentData(),self.backend.currentData(),self.pre_roll.currentData())
        except Exception as exc:
            self.ptt_armed=False;self.error.setText(str(exc));self.set_state('error','No se pudo preparar')
            self.stream_state='Cerrado';self.stream_info.setText('Cerrado')

    def begin_ptt(self):
        if self.ptt_down:return
        if not self.ptt_armed:
            self.error.setText('Primero pulsa "Preparar microfono PTT" para disponer de pre-roll real.')
            return
        self.ptt_down=True
        # Deliberate PTT press can stop output before any self-echo is captured.
        if self.on_barge:self.on_barge()
        try:self.engine.begin_ptt(self.pre_roll.currentData())
        except Exception as exc:
            self.ptt_down=False;self.error.setText(str(exc));self.set_state('error','PTT no disponible')

    def end_ptt(self):
        if self.ptt_down:
            self.ptt_down=False;self.engine.finish()

    def toggle_vad(self,enabled):
        if enabled:
            if not self.headphones.isChecked():
                self.vad.blockSignals(True);self.vad.setChecked(False);self.vad.blockSignals(False)
                self.error.setText('VAD con altavoces pendiente de AEC validado. Usa pulsar para hablar o auriculares.')
                return
            self._start('vad')
        else:self.stop()

    def toggle_capture(self):
        self.show();self.raise_()
        if self.ptt_down:self.end_ptt()
        else:self.begin_ptt()

    def _headphones_changed(self,enabled):
        if not enabled and self.vad.isChecked():
            self.stop()
            self.error.setText('VAD detenido: sin auriculares falta validar cancelacion de eco.')

    def keyboard_superseded(self):
        self.busy=False
        if self.engine.mode=='ptt':self.stop()
        else:
            self.engine.invalidate()
            active=bool(self.engine.capture_thread and self.engine.capture_thread.is_alive())
            code='ptt_armed' if active and self.engine.mode=='armed_ptt' else ('listening' if active else 'mic_off')
            label='PTT preparado' if code=='ptt_armed' else ('Escuchando' if active else 'Microfono apagado')
            self.set_state(code,label)

    def on_event(self,event):
        self.log(event)
        if self.shutting_down:return
        if event['revision']!=self.engine.revision:
            return
        kind=event['kind']
        states={'mic_off':'Microfono apagado','mic_ready':'Microfono listo',
                'listening':'Escuchando','voice_detected':'Voz detectada',
                'ptt_armed':'PTT preparado',
                'transcribing':'Transcribiendo','error':'Error',
                'hallucination_suspected':'Alucinacion sospechada; no enviada',
                'no_speech':'No se detecto voz','stt_loading':'Cargando STT local'}
        if kind in states:self.set_state(kind,states[kind])
        if kind=='mic_ready':
            s=event['stats'];self.stream_state='Activo'
            self.device_info.setText('En uso: '+s['device_name'])
            self.format_info.setText(f"{s['sample_rate']} Hz | {s['channels']} canal | "
                f"{s['buffer_size']} frames / {s['buffer_size']/s['sample_rate']*1000:.0f} ms")
            self.stream_info.setText('Activo | esperando muestras PCM')
        if kind=='ptt_armed':
            self.ptt_armed=True
            self.stream_info.setText(f"Activo | pre-roll fisico hasta {event.get('pre_roll_ms',self.pre_roll.currentData())} ms")
        if kind=='voice_detected':
            self.last_audio_id=event['audio_id']
            if self.on_barge:self.on_barge()
        elif kind=='levels':
            s=event['stats'];db=20*math.log10(max(s['rms'],1e-9))
            self.meter.setValue(round(max(0,min(60,db+60))))
            peak_db=20*math.log10(max(s['peak'],1e-9))
            self.peak_meter.setValue(round(max(0,min(60,peak_db+60))))
            self.levels.setText(f"RMS {s['rms']:.5f} ({db:.1f} dBFS) | peak {s['peak']:.4f} ({peak_db:.1f} dBFS)")
            self.stream_info.setText(f"{self.stream_state} | {s['frames_received']} frames | "
                f"perdidos {s['dropped_frames']} | overflow {s['overflow_events']} | errores {s['stream_errors']}")
        elif kind in {'transcribing','stt_loading'}:self.busy=True
        elif kind=='stt_ready':self.metrics.setText(f"Whisper/STT cargado: {event['load_ms']:.0f} ms")
        elif kind in {'transcript','hallucination_suspected'}:
            self.busy=False
            item=UserInputEvent(**event['input'])
            self.decoder_info.setText(f"MODEL: {event.get('model','Whisper Base INT8')} | "
                f"CONFIG: {event.get('config','desconocida')} | "
                f"LANGUAGE: requested={event.get('language_requested','es')} / "
                f"backend={event.get('language_backend','?')} | CONTEXT: {event.get('context','?')} | "
                f"UTTERANCE_ID: {item.audio_id}")
            self.metrics.setText(f"{event.get('backend','STT')} | STT {event['stt_ms']:.0f} ms | "
                f"RTF {event['rtf']:.2f} | segmentos Whisper {event.get('whisper_segment_count',0)}")
            latency=event.get('voice_end_to_text_ms')
            self.voice_latency.setText(f'{latency:.0f} ms (fin detectado por VAD)' if latency is not None else 'No disponible')
            self.endpoint_latency.setText(f"{event['endpoint_to_text_ms']:.0f} ms")
            self.confidence.setText('No disponible (Whisper; no calibrada)' if item.confidence is None
                else f'{item.confidence:.3f} (score del backend; no calibrado)')
            if item.text.strip():
                label='HALLUCINATION_SUSPECTED' if kind=='hallucination_suspected' else 'TRANSCRIPT'
                self.transcript.appendPlainText(
                    f"MODEL: {event.get('model','Whisper Base INT8')}\n"
                    f"CONFIG: {event.get('config','desconocida')}\n"
                    f"LANGUAGE: {event.get('language_requested','es')}\n"
                    f"CONTEXT: {event.get('context','?')}\n"
                    f"UTTERANCE_ID: {item.audio_id}\n{label}: {item.text}\n")
                if kind=='hallucination_suspected':
                    reasons=', '.join(event.get('hallucination',{}).get('reasons',[]))
                    self.error.setText(f'Resultado bloqueado y guardado para RCA: {reasons}')
                elif self.on_input:self.on_input(item)
            else:self.error.setText('No hubo una transcripcion util; revisa dispositivo y nivel.')
            active=self.stream_state=='Activo'
            if kind=='hallucination_suspected':
                self.set_state('hallucination_suspected','Alucinacion sospechada; no enviada')
            elif active and self.engine.mode=='armed_ptt':self.set_state('ptt_armed','PTT preparado')
            else:self.set_state('listening' if active else 'mic_off','Escuchando' if active else 'Microfono apagado')
        elif kind=='error':
            self.busy=False;self.error.setText(event['message'])
        elif kind=='capture_closed':
            self.ptt_armed=False
            self.stream_state='Cerrado' if event.get('closed',True) else 'Error de cierre'
            self.stream_info.setText(self.stream_state+' | captura finalizada')
            self.meter.setValue(0);self.peak_meter.setValue(0)
            if not self.busy and not self.error.text():self.set_state('mic_off','Microfono apagado')
        elif kind=='diagnostic_saved':
            self.rca_folder=Path(event['diagnostic']['folder'])
            self.resource_path.setText('Audio RCA local: '+str(self.rca_folder))
            for button in (self.open_rca,self.play_raw,self.play_segment):button.setEnabled(True)
        elif kind=='device_test_result':
            self.device_test.setEnabled(True);m=event['metrics'];s=event['stats']
            if event.get('error'):
                self.error.setText(event['error']);self.set_state('error','Fallo de prueba de dispositivo')
            else:
                self.set_state('mic_off','Prueba de dispositivo terminada')
                self.levels.setText(f"RMS {m['rms']:.5f} | peak {m['peak']:.4f} | clipping {m['clipping_percent']:.3f}%")
                self.metrics.setText(f"SNR aproximado {m['snr_estimate_db']:.1f} dB | "
                    f"{s['sample_rate']} Hz nativo | perdidos {s['dropped_frames']} | overflow {s['overflow_events']}")
                self.resource_path.setText('Prueba de dispositivo: '+event['result_file'])

    def stop(self):
        self.ptt_down=False;self.ptt_armed=False;self.busy=False;self.engine.stop();self.meter.setValue(0);self.peak_meter.setValue(0)
        active=bool(self.engine.capture_thread and self.engine.capture_thread.is_alive())
        if not active:
            self.stream_state='Cerrado';self.stream_info.setText('Cerrado')
        self.vad.blockSignals(True);self.vad.setChecked(False);self.vad.blockSignals(False)

    def shutdown(self):
        self.shutting_down=True;self.timer.stop()
        self.device_test_cancel.set()
        if self.device_test_thread:self.device_test_thread.join(timeout=3)
        QApplication.instance().removeEventFilter(self)
        clean=self.engine.close()
        self.log(dict(kind='shutdown',clean=clean,stale_discarded=self.engine.stale_discarded))
        return clean

    def closeEvent(self,event):
        self.stop()
        super().closeEvent(event)

    def eventFilter(self,obj,event):
        if self.isActiveWindow() and event.type() in {QEvent.Type.KeyPress,QEvent.Type.KeyRelease} and event.key()==Qt.Key.Key_F9:
            if not event.isAutoRepeat():
                self.begin_ptt() if event.type()==QEvent.Type.KeyPress else self.end_ptt()
            return True
        if event.type()==QEvent.Type.ApplicationDeactivate and self.ptt_down:self.end_ptt()
        return False
