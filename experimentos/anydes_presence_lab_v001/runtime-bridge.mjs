export const RUNTIME_STATES = ['IDLE', 'LISTENING', 'THINKING', 'SPEAKING', 'WORKING'];
const SOURCES = ['runtime.idle', 'runtime.disconnected', 'voice.capture', 'voice.stt', 'brain.inference', 'voice.synthesis', 'b4.playback', 'tools.execution', 'tools.verification', 'tools.commit'];
const SOURCE_STATE = { 'runtime.idle': 'IDLE', 'runtime.disconnected': 'IDLE', 'voice.capture': 'LISTENING', 'voice.stt': 'LISTENING', 'brain.inference': 'THINKING', 'voice.synthesis': 'THINKING', 'b4.playback': 'SPEAKING', 'tools.execution': 'WORKING', 'tools.verification': 'WORKING', 'tools.commit': 'WORKING' };
const KEYS = ['schema', 'session', 'sequence', 'state', 'source', 'task_id', 'timestamp', 'connected'];
const MOTION_KEYS = ['schema','session','sequence','timestamp','working_mode','rms','peak','voice_active'];
export function validateMotion(frame, now=Date.now()) {
  return !!frame && typeof frame==='object' && Object.keys(frame).length===8 && Object.keys(frame).every(k=>MOTION_KEYS.includes(k))
    && frame.schema==='anydes.presence.motion.v1' && /^[a-f0-9]{16}$/.test(frame.session)
    && Number.isSafeInteger(frame.sequence) && frame.sequence>=0
    && ['NONE','PROGRAMMING','CREATING_FILE','EDITING','EXECUTING','VERIFYING','CONVERTING'].includes(frame.working_mode)
    && [frame.rms,frame.peak].every(x=>Number.isFinite(x)&&x>=0&&x<=1) && frame.rms<=frame.peak
    && typeof frame.voice_active==='boolean' && Math.abs(now-Date.parse(frame.timestamp))<3000;
}
export function validateFrame(frame, now = Date.now()) {
  if (!frame || typeof frame !== 'object' || Array.isArray(frame) || Object.keys(frame).length !== KEYS.length || Object.keys(frame).some(k => !KEYS.includes(k))) return false;
  if (frame.schema !== 'anydes.presence.v1' || typeof frame.session !== 'string' || !/^[a-f0-9]{16}$/.test(frame.session) || !Number.isSafeInteger(frame.sequence) || frame.sequence < 0) return false;
  if (!RUNTIME_STATES.includes(frame.state) || !SOURCES.includes(frame.source) || typeof frame.connected !== 'boolean') return false;
  if (SOURCE_STATE[frame.source] !== frame.state || (!frame.connected && frame.source !== 'runtime.disconnected')) return false;
  if (frame.task_id !== null && (typeof frame.task_id !== 'string' || !/^[a-f0-9]{16}$/.test(frame.task_id))) return false;
  if (!frame.connected && frame.state !== 'IDLE') return false;
  const time = typeof frame.timestamp === 'string' ? Date.parse(frame.timestamp) : NaN;
  return Number.isFinite(time) && Math.abs(now - time) < 6000;
}

export class RuntimeBridge {
  constructor({ onState, onMotion=()=>{}, sourceFactory = url => new EventSource(url), clock = () => Date.now(), timer = (callback, delay) => setInterval(callback, delay), clearTimer = handle => clearInterval(handle) }) {
    Object.assign(this, { onState, onMotion, sourceFactory, clock, timer, clearTimer });
    this.source = null; this.watch = null; this.sequence = -1; this.session = null;
    this.lastSeen = null; this.connected = false; this.lastFrame = null;
    this.generation = 0;
    this.motionSequence = -1;
  }
  neutral(reason) {
    this.connected = false; this.lastFrame = null;
    this.onState({ state: 'IDLE', connected: false, reason });
    this.onMotion(null);
  }
  connect() {
    this.close(); this.sequence = -1; this.session = null; this.lastSeen = null;
    const generation = ++this.generation;
    this.neutral('CONNECTING');
    this.source = this.sourceFactory('http://127.0.0.1:8781/events');
    this.motionSequence = -1;
    this.source.addEventListener?.('motion',message=>{
      if (generation!==this.generation || !this.connected) return;
      let frame;
      try { frame=JSON.parse(message.data); } catch { this.onMotion(null); return; }
      if (!validateMotion(frame,this.clock()) || frame.session!==this.session) { this.onMotion(null); return; }
      if (frame.sequence<=this.motionSequence) return;
      this.motionSequence=frame.sequence; this.onMotion(Object.freeze(frame));
    });
    this.source.onmessage = message => {
      if (generation !== this.generation) return;
      let frame;
      try { frame = JSON.parse(message.data); } catch { this.neutral('INVALID_PAYLOAD'); return; }
      if (!validateFrame(frame, this.clock())) { this.neutral('INVALID_PAYLOAD'); return; }
      if (frame.session === this.session && frame.sequence < this.sequence) return;
      this.session = frame.session; this.sequence = frame.sequence; this.lastSeen = this.clock();
      this.connected = frame.connected; this.lastFrame = Object.freeze(frame);
      this.onState(frame);
    };
    this.source.onerror = () => { if (generation === this.generation) this.neutral('DISCONNECTED'); };
    this.watch = this.timer(() => { if (generation === this.generation && this.lastSeen !== null && this.clock() - this.lastSeen > 4000) this.neutral('STALE_STREAM'); }, 500);
  }
  close() {
    this.generation++;
    this.source?.close(); this.source = null;
    if (this.watch !== null) this.clearTimer(this.watch);
    this.watch = null; this.neutral('DISCONNECTED');
  }
}
