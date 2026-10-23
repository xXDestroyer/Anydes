export const STATES = Object.freeze({
  IDLE: { opening: 0.10, scale: 1, spin: 0.065, light: 0.76, complexity: 0.1 },
  LISTENING: { opening: -0.10, scale: 0.91, spin: 0.025, light: 0.92, complexity: 0.12 },
  THINKING: { opening: -0.08, scale: 0.97, spin: 0.20, light: 0.93, complexity: 0.95 },
  SPEAKING: { opening: 0.14, scale: 1, spin: 0.085, light: 0.93, complexity: 0.32 },
  WORKING: { opening: 0.03, scale: 0.97, spin: 0.16, light: 0.88, complexity: 0.25 },
});
export const WORK_MODES = Object.freeze({
  NONE:{ coherence:0.5, flow:0.2, scan:0.3 }, PROGRAMMING:{ coherence:1, flow:0.35, scan:0.8 },
  CREATING_FILE:{ coherence:0.7, flow:0.65, scan:0.35 }, EDITING:{ coherence:0.65, flow:-0.45, scan:0.5 },
  EXECUTING:{ coherence:0.85, flow:1, scan:0.2 }, VERIFYING:{ coherence:0.95, flow:-0.65, scan:0.95 },
  CONVERTING:{ coherence:0.6, flow:0.5, scan:0.65 },
});
export const clamp = (n, min = 0, max = 1) => Math.max(min, Math.min(max, n));
export const damp = (a, b, rate, dt) => a + (b - a) * (1 - Math.exp(-rate * dt));

// Presentation-only weights. Audio and canonical runtime state remain independent.
export class CognitiveMotion {
  constructor() {
    this.focus=this.structure=this.build=this.check=this.outward=0;
    this.phase=0; this.mode='NONE';
    this.radial=this.height=1;this.fold=this.direction=this.assembly=this.irregular=this.verifyPulse=this.converge=0;
  }
  update(dt,state,mode='NONE') {
    dt=clamp(dt,0,0.05);
    const work=state==='WORKING';
    const key=work?mode:'NONE';
    if(key!==this.mode){this.phase=0;this.mode=key;}
    this.phase+=dt;
    const targets={focus:state==='THINKING'?1:0,structure:work?1:0,
      build:work&&mode==='PROGRAMMING'?1:0,check:work&&mode==='VERIFYING'?1:0,
      outward:work&&mode==='EXECUTING'?1:0};
    for(const k of Object.keys(targets))this[k]=damp(this[k],targets[k],4.8,dt);
    const q=this.phase/0.95,step=Math.floor(q),fraction=q-step;
    const ease=clamp((fraction-0.55)/0.45);
    const s=ease*ease*(3-2*ease),positions=[-1,0,1,0];
    this.assembly=(positions[step%4]*(1-s)+positions[(step+1)%4]*s)*0.32;
    this.irregular=(Math.sin(this.phase*2.3)+Math.sin(this.phase*3.71+0.7)*0.55)/1.55;
    this.verifyPulse=Math.pow(Math.max(0,Math.sin(this.phase*2.6)),8);
    this.converge=(1-Math.cos(this.phase*2.6))*0.5;
    this.radial=1-this.focus*0.20+this.structure*0.07+this.build*0.06;
    this.height=1-this.focus*0.10+this.build*0.025;
    this.fold=-this.focus*0.32+this.build*0.30;
    this.direction=-this.focus+this.structure*0.5+this.build*0.5;
    return this;
  }
}

// Syllabic envelopes with phrase pauses, not a single periodic breathing wave.
export function speechEnvelope(t) {
  const phrase = t % 9.6;
  if (phrase > 7.9 || (phrase > 3.2 && phrase < 3.85)) return 0;
  const syllable = Math.pow(Math.max(0, Math.sin(phrase * 15.7 + Math.sin(phrase * 2.8))), 0.65);
  const cadence = 0.45 + 0.55 * Math.pow(Math.sin(phrase * 3.17 + 0.4), 2);
  return clamp(syllable * cadence);
}

export class Motion {
  constructor() {
    this.state = 'IDLE';
    Object.assign(this, STATES.IDLE);
    this.energy = 0;
    this.time = 0;
    this.rotation = 0;
    this.workingMode = 'NONE';
    this.coherence = this.flow = this.scan = 0;
  }
  setWorkingMode(mode) {
    if (!Object.hasOwn(WORK_MODES,mode)) throw new TypeError('Unknown work mode');
    this.workingMode = mode;
  }
  setState(state) {
    if (!Object.hasOwn(STATES, state)) throw new TypeError('Unknown visual state');
    this.state = state;
  }
  update(dt, level = 0, speed = 1) {
    dt = clamp(dt, 0, 0.05);
    this.time += dt * speed;
    const target = STATES[this.state];
    for (const key of Object.keys(target)) this[key] = damp(this[key], target[key], 3.1, dt);
    for (const key of ['coherence','flow','scan']) this[key] = damp(this[key],this.state==='WORKING' ? WORK_MODES[this.workingMode][key] : 0,6,dt);
    const audio = this.state === 'SPEAKING' ? clamp(level) : 0;
    this.energy = damp(this.energy, audio, audio > this.energy ? 19 : 6.5, dt);
    this.rotation += dt * this.spin * speed;
    return this;
  }
}
