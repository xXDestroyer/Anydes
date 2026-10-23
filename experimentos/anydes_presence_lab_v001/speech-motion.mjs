import { clamp, damp } from './motion.mjs';

export class SpeechMotionDriver {
  constructor() { this.energy = 0; this.body = 0; this.accent = 0; this.raw = 0; this.active = false; }
  update(dt, rms = 0, peak = 0, speaking = false) {
    dt = clamp(dt, 0, 0.05);
    const power = Number.isFinite(rms) ? Math.max(0,rms-0.005) : 0;
    this.raw = speaking ? clamp(power/0.16,0,0.88) : 0;
    this.active = speaking && power > 0.001;
    this.energy = damp(this.energy,this.raw,this.raw>this.energy ? 32 : 13,dt);
    this.body = damp(this.body,this.energy,5,dt);
    this.accent = damp(this.accent,speaking ? clamp((peak-0.025)/0.5,0,0.85) : 0,18,dt);
    return { core:clamp(this.energy*0.8+this.accent*0.2), petals:this.body*0.65+this.energy*0.2,
      rings:this.body*0.65+this.accent*0.15, particles:this.energy*0.3+this.accent*0.4 };
  }
}
