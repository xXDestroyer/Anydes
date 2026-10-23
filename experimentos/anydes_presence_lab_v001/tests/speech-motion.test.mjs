import test from 'node:test';
import assert from 'node:assert/strict';
import { SpeechMotionDriver } from '../speech-motion.mjs';
import { Motion, STATES, WORK_MODES } from '../motion.mjs';
import { validateMotion } from '../runtime-bridge.mjs';

test('Silence is quiet; soft and loud speech differ within clamps',()=>{
  const strengths=[0,.025,.25].map(rms=>{const s=new SpeechMotionDriver();for(let i=0;i<120;i++)s.update(.016,rms,rms*2,true);return s.energy;});
  assert.equal(strengths[0],0);assert.ok(strengths[1]>.05);assert.ok(strengths[1]<strengths[2]);assert.ok(strengths[2]<=.88);
});
test('Fast pulses are smoothed, channels differ, pause does not own state',()=>{
  const s=new SpeechMotionDriver();const m=new Motion();m.setState('SPEAKING');
  let prior=0,channels;
  for(let i=0;i<120;i++){channels=s.update(.016,i%2?.2:0,.4,true);assert.ok(Math.abs(s.energy-prior)<.4);prior=s.energy;}
  assert.notEqual(channels.core,channels.petals);const high=s.energy;
  for(let i=0;i<20;i++)s.update(.016,0,0,true);
  assert.ok(s.energy<high*.03);assert.equal(m.state,'SPEAKING');
  m.setState('IDLE');for(let i=0;i<50;i++)s.update(.016,1,1,false);assert.ok(s.energy<.001);
});
test('THINKING and WORKING have distinct signatures; variants remain WORKING',()=>{
  assert.notEqual(STATES.THINKING.opening,STATES.WORKING.opening);
  assert.notEqual(STATES.THINKING.complexity,STATES.WORKING.complexity);
  const signatures=[];
  for(const mode of ['PROGRAMMING','VERIFYING','EXECUTING']){const m=new Motion();m.setState('WORKING');m.setWorkingMode(mode);for(let i=0;i<150;i++)m.update(.016);signatures.push([m.coherence,m.flow,m.scan]);assert.equal(m.state,'WORKING');}
  assert.equal(new Set(signatures.map(JSON.stringify)).size,3);
  assert.throws(()=>new Motion().setWorkingMode('RUN_TOOL'));
  assert.equal(Object.keys(WORK_MODES).length,7);
});
test('Motion projection rejects extra fields, impossible energy and stale timestamps',()=>{
  const f={schema:'anydes.presence.motion.v1',session:'1234567890abcdef',sequence:1,timestamp:new Date().toISOString(),working_mode:'NONE',rms:.1,peak:.3,voice_active:true};
  assert.ok(validateMotion(f));assert.ok(!validateMotion({...f,pcm:'private'}));assert.ok(!validateMotion({...f,rms:2}));assert.ok(!validateMotion({...f,peak:.01}));assert.ok(!validateMotion({...f,timestamp:'2000-01-01'}));
});
