import test from 'node:test';
import assert from 'node:assert/strict';
import { CognitiveMotion } from '../motion.mjs';

test('Cognitive variants have orthogonal signatures, not a speed difference',()=>{
  const cases=[['THINKING','NONE','focus'],['WORKING','PROGRAMMING','build'],['WORKING','VERIFYING','check'],['WORKING','EXECUTING','outward']];
  for(const [state,mode,key] of cases){
    const c=new CognitiveMotion();for(let i=0;i<180;i++)c.update(.016,state,mode);
    assert.ok(c[key]>.99);
    for(const other of ['focus','build','check','outward'].filter(x=>x!==key))assert.equal(c[other],0);
  }
});
test('Transitions are smooth, bounded and do not change idle or speaking',()=>{
  const c=new CognitiveMotion();let previous=0;
  for(let i=0;i<200;i++){c.update(.016,'THINKING');assert.ok(Math.abs(c.focus-previous)<.08);previous=c.focus;assert.ok(Math.abs(c.irregular)<=1);}
  for(let i=0;i<200;i++)c.update(.016,'WORKING','PROGRAMMING');
  assert.ok(c.focus<.001);assert.ok(c.build>.99);assert.ok(Math.abs(c.assembly)<=.32);
  for(let i=0;i<250;i++)c.update(.016,'SPEAKING','PROGRAMMING');
  for(const key of ['focus','structure','build','check','outward'])assert.ok(c[key]<.001);
});
test('Programming holds/alignment, verification pulses and execution direction differ',()=>{
  const c=new CognitiveMotion(),angles=[],pulses=[];
  for(let i=0;i<240;i++){c.update(.016,'WORKING','PROGRAMMING');angles.push(c.assembly);pulses.push(c.verifyPulse);}
  assert.ok(angles.some((x,i)=>i>0&&Math.abs(x-angles[i-1])<1e-8));
  assert.ok(angles.some(x=>x>.2)&&angles.some(x=>x<-.2));
  assert.ok(pulses.some(x=>x>.9)&&pulses.some(x=>x===0));
});

test('Strong signatures differ in silhouette, direction and coordination without color',()=>{
  const thought=new CognitiveMotion(),build=new CognitiveMotion();
  for(let i=0;i<180;i++){thought.update(.016,'THINKING');build.update(.016,'WORKING','PROGRAMMING');}
  assert.ok(thought.radial<.81);assert.ok(build.radial>1.12);
  assert.ok(build.radial/thought.radial>1.35);
  assert.ok(thought.height<.91&&build.height>1.02);
  assert.ok(thought.direction<-.99&&build.direction>.99);
  assert.ok(thought.structure===0&&build.structure>.99);
  assert.ok(thought.fold<-.31&&build.fold>.29);
});

test('Thinking to construction releases within 300-700 ms without a cut',()=>{
  const c=new CognitiveMotion();for(let i=0;i<200;i++)c.update(.016,'THINKING');
  const start=c.radial;c.update(.016,'WORKING','PROGRAMMING');assert.ok(c.radial-start<.03);
  for(let i=0;i<20;i++)c.update(.016,'WORKING','PROGRAMMING');
  assert.ok(c.radial>1.05&&c.build>.79);
  for(let i=0;i<21;i++)c.update(.016,'WORKING','PROGRAMMING');
  assert.ok(c.radial>1.11&&c.focus<.05&&c.build>.95);
});
