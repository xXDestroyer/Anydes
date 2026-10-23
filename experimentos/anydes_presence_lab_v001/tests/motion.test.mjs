import test from 'node:test';
import assert from 'node:assert/strict';
import { Motion, STATES, clamp, damp, speechEnvelope } from '../motion.mjs';

test('All five visual states transition gradually and stay bounded', () => {
  const m = new Motion();
  for (const state of Object.keys(STATES)) {
    const priorScale = m.scale;
    m.setState(state); m.update(0.016, 0.8);
    assert.ok(Math.abs(m.scale - priorScale) < 0.05);
    for (let i = 0; i < 200; i++) m.update(0.016, 0.8);
    assert.ok(Math.abs(m.scale - STATES[state].scale) < 0.001);
    assert.ok(Number.isFinite(m.rotation));
    assert.ok(m.energy >= 0 && m.energy <= 1);
  }
});
test('Speaking reacts to amplitude, other states do not pretend to speak', () => {
  const m = new Motion();
  m.setState('SPEAKING');
  for (let i = 0; i < 40; i++) m.update(0.016, 1);
  assert.ok(m.energy > 0.95);
  m.setState('LISTENING');
  for (let i = 0; i < 100; i++) m.update(0.016, 1);
  assert.ok(m.energy < 0.001);
});
test('Simulated speech has varied syllables and real phrase pauses', () => {
  const samples = Array.from({ length: 960 }, (_, i) => speechEnvelope(i / 100));
  assert.ok(samples.every(n => n >= 0 && n <= 1));
  assert.ok(samples.some(n => n > 0.8));
  assert.equal(speechEnvelope(3.5), 0);
  assert.equal(speechEnvelope(8.4), 0);
  assert.ok(new Set(samples.map(n => n.toFixed(2))).size > 60);
});
test('Invalid states fail without altering the current state', () => {
  const m = new Motion();
  assert.throws(() => m.setState('OFFICIAL_RUNTIME'), TypeError);
  assert.equal(m.state, 'IDLE');
});
test('Frame delta and signal inputs cannot produce a jump', () => {
  const m = new Motion(); m.setState('SPEAKING'); m.update(9, 50);
  assert.equal(m.time, 0.05);
  assert.ok(m.energy <= 1);
  assert.equal(clamp(-1), 0); assert.equal(clamp(5), 1);
  assert.ok(damp(0, 1, 3, 0.1) < 1);
});
