import test from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeBridge, validateFrame } from '../runtime-bridge.mjs';
const timestamp = () => new Date().toISOString();
const frame = (state = 'IDLE', sequence = 1) => ({ schema: 'anydes.presence.v1', session: '1234567890abcdef', sequence, state, source: state === 'IDLE' ? 'runtime.idle' : 'b4.playback', task_id: null, timestamp: timestamp(), connected: true });
test('Only small typed frames are accepted, sensitive extras fail closed', () => {
  assert.ok(validateFrame(frame()));
  assert.equal(validateFrame({ ...frame(), prompt: 'private' }), false);
  assert.equal(validateFrame({ ...frame(), state: 'RUN_TOOL' }), false);
  assert.equal(validateFrame({ ...frame(), source: 'private response' }), false);
  assert.equal(validateFrame({ ...frame('SPEAKING'), source: 'tools.execution' }), false);
  assert.equal(validateFrame({ ...frame(), timestamp: '2001-01-01' }), false);
  assert.equal(validateFrame({ ...frame('SPEAKING'), connected: false }), false);
});
test('Disconnect and stale streams are neutral, reconnect uses actual snapshots', () => {
  let received, tick, now = Date.now(), closed = false;
  const fake = { close: () => { closed = true; } };
  const bridge = new RuntimeBridge({ onState: value => { received = value; }, sourceFactory: () => fake,
    clock: () => now, timer: callback => { tick = callback; return 1; }, clearTimer: () => {} });
  bridge.connect(); assert.equal(received.state, 'IDLE');
  fake.onmessage({ data: JSON.stringify(frame('SPEAKING', 2)) });
  assert.equal(received.state, 'SPEAKING'); assert.ok(bridge.connected);
  fake.onmessage({ data: JSON.stringify(frame('IDLE', 1)) });
  assert.equal(received.state, 'SPEAKING');
  fake.onerror(); assert.equal(received.state, 'IDLE'); assert.equal(bridge.connected, false);
  fake.onmessage({ data: JSON.stringify(frame('IDLE', 3)) });
  assert.ok(bridge.connected);
  now += 4100; tick(); assert.equal(received.state, 'IDLE'); assert.equal(bridge.connected, false);
  bridge.close(); assert.ok(closed);
});
test('Malformed payloads never become visual activity', () => {
  let received;
  const fake = { close() {} };
  const bridge = new RuntimeBridge({ onState: value => { received = value; }, sourceFactory: () => fake, timer: () => 1, clearTimer() {} });
  bridge.connect(); fake.onmessage({ data: 'not json' });
  assert.equal(received.connected, false); assert.equal(received.state, 'IDLE');
  bridge.close();
});
test('Late callbacks from a closed subscription cannot restore activity', () => {
  let received;
  const fake = { close() {} };
  const bridge = new RuntimeBridge({ onState: value => { received = value; }, sourceFactory: () => fake, timer: () => 1, clearTimer() {} });
  bridge.connect(); bridge.close();
  fake.onmessage({ data: JSON.stringify(frame('SPEAKING')) });
  assert.equal(received.state, 'IDLE'); assert.equal(bridge.connected, false);
});
