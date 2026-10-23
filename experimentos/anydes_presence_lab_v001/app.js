import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Motion, CognitiveMotion, STATES, speechEnvelope, clamp } from './motion.mjs';
import { RuntimeBridge } from './runtime-bridge.mjs';
import { SpeechMotionDriver } from './speech-motion.mjs';

const $ = id => document.getElementById(id);
const blendAngle=(a,b,w)=>a+Math.atan2(Math.sin(b-a),Math.cos(b-a))*w;
const names = { IDLE: 'Reposo', LISTENING: 'Escuchando', THINKING: 'Pensando', SPEAKING: 'Hablando', WORKING: 'Trabajando' };
function icons() {
  document.querySelectorAll('[data-icon]').forEach(el => {
    const name = el.dataset.icon;
    if (el.tagName === 'BUTTON') el.innerHTML = `<i data-lucide="${name}"></i>`;
    else el.setAttribute('data-lucide', name);
    delete el.dataset.icon;
  });
  window.lucide.createIcons();
}
function setIcon(button, name) { button.innerHTML = `<i data-lucide="${name}"></i>`; window.lucide.createIcons(); }
icons();

const motion = new Motion();
const cognition = new CognitiveMotion();
const stage = $('stage');
const canvas = $('scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#09090f');
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 60);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.96;
renderer.info.autoReset = false;
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1024, 768), 0.5, 0.38, 0.7);
composer.addPass(bloom);
composer.addPass(new OutputPass());
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.enableZoom = true;
controls.minDistance = 5.8;
controls.maxDistance = 15;
controls.minPolarAngle = Math.PI * 0.22;
controls.maxPolarAngle = Math.PI * 0.78;
let homeDistance = 8.7;
function resetView() {
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.update();
  camera.position.set(0, 0.5, homeDistance);
  controls.target.set(0, 0, 0); controls.update();
  controls.enableDamping = damping;
}
const observer = new ResizeObserver(() => {
  const w = stage.clientWidth, h = stage.clientHeight;
  camera.aspect = w / h;
  homeDistance = w / h < 1 ? 11.8 : 8.7;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  resetView();
});
observer.observe(stage);

const body = new THREE.Group();
scene.add(body);
const shaderUniforms = { uTime: { value: 0 }, uEnergy: { value: 0 }, uLight: { value: 0.75 }, uFocus:{value:0} };
const petalVertex = `
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vUv = uv;
  vec4 p = modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-p.xyz);
  gl_Position = projectionMatrix * p;
}`;
const petalFragment = `
uniform float uTime;
uniform float uEnergy;
uniform float uLight;
uniform vec3 uColor;
uniform vec3 uTip;
uniform float uFocus;
uniform float uInner;
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  float edge = pow(abs(vUv.x * 2.0 - 1.0), 19.0);
  float spine = pow(max(0.0, 1.0 - abs(vUv.x * 2.0 - 1.0)), 38.0);
  float fresnel = pow(1.0 - abs(dot(normalize(vNormal), vView)), 2.4);
  float vein = pow(0.5 + 0.5 * sin(vUv.y * 82.0 + vUv.x * 28.0), 20.0);
  float flow = pow(0.5 + 0.5 * sin(vUv.y * 11.0 - uTime * 1.8), 9.0);
  float taper = smoothstep(0.0, 0.12, vUv.y) * smoothstep(1.0, 0.88, vUv.y);
  vec3 color = mix(uColor, uTip, smoothstep(0.3, 1.0, vUv.y));
  color = mix(color, vec3(1.0,0.72,0.91), edge * 0.24);
  float innerAccent = (edge * (1.0-smoothstep(0.15,0.62,vUv.y))*0.78 + spine*0.30)*uInner;
  color = mix(color, vec3(1.0,0.19,0.47), uFocus*innerAccent);
  float intensity = 0.44 + edge * 1.45 + spine * 0.15 + fresnel * 0.48 + vein * 0.06 + flow * (0.07 + uEnergy * 0.2);
  float alpha = (0.16 + fresnel * 0.29 + edge * 0.63 + spine * 0.09) * taper;
  gl_FragColor = vec4(color * intensity * (0.75 + uLight * 0.8 + uEnergy * 0.3), alpha);
}`;

function petalPoint(t, s, twist, width, layer) {
  const curve = Math.sin(Math.PI * t);
  const r = 0.16 + curve * (1.25 - layer * 0.06) + 0.7 * t;
  const angle = twist * t + s * width * Math.pow(curve, 0.72);
  const dome = 0.11 * (1 - s * s) * curve;
  return new THREE.Vector3((r + dome) * Math.cos(angle), -1.38 + 2.78 * t + 0.09 * s * s * curve, (r + dome) * Math.sin(angle));
}

function petalGeometry(twist, width, layer) {
  const positions = [], uv = [], indices = [];
  const rows = 64, cols = 20;
  for (let y = 0; y <= rows; y++) for (let x = 0; x <= cols; x++) {
    const p = petalPoint(y / rows, x / cols * 2 - 1, twist, width, layer);
    positions.push(p.x, p.y, p.z); uv.push(x / cols, y / rows);
    if (y < rows && x < cols) {
      const a = y * (cols + 1) + x, b = a + cols + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(indices); g.computeVertexNormals();
  return g;
}

const layers = [];
const petals = [];
for (let layer = 0; layer < 3; layer++) {
  const group = new THREE.Group();
  body.add(group); layers.push(group);
  const count = [7, 6, 5][layer], twist = [1.25, -1.05, 1.4][layer];
  const geometry = petalGeometry(twist, [0.39, 0.46, 0.5][layer], layer);
  for (let i = 0; i < count; i++) {
    const pivot = new THREE.Group();
    const angle = i / count * Math.PI * 2 + layer * 0.65;
    pivot.rotation.y = angle;
    const material = new THREE.ShaderMaterial({
      uniforms: { ...shaderUniforms, uInner:{value:[0.1,0.6,1][layer]}, uColor: { value: new THREE.Color(layer === 1 ? '#c746b0' : '#9346ef') }, uTip: { value: new THREE.Color(i % 3 === 0 ? '#f5a8b7' : '#d878db') } },
      vertexShader: petalVertex, fragmentShader: petalFragment,
      side: THREE.DoubleSide, transparent: true, depthWrite: false, blending: THREE.NormalBlending,
    });
    const mesh = new THREE.Mesh(geometry, material);
    pivot.add(mesh); group.add(pivot);
    for (const s of [-1, 1]) {
      const points = Array.from({ length: 80 }, (_, j) => petalPoint(j / 79, s, twist, [0.39, 0.46, 0.5][layer], layer));
      const edge = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: i % 2 ? '#ce85e4' : '#e2a0c8', transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false }));
      pivot.add(edge);
    }
    petals.push({ pivot, layer, angle, phase: i * 1.7 + layer, baseScale: [1, 0.8, 0.61][layer] });
    pivot.scale.setScalar([1, 0.8, 0.61][layer]);
  }
}

const core = new THREE.Group();
body.add(core);
const pearl = new THREE.Mesh(new THREE.IcosahedronGeometry(0.115, 4), new THREE.MeshBasicMaterial({ color: new THREE.Color(2.5, 1.2, 1.8) }));
core.add(pearl);
const diamond = new THREE.Mesh(new THREE.OctahedronGeometry(0.47), new THREE.MeshBasicMaterial({ color: '#f2badb', wireframe: true, transparent: true, opacity: 0.58, blending: THREE.AdditiveBlending, depthWrite: false }));
core.add(diamond);
const pearlBase=pearl.material.color.clone(),diamondBase=diamond.material.color.clone();
const pearlWarm=new THREE.Color(2.2,0.32,0.95),diamondWarm=new THREE.Color('#f478b3');
const coronaMaterial = new THREE.ShaderMaterial({
  uniforms: { uEnergy: shaderUniforms.uEnergy },
  vertexShader: `varying vec3 n; varying vec3 v; void main(){ vec4 p=modelViewMatrix*vec4(position,1.); n=normalize(normalMatrix*normal); v=normalize(-p.xyz);gl_Position=projectionMatrix*p;}`,
  fragmentShader: `varying vec3 n; varying vec3 v; uniform float uEnergy; void main(){float f=pow(1.-abs(dot(normalize(n),v)),3.);gl_FragColor=vec4(vec3(1.4,.35,.78)*(0.5+uEnergy),f*.2);}`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
core.add(new THREE.Mesh(new THREE.SphereGeometry(0.37, 32, 24), coronaMaterial));

const rings = [];
for (let i = 0; i < 4; i++) {
  const g = new THREE.Group(); body.add(g);
  const points = Array.from({ length: 241 }, (_, j) => {
    const a = j / 240 * Math.PI * 2;
    return new THREE.Vector3(Math.cos(a) * (1.87 + i * 0.075), 0, Math.sin(a) * (1.87 + i * 0.075));
  });
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: ['#e296c5', '#c29ae7', '#c692e2', '#eab8d2'][i], transparent: true, opacity: i === 3 ? 0.11 : 0.27, blending: THREE.AdditiveBlending, depthWrite: false }));
  g.add(line);
  for (let j = 0; j < 3; j++) {
    const node = new THREE.Mesh(new THREE.OctahedronGeometry(j === 0 ? 0.035 : 0.017), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.7, 0.6, 1.3) }));
    g.add(node);
  }
  rings.push(g);
}

const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -1.85, 0), new THREE.Vector3(0, 1.85, 0)]), new THREE.LineBasicMaterial({ color: '#d68fcf', transparent: true, opacity: 0.27, blending: THREE.AdditiveBlending, depthWrite: false }));
body.add(axis);
for (const y of [-1.82, 1.82]) {
  const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(0.055), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 0.7, 1.2) }));
  crystal.position.y = y; crystal.scale.set(0.65,1.7,0.65); body.add(crystal);
}

let seed = 7919;
function random() { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
const particlePositions = [], sizes = [], phases = [];
for (let i = 0; i < 390; i++) {
  const a = random() * Math.PI * 2, y = (random() - 0.5) * 2.9;
  const r = (1.3 + random() * 0.8) * Math.sqrt(Math.max(0.12, 1 - (y / 1.8) ** 2));
  particlePositions.push(Math.cos(a) * r, y, Math.sin(a) * r);
  sizes.push(1 + random() * 1.7); phases.push(random() * 9);
}
const pg = new THREE.BufferGeometry();
pg.setAttribute('position', new THREE.Float32BufferAttribute(particlePositions, 3));
pg.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
pg.setAttribute('phase', new THREE.Float32BufferAttribute(phases, 1));
const points = new THREE.Points(pg, new THREE.ShaderMaterial({
  uniforms: { ...shaderUniforms },
  vertexShader: `uniform float uTime;uniform float uEnergy;attribute float size;attribute float phase;varying float opacity;void main(){vec3 p=position; p.y+=sin(uTime*.4+phase)*.045;vec4 mv=modelViewMatrix*vec4(p,1.);gl_Position=projectionMatrix*mv;gl_PointSize=size*(5.0/-mv.z)*(1.+uEnergy*.5);opacity=.2+.65*pow(.5+.5*sin(uTime+phase),3.);}`,
  fragmentShader: `varying float opacity;void main(){float d=length(gl_PointCoord-.5)*2.;gl_FragColor=vec4(1.,.63,.9,pow(max(0.,1.-d),2.)*opacity);}`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
}));
body.add(points);

const floor = new THREE.Group(); scene.add(floor); floor.position.y = -1.97;
for (const r of [0.82, 1.14, 1.22]) {
  const p = Array.from({ length: 161 }, (_, j) => new THREE.Vector3(Math.cos(j / 160 * Math.PI * 2) * r, 0, Math.sin(j / 160 * Math.PI * 2) * r));
  floor.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), new THREE.LineBasicMaterial({ color: '#bb86ce', transparent: true, opacity: 0.08, blending: THREE.AdditiveBlending, depthWrite: false })));
}

const technical = new THREE.Group(); body.add(technical);
const ticks = [];
for (let i = 0; i < 24; i++) {
  const a = i / 24 * Math.PI * 2;
  ticks.push(new THREE.Vector3(Math.cos(a) * 2.06, Math.sin(a) * 2.06, 0), new THREE.Vector3(Math.cos(a) * (i % 3 ? 2.11 : 2.17), Math.sin(a) * (i % 3 ? 2.11 : 2.17), 0));
}
const technicalMaterial = new THREE.LineBasicMaterial({ color: '#d7b4e9', transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
technical.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(ticks), technicalMaterial));
const arc = Array.from({ length: 60 }, (_, i) => new THREE.Vector3(Math.cos(i / 59 * 1.6) * 2.04, Math.sin(i / 59 * 1.6) * 2.04, 0));
technical.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(arc), technicalMaterial));
let technicalEnergy = 0;

let source = 'demo', audioContext, analyser, mediaNode, audioURL;
let mode = 'SIMULATION', runtimeFrame = null;
let runtimeMotion=null, runtimeMotionAt=0;
const speechDriver=new SpeechMotionDriver();
let speechChannels={core:0,petals:0,rings:0,particles:0};
let audioData, bands = { low: 0, mid: 0, high: 0 }, externalLevel = 0;
const audio = $('audio');
function setState(state, manual = true) {
  if (mode === 'RUNTIME' && manual) return;
  motion.setState(state);
  $('state-name').textContent = names[state]; $('state-code').textContent = state;
  document.querySelectorAll('[data-state]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.state === state)));
  if (manual) { cycle = false; $('cycle').setAttribute('aria-pressed', 'false'); }
}
function selectSource(next) {
  source = next;
  $('source-label').textContent = { demo: 'SIMULADA', file: 'AUDIO LOCAL', external: 'SEÑAL EXTERNA' }[next];
  $('demo').classList.toggle('active', next === 'demo');
  $('demo').setAttribute('aria-pressed', String(next === 'demo'));
  document.querySelector('.audio-file').classList.toggle('active', next === 'file');
  $('audio-status').textContent = '';
}
function sampleAudio() {
  if (!analyser || audio.paused) return 0;
  analyser.getByteTimeDomainData(audioData);
  let power = 0;
  for (const value of audioData) power += ((value - 128) / 128) ** 2;
  const spectrum = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(spectrum);
  const mean = (a, b) => spectrum.slice(a, b).reduce((s, n) => s + n, 0) / ((b - a) * 255);
  bands = { low: mean(1, 12), mid: mean(12, 70), high: mean(70, 170) };
  return clamp(Math.sqrt(power / audioData.length) * 3.4);
}
async function playAudio() {
  try {
    audioContext ??= new AudioContext();
    if (!mediaNode) {
      mediaNode = audioContext.createMediaElementSource(audio);
      analyser = audioContext.createAnalyser(); analyser.fftSize = 1024;
      audioData = new Uint8Array(analyser.fftSize);
      mediaNode.connect(analyser); analyser.connect(audioContext.destination);
    }
    await audioContext.resume(); await audio.play();
    paused = false; updatePause();
    selectSource('file'); setState('SPEAKING'); setIcon($('audio-play'), 'pause');
    $('audio-play').setAttribute('aria-label', 'Pausar audio');
  } catch { $('audio-status').textContent = 'No se pudo reproducir este audio.'; }
}
$('audio-file').addEventListener('change', () => {
  const file = $('audio-file').files[0]; if (!file) return;
  audio.pause();
  if (audioURL) URL.revokeObjectURL(audioURL);
  audioURL = URL.createObjectURL(file); audio.src = audioURL;
  $('filename').textContent = file.name; $('audio-play').disabled = false;
  selectSource('file'); setIcon($('audio-play'), 'play');
  $('audio-play').setAttribute('aria-label', 'Reproducir audio');
});
$('audio-play').addEventListener('click', () => {
  if (audio.paused) playAudio();
  else { audio.pause(); setIcon($('audio-play'), 'play'); $('audio-play').setAttribute('aria-label', 'Reproducir audio'); }
});
audio.addEventListener('ended', () => {
  setIcon($('audio-play'), 'play'); $('audio-play').setAttribute('aria-label', 'Reproducir audio');
  if (source === 'file' && motion.state === 'SPEAKING') setState('IDLE');
});
audio.addEventListener('error', () => { if (audio.src) $('audio-status').textContent = 'Formato de audio no disponible.'; });
$('demo').addEventListener('click', () => { audio.pause(); selectSource('demo'); setState('SPEAKING'); setIcon($('audio-play'), 'play'); });

let paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
let cycle = false, cycleTime = 0, lastCycleIndex = -1;
let intensity = 0.75, speed = 1;
const runtimeBridge = new RuntimeBridge({ onState: frame => {
  if (mode !== 'RUNTIME') return;
  runtimeFrame = frame.connected ? frame : null;
  setState(frame.connected ? frame.state : 'IDLE', false);
  $('connection-text').textContent = frame.connected ? 'Conectado' : 'Sin conexi\u00f3n';
  document.querySelector('.mode-switch').classList.toggle('connected', frame.connected);
}, onMotion: frame=>{runtimeMotion=frame;runtimeMotionAt=performance.now();
  if(mode==='RUNTIME') motion.setWorkingMode(frame?.working_mode??'NONE');
} });
function setMode(next) {
  if (!['SIMULATION', 'RUNTIME'].includes(next)) throw new TypeError('Unknown presence mode');
  mode = next; audio.pause(); cycle = false;
  $('cycle').setAttribute('aria-pressed', 'false');
  document.querySelectorAll('[data-mode]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.mode === mode)));
  document.querySelectorAll('[data-state], #cycle, #demo, #audio-file, #working-mode, #speech-profile, #speech-gain').forEach(el => { el.disabled = mode === 'RUNTIME'; });
  $('audio-play').disabled = mode === 'RUNTIME' || !audioURL;
  document.querySelector('.audio-file').setAttribute('aria-disabled', String(mode === 'RUNTIME'));
  if (mode === 'RUNTIME') {
    source = 'runtime'; externalLevel = 0;
    $('source-label').textContent = 'B4 · PCM REAL';
    $('audio-status').textContent = '';
    runtimeBridge.connect();
  } else {
    runtimeBridge.close(); runtimeFrame = null;
    runtimeMotion=null; motion.setWorkingMode('NONE');
    selectSource('demo'); setState('IDLE');
    $('connection-text').textContent = 'Local';
    document.querySelector('.mode-switch').classList.remove('connected');
  }
}
document.querySelectorAll('[data-mode]').forEach(el => el.addEventListener('click', () => setMode(el.dataset.mode)));
document.querySelectorAll('[data-state]').forEach(el => el.addEventListener('click', () => setState(el.dataset.state)));
$('working-mode').addEventListener('change',e=>{if(mode==='SIMULATION'){motion.setWorkingMode(e.target.value);setState('WORKING');}});
$('cycle').addEventListener('click', () => { cycle = !cycle; cycleTime = 0; lastCycleIndex = -1; $('cycle').setAttribute('aria-pressed', String(cycle)); });
$('intensity').addEventListener('input', e => { intensity = Number(e.target.value) / 100; $('intensity-value').textContent = e.target.value + '%'; });
$('speed').addEventListener('input', e => { speed = Number(e.target.value) / 100; $('speed-value').textContent = speed.toFixed(1) + 'x'; });
$('reset').addEventListener('click', resetView);
function updatePause() { setIcon($('pause'), paused ? 'play' : 'pause'); $('pause').setAttribute('aria-label', paused ? 'Reanudar movimiento' : 'Pausar movimiento'); $('pause').title = $('pause').getAttribute('aria-label'); }
$('pause').addEventListener('click', () => { paused = !paused; if (paused) audio.pause(); updatePause(); });
updatePause();
$('fullscreen').addEventListener('click', async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { /* Browser may deny fullscreen; scene stays usable. */ }
});
document.addEventListener('fullscreenchange', () => setIcon($('fullscreen'), document.fullscreenElement ? 'minimize' : 'maximize'));
$('capture').addEventListener('click', () => {
  composer.render();
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `anydes-petal-${motion.state.toLowerCase()}.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  });
});
for (let i = 0; i < 21; i++) $('wave').append(document.createElement('i'));
const bars = [...$('wave').children];

let last = performance.now(), frames = 0;
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (!paused) {
    if (cycle) {
      cycleTime += dt;
      const index = Math.floor(cycleTime / 5) % 5;
      if (index !== lastCycleIndex) { setState(Object.keys(STATES)[index], false); lastCycleIndex = index; }
    }
    const profile=$('speech-profile').value;
    const gain=Number($('speech-gain').value)/100*(profile==='quiet'?0.25:profile==='loud'?1.3:profile==='pause'?0:1);
    const level = source === 'demo' ? speechEnvelope(motion.time)*gain : source === 'file' ? sampleAudio() : externalLevel;
    const real=mode==='RUNTIME' && runtimeMotion && now-runtimeMotionAt<180 ? runtimeMotion : null;
    const rms=mode==='RUNTIME' ? real?.rms??0 : level*0.16;
    const peak=mode==='RUNTIME' ? real?.peak??0 : level*0.5;
    speechChannels=speechDriver.update(dt,rms,peak,motion.state==='SPEAKING');
    motion.update(dt,speechDriver.energy,speed);
    cognition.update(dt,motion.state,motion.workingMode);
  }
  const t = motion.time, e = motion.energy, petalEnergy=speechChannels.petals, ringEnergy=speechChannels.rings;
  if (!paused) technicalEnergy += ((motion.state === 'WORKING' ? 1 : 0) - technicalEnergy) * (1 - Math.exp(-dt * 4));
  technicalMaterial.opacity = technicalEnergy * (0.52+cognition.build*0.22+cognition.check*cognition.verifyPulse*0.12);
  technical.rotation.set(0.16, 0.1, -t * 0.18*(1-cognition.build)+cognition.assembly*cognition.build);
  const breath = Math.sin(t * 1.15) * 0.014;
  body.scale.setScalar(motion.scale + breath + e * 0.026);
  body.rotation.y = motion.rotation;
  body.rotation.z = Math.sin(t * 0.17) * 0.035;
  for (let i = 0; i < layers.length; i++) {
    layers[i].rotation.y = Math.sin(t * 0.23 + i) * motion.complexity * 0.4 + (i === 1 ? -motion.rotation * 1.4 : motion.rotation * i * 0.25);
    layers[i].rotation.z = [0.07, -0.16, 0.23][i] + Math.sin(t * 0.3 + i) * motion.complexity * 0.05+Math.sin(t*1.6+i*2.1)*motion.flow*0.025;
    layers[i].rotation.y=blendAngle(layers[i].rotation.y,cognition.assembly*(i%2?-1:1)+i*0.38,cognition.build);
    layers[i].rotation.z+=cognition.focus*Math.sin(t*(0.55+i*0.31)+i)*0.11;
    const inward=1-cognition.focus*(0.018+i*0.008)*(0.5+0.5*Math.sin(cognition.phase*(1.4+i*0.21)+i));
    layers[i].scale.set(cognition.radial*inward,cognition.height-cognition.check*cognition.converge*0.045,cognition.radial*inward);
  }
  for (const p of petals) {
    const wave = Math.sin(t * 2.7 - p.phase) * petalEnergy * 0.022;
    p.pivot.rotation.z = (motion.opening + petalEnergy * 0.21 + wave+motion.flow*Math.sin(t*2-p.layer)*0.035) * (p.layer === 1 ? -0.6 : 0.55);
    p.pivot.rotation.z+=(cognition.fold+cognition.build*cognition.assembly*0.18-cognition.check*cognition.converge*0.09)*(p.layer===1?-0.6:0.55);
    p.pivot.rotation.x = Math.sin(p.angle * 2 + p.layer) * 0.11;
    p.pivot.position.y = Math.sin(t * 0.9 + p.phase) * 0.016 + petalEnergy * Math.sin(p.phase) * 0.025;
    const assemble=0.5+0.5*Math.sin(cognition.phase*3.3);
    p.pivot.position.y+=cognition.build*assemble*0.055;
    const inwardPulse=0.5+0.5*Math.sin(cognition.phase*(1.8+p.layer*0.19)+p.phase);
    const radialOffset=-cognition.focus*(0.025+inwardPulse*0.065)+cognition.build*(0.055+assemble*0.065);
    p.pivot.position.x=Math.cos(p.angle)*radialOffset;
    p.pivot.position.z=Math.sin(p.angle)*radialOffset;
    p.pivot.scale.setScalar(p.baseScale * (1 + e * 0.025));
  }
  rings.forEach((r, i) => {
    const working = motion.state === 'WORKING' ? 1 : 0;
    r.rotation.set([0.35, -0.38, 0.65, 1.08][i] + Math.sin(t * 0.21 + i) * motion.complexity * 0.16,
      t * (i % 2 ? -0.08 : 0.1) * (1 + motion.complexity), [0.2, -0.12, -0.28, 0.1][i] + working * 0.12);
    r.scale.setScalar(1 + ringEnergy * 0.035+Math.sin(t*2-i)*motion.flow*0.012);
    r.rotation.x=THREE.MathUtils.lerp(r.rotation.x,0.10,cognition.build);
    r.rotation.z=blendAngle(r.rotation.z,cognition.assembly*0.25,cognition.build);
    r.position.y=cognition.build*(i-1.5)*0.30;
    r.scale.multiplyScalar(1+cognition.build*0.10-cognition.check*cognition.converge*0.12-cognition.focus*0.14);
    r.children[0].material.opacity=[0.27,0.27,0.27,0.11][i]+cognition.structure*0.28;
    r.children.slice(1).forEach((node, j) => {
      const baseA = t * (0.2 + motion.complexity * 0.3+motion.coherence*0.28+ringEnergy*0.12) * (i % 2 ? -1 : 1) + j * 2.1 + i;
      const a=blendAngle(baseA,cognition.phase*1.8+j*Math.PI*2/3+i*Math.PI/2,cognition.build);
      const progress=(cognition.phase*0.48+j/3)%1;
      const inwardProgress=(cognition.phase*(0.36+i*0.057)+i*0.17+j/3)%1;
      const radial=1-cognition.focus*inwardProgress*0.65+cognition.build*(-0.54+progress*0.68)-cognition.check*cognition.converge*0.32-cognition.outward*(1-progress)*0.65;
      node.position.set(Math.cos(a) * (1.87 + i * 0.075)*radial, 0, Math.sin(a) * (1.87 + i * 0.075)*radial);
      const impulse=Math.pow(Math.max(0,Math.cos(cognition.phase*3.3-j*2.1)),4);
      const fade=(1-cognition.build-cognition.focus)+cognition.build*Math.sin(progress*Math.PI)+cognition.focus*Math.sin(inwardProgress*Math.PI);
      const glow=(1+cognition.build*impulse*2.0+cognition.check*cognition.verifyPulse*0.8)*Math.max(0.05,fade);
      node.scale.setScalar(1+cognition.build*(0.55+impulse*0.7)+cognition.focus*0.25);
      node.material.color.setRGB(1.7*glow,0.6*glow,1.3*glow);
      node.visible=cognition.outward<0.8 || (progress>0.04&&progress<0.96);
    });
  });
  const thought=cognition.focus*(0.025+cognition.irregular*0.04)+cognition.check*cognition.verifyPulse*0.065;
  core.scale.setScalar(1 + speechChannels.core * 0.24+thought);
  pearl.material.color.copy(pearlBase).lerp(pearlWarm,cognition.focus*0.78);
  diamond.material.color.copy(diamondBase).lerp(diamondWarm,cognition.focus*0.65);
  diamond.rotation.set(t * 0.14, t * -0.22, Math.PI / 4);
  pearl.scale.setScalar(1 + Math.sin(t * 1.5) * 0.04 + e * 0.25);
  shaderUniforms.uTime.value = t;
  shaderUniforms.uEnergy.value = speechChannels.particles;
  shaderUniforms.uFocus.value = cognition.focus;
  shaderUniforms.uLight.value = intensity * motion.light;
  bloom.strength = 0.38 + intensity * 0.15 + e * 0.11;
  controls.update(); renderer.info.reset(); composer.render(); frames++;
  if (frames % 3 === 0) {
    bars.forEach((bar, i) => { bar.style.height = `${3 + e * (7 + 14 * Math.pow(Math.sin(t * 8 + i * 0.9), 2))}px`; });
    $('signal-value').textContent = String(Math.round(e * 99)).padStart(2, '0');
    if($('motion-debug').open) $('motion-values').textContent=`${mode} / ${motion.state}\nworking_mode ${motion.workingMode}\naudio_energy ${speechDriver.raw.toFixed(3)}\nsmoothed_energy ${speechDriver.energy.toFixed(3)}\nvoice_active ${speechDriver.active}\nSSE ms ${runtimeMotion ? Math.max(0,Date.now()-Date.parse(runtimeMotion.timestamp)) : '-'} `;
  }
}
canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); $('error').hidden = false; $('error').textContent = 'Se perdió el contexto gráfico. Recarga el laboratorio para recuperar la escena.'; });
document.addEventListener('visibilitychange', () => { last = performance.now(); if (document.hidden) audio.pause(); });
resetView(); requestAnimationFrame(animate);

// Runtime receives only numeric audio features, never PCM or core commands.
window.anydesPresence = Object.freeze({
  setState: state => { if (mode === 'RUNTIME') throw new Error('Runtime mode is observer-controlled'); setState(state); },
  setMode,
  setAudioLevel: value => {
    if (mode === 'RUNTIME') throw new Error('Runtime audio is read-only');
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) throw new TypeError('Audio level must be 0..1');
    selectSource('external'); externalLevel = value;
  },
  snapshot: () => ({ mode, state: motion.state, source, energy: motion.energy, time: motion.time, paused, frames, petals: petals.length, bands: { ...bands }, camera: camera.position.toArray(), drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles, webgl: !!renderer.getContext(), connectedToAnydes: mode === 'RUNTIME' && runtimeBridge.connected, runtimeFrame, runtimeMotion, workingMode:motion.workingMode, cognition:{focus:cognition.focus,structure:cognition.structure,build:cognition.build,check:cognition.check,outward:cognition.outward,assembly:cognition.assembly,radial:cognition.radial,height:cognition.height,direction:cognition.direction}, speech:{raw:speechDriver.raw,smoothed:speechDriver.energy,active:speechDriver.active,channels:{...speechChannels}} }),
});
if (new URLSearchParams(location.search).get('mode') === 'runtime') setMode('RUNTIME');
addEventListener('pagehide', () => runtimeBridge.close());
