import * as THREE from 'three';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { MToonMaterialLoaderPlugin } from './vendor/three-vrm-materials-mtoon.module.js';

const canvas = document.querySelector('#avatar');
const loading = document.querySelector('#loading');
const clock = new THREE.Clock();
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);
const queryParameters = new URLSearchParams(window.location.search);
const HQ_STAGE_ORDER = {
  baseline: 0,
  eyes: 1,
  hair: 2,
  skin: 3,
  clothing: 4,
  lighting: 5,
  postprocess: 6,
};
const HQ_PROFILE = queryParameters.get('profile') === 'hq_efficient';
const requestedHqStage = String(queryParameters.get('hqStage') || 'postprocess').toLowerCase();
const HQ_STAGE = Object.hasOwn(HQ_STAGE_ORDER, requestedHqStage)
  ? requestedHqStage
  : 'postprocess';

function hqAtLeast(stageName) {
  return HQ_PROFILE && HQ_STAGE_ORDER[HQ_STAGE] >= HQ_STAGE_ORDER[stageName];
}
const DEFAULT_CAMERA_MODES = {
  environment: {
    position: new THREE.Vector3(0, 1.42, 5.65),
    target: new THREE.Vector3(0, 1.38, 0),
    fov: 28,
  },
  conversation: {
    position: new THREE.Vector3(0, 1.72, 4.1),
    target: new THREE.Vector3(0, 1.78, 0),
    fov: 25,
  },
  face_review: {
    position: new THREE.Vector3(0, 2.22, 20),
    target: new THREE.Vector3(0, 2.22, 0),
    fov: 2.2,
  },
};
const renderer = new THREE.WebGLRenderer({
  canvas,
  alpha: true,
  antialias: true,
  powerPreference: 'high-performance',
  premultipliedAlpha: true,
});

renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = hqAtLeast('lighting') ? 1.18 : 1.28;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

camera.position.copy(DEFAULT_CAMERA_MODES.environment.position);
camera.lookAt(DEFAULT_CAMERA_MODES.environment.target);

const stage = new THREE.Group();
scene.add(stage);

const hemi = new THREE.HemisphereLight(
  hqAtLeast('lighting') ? 0xfff2ea : 0xffeee4,
  hqAtLeast('lighting') ? 0x30283d : 0x40364f,
  hqAtLeast('lighting') ? 0.58 : 0.72,
);
scene.add(hemi);

const keyLight = new THREE.DirectionalLight(0xffd4ad, hqAtLeast('lighting') ? 1.36 : 1.18);
keyLight.position.set(2.55, 3.95, 3.45);
keyLight.castShadow = true;
keyLight.shadow.mapSize.setScalar(hqAtLeast('lighting') ? 2048 : 1024);
keyLight.shadow.bias = -0.00035;
keyLight.shadow.normalBias = hqAtLeast('lighting') ? 0.018 : 0;
keyLight.shadow.radius = hqAtLeast('lighting') ? 2.2 : 1;
keyLight.shadow.camera.left = -2.5;
keyLight.shadow.camera.right = 2.5;
keyLight.shadow.camera.top = 4;
keyLight.shadow.camera.bottom = -0.4;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xb8d0ff, hqAtLeast('lighting') ? 1.28 : 1.05);
rimLight.position.set(-2.7, 2.9, -2.5);
scene.add(rimLight);

const warmFill = new THREE.PointLight(0xffb77e, hqAtLeast('lighting') ? 2.35 : 2.8, 7.0, 2.0);
warmFill.position.set(2.1, 1.7, 2.2);
scene.add(warmFill);

const coolFill = new THREE.DirectionalLight(0xc7dcff, hqAtLeast('lighting') ? 0.42 : 0);
coolFill.position.set(-1.8, 2.1, 2.8);
scene.add(coolFill);

const faceFill = new THREE.PointLight(0xffeee6, hqAtLeast('lighting') ? 0.74 : 0, 4.5, 2.0);
faceFill.position.set(0, 2.18, 3.1);
scene.add(faceFill);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(1.08, 64),
  new THREE.MeshBasicMaterial({
    color: hqAtLeast('lighting') ? 0x130d18 : 0x09070d,
    transparent: true,
    opacity: hqAtLeast('lighting') ? 0.22 : 0.28,
    depthWrite: false,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = 0.006;
floor.receiveShadow = true;
floor.renderOrder = -1;
stage.add(floor);

const runtime = {
  ready: false,
  error: '',
  frames: 0,
  adapter: null,
  model: null,
  state: {
    expression: 'neutral',
    expressionIntensity: 0.5,
    gesture: 'idle',
    gaze: 'user',
    posture: 'relaxed',
    attention: 'user',
    speech: {},
    lipSync: {},
  },
  pointer: { x: 0, y: 0 },
  speech: {
    active: false,
    mode: 'legacy',
    start: 0,
    end: 0,
    bodyEnd: 0,
    sequence: ['aa'],
    timeline: [],
  },
  blink: { value: 0, next: 1.7, started: -1 },
  environment: { warmLight: 0.58, coolLight: 0.72, lampGlow: 0.8 },
  restPose: {
    enabled: true,
    name: '',
    bones: {},
    ikTargets: {},
  },
  camera: {
    mode: 'environment',
    modes: DEFAULT_CAMERA_MODES,
    target: DEFAULT_CAMERA_MODES.environment.target.clone(),
  },
  motionSample: null,
  lastMotion: { breath: 0, sway: 0 },
  eyeWhiteMaterials: 0,
  mtoonMaterials: 0,
  hqMaterials: { eyes: 0, hair: 0, skin: 0, clothing: 0 },
};

const clothRoughnessTexture = hqAtLeast('clothing')
  ? createClothRoughnessTexture()
  : null;

const profiler = {
  targetFps: 0,
  lastFrameMs: 0,
  lastRenderedAt: 0,
  frameIntervals: [],
};

const postPipeline = null;
if (hqAtLeast('postprocess')) {
  // Qt WebEngine already creates a multisampled antialiased canvas. A second
  // framebuffer costs substantially more VRAM and is less reliable on its
  // ANGLE path, so HQ Efficient keeps presentation correction in the final
  // compositor without another full-resolution render target.
  canvas.style.filter = 'contrast(1.035) saturate(1.025)';
}

class VrmAdapter {
  constructor(gltf) {
    this.gltf = gltf;
    this.root = gltf.scene;
    this.document = gltf.parser.json;
    this.vrm = this.document.extensions?.VRMC_vrm || null;
    this.bones = new Map();
    this.baseBones = new Map();
    this.expressions = new Map();
    this.expressionValues = new Map();
  }

  async initialize() {
    if (!this.vrm) throw new Error('missing_vrm_1_extension');
    const humanBones = this.vrm.humanoid?.humanBones || {};
    await Promise.all(Object.entries(humanBones).map(async ([name, spec]) => {
      const bone = await this.gltf.parser.getDependency('node', spec.node);
      if (!bone) return;
      this.bones.set(name, bone);
      this.baseBones.set(name, {
        quaternion: bone.quaternion.clone(),
        position: bone.position.clone(),
      });
    }));

    const expressionGroups = this.vrm.expressions || {};
    const definitions = {
      ...(expressionGroups.preset || {}),
      ...(expressionGroups.custom || {}),
    };
    await Promise.all(Object.entries(definitions).map(async ([name, definition]) => {
      const targets = [];
      for (const bind of definition.morphTargetBinds || []) {
        const node = await this.gltf.parser.getDependency('node', bind.node);
        if (!node) continue;
        node.traverse((object) => {
          if (!object.morphTargetInfluences || bind.index >= object.morphTargetInfluences.length) return;
          targets.push({
            object,
            index: bind.index,
            scale: Number.isFinite(bind.weight) ? bind.weight : 1,
          });
        });
      }
      if (targets.length) {
        this.expressions.set(name, targets);
        this.expressionValues.set(name, 0);
      }
    }));
  }

  setExpression(name, target, smoothing = 0.22) {
    if (!this.expressions.has(name)) return;
    const current = this.expressionValues.get(name) || 0;
    this.expressionValues.set(name, THREE.MathUtils.lerp(current, clamp01(target), smoothing));
  }

  flushExpressions() {
    for (const [name, targets] of this.expressions.entries()) {
      const value = this.expressionValues.get(name) || 0;
      for (const target of targets) {
        target.object.morphTargetInfluences[target.index] = value * target.scale;
      }
    }
  }

  poseBone(name, rotation, position, smoothing = 0.1) {
    const bone = this.bones.get(name);
    const base = this.baseBones.get(name);
    if (!bone || !base) return;
    const offset = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(rotation.x || 0, rotation.y || 0, rotation.z || 0, 'XYZ'),
    );
    const targetQuaternion = base.quaternion.clone().multiply(offset);
    bone.quaternion.slerp(targetQuaternion, smoothing);
    const targetPosition = base.position.clone().add(
      new THREE.Vector3(position.x || 0, position.y || 0, position.z || 0),
    );
    bone.position.lerp(targetPosition, smoothing);
  }

  aimBoneAt(name, childName, targetModel, smoothing = 0.16) {
    const bone = this.bones.get(name);
    const child = this.bones.get(childName);
    if (!bone || !child || !runtime.model) return;
    this.root.updateMatrixWorld(true);
    const head = bone.getWorldPosition(new THREE.Vector3());
    const tail = child.getWorldPosition(new THREE.Vector3());
    const target = runtime.model.localToWorld(targetModel.clone());
    const currentDirection = tail.sub(head).normalize();
    const targetDirection = target.sub(head).normalize();
    if (currentDirection.lengthSq() < 1e-8 || targetDirection.lengthSq() < 1e-8) return;

    const currentWorld = bone.getWorldQuaternion(new THREE.Quaternion());
    const worldDelta = new THREE.Quaternion().setFromUnitVectors(
      currentDirection,
      targetDirection,
    );
    const targetWorld = worldDelta.multiply(currentWorld);
    const parentWorld = bone.parent
      ? bone.parent.getWorldQuaternion(new THREE.Quaternion())
      : new THREE.Quaternion();
    const targetLocal = parentWorld.invert().multiply(targetWorld);
    bone.quaternion.slerp(targetLocal, smoothing);
    this.root.updateMatrixWorld(true);
  }
}

function configureModel(model) {
  model.traverse((object) => {
    if (!object.isMesh && !object.isSkinnedMesh) return;
    const isPocketObject = String(object.name || '')
      .toLowerCase()
      .includes('kangaroopocket');
    object.castShadow = true;
    // The hidden hands belong to the merged body mesh and would project a hard
    // silhouette through opaque cloth. Keep direct light, normals and material
    // shading on the pocket while rejecting only that invalid projected shadow.
    object.receiveShadow = !isPocketObject;
    object.frustumCulled = false;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (let materialIndex = 0; materialIndex < materials.length; materialIndex += 1) {
      let material = materials[materialIndex];
      if (!material) continue;
      let layer = materialLayer(material.name || '');
      if (hqAtLeast('eyes') && layer.role === 'eyeIris') {
        material = createHqIrisMaterial(material);
        runtime.hqMaterials.eyes += 1;
      } else if (hqAtLeast('eyes') && layer.role === 'eyeHighlight') {
        material = createHqEyeHighlightMaterial(material);
        runtime.hqMaterials.eyes += 1;
      } else if (hqAtLeast('hair') && layer.role === 'hair') {
        material = createHqHairMaterial(material);
        runtime.hqMaterials.hair += 1;
      } else if (hqAtLeast('skin') && layer.role === 'skin') {
        configureHqMToonSkinMaterial(material, { face: true });
        runtime.hqMaterials.skin += 1;
      } else if (hqAtLeast('skin') && layer.role === 'bodySkin') {
        configureHqMToonSkinMaterial(material, { face: false });
        runtime.hqMaterials.skin += 1;
      } else if (hqAtLeast('clothing') && layer.role === 'clothing') {
        material = createHqClothingMaterial(material, { pocket: isPocketObject });
        runtime.hqMaterials.clothing += 1;
      }
      if (material !== materials[materialIndex]) {
        materials[materialIndex] = material;
        if (Array.isArray(object.material)) {
          object.material[materialIndex] = material;
        } else {
          object.material = material;
        }
        layer = materialLayer(material.name || '');
      }
      if (
        material.isMToonMaterial === true
        && layer.role === 'clothing'
      ) {
        material = createClothingMaterial(material, {
          pocket: isPocketObject,
        });
        materials[materialIndex] = material;
        if (Array.isArray(object.material)) {
          object.material[materialIndex] = material;
        } else {
          object.material = material;
        }
        layer = materialLayer(material.name || '');
      }
      if (
        material.isMToonMaterial === true
        && layer.role === 'eyeWhite'
      ) {
        material = createNeutralEyeWhiteMaterial(material);
        materials[materialIndex] = material;
        if (Array.isArray(object.material)) {
          object.material[materialIndex] = material;
        } else {
          object.material = material;
        }
        layer = materialLayer(material.name || '');
      }
      if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
      const isMToon = material.isMToonMaterial === true;
      if (isMToon) {
        runtime.mtoonMaterials += 1;
      } else {
        material.transparent = Boolean(material.transparent);
        material.alphaTest = Math.max(material.alphaTest || 0, material.transparent ? 0.02 : 0);
        material.depthWrite = layer.overlay ? false : material.opacity >= 0.96;
        if (layer.overlay) material.side = THREE.DoubleSide;
        object.renderOrder = Math.max(object.renderOrder || 0, layer.order);
      }
      if (layer.role === 'eyeWhite') {
        if (material.userData.anydesNeutralEyeWhite === true) {
          material.needsUpdate = true;
        } else if (isMToon) {
          configureMToonEyeWhiteMaterial(material);
        } else {
          configureEyeWhiteMaterial(material);
        }
        runtime.eyeWhiteMaterials += 1;
      }
      if (isMToon && layer.role === 'skin') {
        configureMToonFaceMaterial(material);
      }
      if (isMToon && layer.role === 'eyeIris') {
        configureMToonIrisMaterial(material);
      }
      material.needsUpdate = true;
    }
  });

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const targetHeight = 2.72;
  const scale = targetHeight / Math.max(0.01, size.y);
  model.scale.setScalar(scale);
  model.updateMatrixWorld(true);

  const scaledBox = new THREE.Box3().setFromObject(model);
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
  model.position.x -= scaledCenter.x;
  model.position.y -= scaledBox.min.y;
  model.position.z -= scaledCenter.z;
  model.userData.anydesBaseY = model.position.y;
}

function createClothingMaterial(source, options = {}) {
  const sourceName = String(source.name || '').toLowerCase();
  const isSkirt = sourceName.includes('bottoms_01_cloth');
  const isPocket = options.pocket === true;
  return new THREE.MeshStandardMaterial({
    name: `${source.name || 'Anydes clothing'} - standard`,
    color: source.color?.clone?.() || new THREE.Color(0x6a3f7a),
    map: source.map || null,
    normalMap: source.normalMap || null,
    normalScale: isPocket
      ? new THREE.Vector2(0.18, 0.18)
      : (source.normalScale?.clone?.() || new THREE.Vector2(0.52, 0.52)),
    emissive: new THREE.Color(isSkirt ? 0x16070a : 0x210e29),
    emissiveMap: source.map || null,
    emissiveIntensity: isSkirt ? 0.08 : 0.14,
    roughness: isSkirt ? 0.89 : 0.94,
    metalness: 0,
    side: THREE.DoubleSide,
    flatShading: false,
  });
}

function surfaceOptions(source) {
  return {
    color: source.color?.clone?.() || new THREE.Color(0xffffff),
    map: source.map || null,
    normalMap: source.normalMap || null,
    normalScale: source.normalScale?.clone?.() || new THREE.Vector2(0.45, 0.45),
    transparent: Boolean(source.transparent),
    opacity: Number.isFinite(source.opacity) ? source.opacity : 1,
    alphaTest: source.alphaTest || 0,
    depthTest: source.depthTest !== false,
    depthWrite: source.depthWrite !== false,
    side: source.side,
    flatShading: false,
  };
}

function createHqIrisMaterial(source) {
  const material = new THREE.MeshPhysicalMaterial({
    ...surfaceOptions(source),
    name: `${source.name || 'Anydes iris'} - hq-eyes`,
    metalness: 0,
    roughness: 0.24,
    clearcoat: 0.72,
    clearcoatRoughness: 0.2,
    specularIntensity: 0.58,
    emissive: new THREE.Color(0x2a101d),
    emissiveMap: source.map || null,
    emissiveIntensity: 0.16,
  });
  material.alphaToCoverage = true;
  material.userData.anydesHqRole = 'eyes';
  return material;
}

function createHqEyeHighlightMaterial(source) {
  const material = new THREE.MeshBasicMaterial({
    name: `${source.name || 'Anydes eye highlight'} - hq-eyes`,
    color: 0xffffff,
    map: source.map || null,
    transparent: true,
    opacity: Math.min(1, (Number(source.opacity) || 1) * 0.94),
    alphaTest: source.alphaTest || 0,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  material.alphaToCoverage = true;
  material.userData.anydesHqRole = 'eyes';
  return material;
}

function createHqHairMaterial(source) {
  const material = new THREE.MeshPhysicalMaterial({
    ...surfaceOptions(source),
    name: `${source.name || 'Anydes hair'} - hq-hair`,
    metalness: 0,
    roughness: 0.46,
    anisotropy: 0.42,
    anisotropyRotation: Math.PI / 2,
    sheen: 0.24,
    sheenColor: new THREE.Color(0xffa0bc),
    sheenRoughness: 0.58,
    specularIntensity: 0.48,
    emissive: new THREE.Color(0x2b0714),
    emissiveMap: source.emissiveMap || source.map || null,
    emissiveIntensity: 0.09,
  });
  material.alphaToCoverage = true;
  material.userData.anydesHqRole = 'hair';
  return material;
}

function configureHqMToonSkinMaterial(source, options = {}) {
  if (source.userData.anydesHqSkin === true) return source;
  source.userData.anydesHqSkin = true;
  const face = options.face === true;
  if (source.normalScale) {
    source.normalScale.set(face ? 0.36 : 0.42, face ? 0.36 : 0.42);
  }
  if (source.shadeColorFactor) {
    source.shadeColorFactor.setRGB(
      face ? 0.93 : 0.88,
      face ? 0.72 : 0.69,
      face ? 0.67 : 0.65,
    );
  }
  if (source.emissive) {
    source.emissive.setRGB(face ? 0.38 : 0.22, face ? 0.2 : 0.12, face ? 0.13 : 0.1);
    source.emissiveIntensity = face ? 0.46 : 0.24;
  }
  source.needsUpdate = true;
  return source;
}

function createHqClothingMaterial(source, options = {}) {
  const sourceName = String(source.name || '').toLowerCase();
  const isSkirt = sourceName.includes('bottoms_01_cloth');
  const isPocket = options.pocket === true || sourceName.includes('finalpocket');
  const material = new THREE.MeshPhysicalMaterial({
    ...surfaceOptions(source),
    name: `${source.name || 'Anydes clothing'} - hq-clothing`,
    metalness: 0,
    roughness: isSkirt ? 0.78 : (isPocket ? 0.88 : 0.84),
    roughnessMap: clothRoughnessTexture,
    sheen: isSkirt ? 0.18 : 0.12,
    sheenColor: new THREE.Color(isSkirt ? 0x7d2635 : 0x805195),
    sheenRoughness: 0.76,
    specularIntensity: isSkirt ? 0.24 : 0.18,
    emissive: new THREE.Color(isSkirt ? 0x120407 : 0x15091c),
    emissiveMap: source.emissiveMap || source.map || null,
    emissiveIntensity: isSkirt ? 0.045 : 0.055,
  });
  material.normalScale = isPocket
    ? new THREE.Vector2(0.2, 0.2)
    : (source.normalScale?.clone?.() || new THREE.Vector2(0.48, 0.48));
  material.userData.anydesHqRole = 'clothing';
  return material;
}

function createClothRoughnessTexture() {
  const detail = document.createElement('canvas');
  detail.width = 64;
  detail.height = 64;
  const context = detail.getContext('2d', { alpha: false });
  context.fillStyle = '#dedede';
  context.fillRect(0, 0, 64, 64);
  for (let index = 0; index < 64; index += 2) {
    const shade = 205 + ((index * 17) % 19);
    context.strokeStyle = `rgb(${shade},${shade},${shade})`;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(index, 0);
    context.lineTo(index - 16, 64);
    context.stroke();
  }
  const texture = new THREE.CanvasTexture(detail);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(7, 11);
  texture.colorSpace = THREE.NoColorSpace;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function createNeutralEyeWhiteMaterial(source) {
  const material = new THREE.MeshBasicMaterial({
    name: `${source.name || 'Anydes EyeWhite'} - neutral`,
    color: 0xffffff,
    map: source.map || null,
    transparent: Boolean(source.transparent),
    opacity: source.opacity,
    alphaTest: source.alphaTest || 0,
    depthTest: source.depthTest,
    depthWrite: source.depthWrite,
    side: source.side,
    blending: source.blending,
    blendSrc: source.blendSrc,
    blendDst: source.blendDst,
    blendEquation: source.blendEquation,
    premultipliedAlpha: source.premultipliedAlpha,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
    alphaToCoverage: true,
    toneMapped: false,
  });
  material.userData.anydesNeutralEyeWhite = true;
  return material;
}

function materialLayer(name) {
  const value = String(name).toLowerCase();
  if (value.includes('tops_01_cloth')) return { order: 0, overlay: false, role: 'clothing' };
  if (value.includes('bottoms_01_cloth')) return { order: 0, overlay: false, role: 'clothing' };
  if (value.includes('body_00_skin')) return { order: 0, overlay: false, role: 'bodySkin' };
  if (value.includes('face_00_skin')) return { order: 10, overlay: false, role: 'skin' };
  if (value.includes('eyewhite')) return { order: 20, overlay: true, role: 'eyeWhite' };
  if (value.includes('eyeiris')) return { order: 30, overlay: true, role: 'eyeIris' };
  if (value.includes('eyehighlight')) return { order: 40, overlay: true, role: 'eyeHighlight' };
  if (value.includes('facemouth')) return { order: 50, overlay: true };
  if (value.includes('facebrow')) return { order: 55, overlay: true };
  if (value.includes('faceeyeline')) return { order: 60, overlay: true };
  if (value.includes('glasseslowlens')) return { order: 70, overlay: true };
  if (value.includes('glasseslowframe')) return { order: 80, overlay: true };
  if (value.includes('hair_00_hair')) return { order: 90, overlay: false, role: 'hair' };
  return { order: 0, overlay: false };
}

function configureMToonEyeWhiteMaterial(material) {
  if (material.userData.anydesEyeWhiteNeutralized) return;
  material.userData.anydesEyeWhiteNeutralized = true;
  material.color.setRGB(1, 1, 1);
  if (material.shadeColorFactor) {
    material.shadeColorFactor.setRGB(1, 1, 1);
  }
  if (material.emissive) {
    material.emissiveMap = null;
    material.emissive.setRGB(1, 1, 1);
    material.emissiveIntensity = 0.24;
  }
  material.toneMapped = false;
  // Keep MToon's original texture, alpha mode, depth and render queue. Those
  // values define the eyelid mask and prevent both pink bleed and white halos.
  material.needsUpdate = true;
}

function configureMToonFaceMaterial(material) {
  if (material.userData.anydesFaceCalibrated) return;
  material.userData.anydesFaceCalibrated = true;
  if (material.userData.anydesHqSkin === true) return;
  // A restrained warm lift matches the canonical VRoid render without
  // recoloring the eye masks or baking room lighting into the model.
  if (material.emissive) {
    material.emissive.setRGB(0.34, 0.19, 0.11);
    material.emissiveIntensity = 0.42;
  }
  material.needsUpdate = true;
}

function configureMToonIrisMaterial(material) {
  if (material.userData.anydesIrisCalibrated) return;
  material.userData.anydesIrisCalibrated = true;
  // The iris keeps its canonical pink texture; this only prevents the room
  // light from making its dark edge look crushed against the white sclera.
  if (material.emissive) {
    material.emissive.setRGB(0.13, 0.11, 0.12);
    material.emissiveIntensity = 0.16;
  }
  material.needsUpdate = true;
}

function configureEyeWhiteMaterial(material) {
  if (material.userData.anydesEyeWhiteNeutralized) return;
  material.userData.anydesEyeWhiteNeutralized = true;
  material.map = null;
  material.alphaMap = null;
  material.color.setRGB(0.992, 0.992, 0.992);
  if (material.emissive) {
    material.emissive.setRGB(0.72, 0.72, 0.72);
    material.emissiveIntensity = 0.48;
  }
  material.opacity = 1;
  material.alphaTest = 0;
  material.transparent = false;
  // The eye-white mesh extends beneath the eyelids. Keeping normal depth
  // occlusion is essential: forcing it on top creates a white mask around
  // the eyes instead of a sclera.
  material.depthTest = true;
  material.depthWrite = true;
  material.polygonOffset = false;
  material.polygonOffsetFactor = 0;
  material.polygonOffsetUnits = 0;
  material.metalness = 0;
  material.roughness = 0.82;
  material.toneMapped = false;
  material.customProgramCacheKey = () => 'anydes-eye-white-neutral-v4';
  material.needsUpdate = true;
}

function expressionName(raw) {
  const value = String(raw || '').toLowerCase();
  const mapping = {
    smile: 'happy',
    soft_smile: 'happy',
    excited: 'happy',
    joyful: 'happy',
    thinking: 'relaxed',
    curious: 'relaxed',
    listening: 'neutral',
    speaking: 'neutral',
    shy: 'relaxed',
    calm: 'relaxed',
  };
  return mapping[value] || value || 'neutral';
}

function updateExpressions(time) {
  const adapter = runtime.adapter;
  if (!adapter) return;
  const selected = expressionName(runtime.state.expression);
  const intensity = clamp01(runtime.state.expressionIntensity);
  const expressionScale = {
    happy: 0.34,
    relaxed: 0.48,
    sad: 0.58,
    surprised: 0.52,
    angry: 0.46,
  }[selected] || 0.55;
  for (const name of adapter.expressions.keys()) {
    if (['blink', 'blinkLeft', 'blinkRight', 'aa', 'ee', 'ih', 'oh', 'ou'].includes(name)) continue;
    adapter.setExpression(name, name === selected ? intensity * expressionScale : 0, 0.12);
  }

  let blinkValue = 0;
  if (runtime.blink.started >= 0) {
    const phase = (time - runtime.blink.started) / 0.13;
    if (phase >= 1) {
      runtime.blink.started = -1;
      runtime.blink.next = time + 2.1 + Math.random() * 3.2;
    } else {
      blinkValue = Math.sin(Math.PI * clamp01(phase));
    }
  } else if (time >= runtime.blink.next) {
    runtime.blink.started = time;
  }
  adapter.setExpression('blink', blinkValue, 0.55);
  adapter.setExpression('blinkLeft', blinkValue, 0.55);
  adapter.setExpression('blinkRight', blinkValue, 0.55);

  const mouth = { aa: 0, ee: 0, ih: 0, oh: 0, ou: 0 };
  if (runtime.speech.active) {
    if (performance.now() >= runtime.speech.start) {
      if (runtime.speech.mode === 'audio_timeline') {
        const elapsedMs = performance.now() - runtime.speech.start;
        const cue = runtime.speech.timeline.find(
          (item) => elapsedMs >= item.startMs && elapsedMs < item.endMs,
        );
        if (cue && Object.hasOwn(mouth, cue.viseme)) {
          mouth[cue.viseme] = clamp01(cue.weight);
        }
      } else if (performance.now() < runtime.speech.end) {
        const elapsed = (performance.now() - runtime.speech.start) / 145;
        const index = Math.floor(elapsed) % runtime.speech.sequence.length;
        const nextIndex = (index + 1) % runtime.speech.sequence.length;
        const blend = (Math.sin(elapsed * Math.PI * 2) + 1) * 0.5;
        mouth[runtime.speech.sequence[index]] = 0.18 + 0.48 * blend;
        mouth[runtime.speech.sequence[nextIndex]] = 0.12 + 0.18 * (1 - blend);
      }
    }
  }
  const pcmEnded = runtime.speech.mode === 'audio_timeline'
    && (!runtime.speech.active || performance.now() >= runtime.speech.end);
  for (const [name, value] of Object.entries(mouth)) adapter.setExpression(name, value, pcmEnded ? 1 : 0.34);
  adapter.flushExpressions();
}

function updatePose(time, delta) {
  const adapter = runtime.adapter;
  if (!adapter) return;
  const breath = runtime.motionSample
    ? THREE.MathUtils.clamp(Number(runtime.motionSample.breath) || 0, -1, 1)
    : Math.sin(time * 1.55);
  const sway = runtime.motionSample
    ? THREE.MathUtils.clamp(Number(runtime.motionSample.sway) || 0, -1, 1)
    : Math.sin(time * 0.43);
  runtime.lastMotion = { breath, sway };
  const gaze = gazeTarget(runtime.state.gaze, runtime.pointer);
  const posture = String(runtime.state.posture || 'relaxed').toLowerCase();
  const gesture = String(runtime.state.gesture || 'idle').toLowerCase();

  adapter.poseBone('hips', { y: sway * 0.012 }, { y: breath * 0.0025 }, 0.08);
  adapter.poseBone('spine', { x: breath * 0.006, z: sway * 0.006 }, {}, 0.08);
  adapter.poseBone('chest', { x: breath * 0.009, z: -sway * 0.005 }, {}, 0.08);
  adapter.poseBone('upperChest', { x: breath * 0.008 }, {}, 0.08);
  adapter.poseBone(
    'head',
    {
      x: gaze.y * 0.12 + (gesture === 'thinking' ? 0.08 : 0),
      y: gaze.x * 0.18 + sway * 0.018,
      z: posture === 'shy' ? 0.035 : 0,
    },
    {},
    0.09,
  );
  adapter.poseBone('leftEye', { x: gaze.y * 0.08, y: gaze.x * 0.12 }, {}, 0.16);
  adapter.poseBone('rightEye', { x: gaze.y * 0.08, y: gaze.x * 0.12 }, {}, 0.16);

  const talking = runtime.speech.active
    && performance.now() >= runtime.speech.start
    && performance.now() < runtime.speech.bodyEnd;
  const talkMotion = talking ? Math.sin(time * 3.1) : 0;
  adapter.poseBone('neck', { x: talkMotion * 0.012, y: talking ? Math.sin(time * 1.7) * 0.015 : 0 }, {}, 0.08);

  if (gesture === 'wave') {
    adapter.poseBone('rightShoulder', {}, {}, 0.1);
    adapter.poseBone('rightUpperArm', { x: -0.22, y: 0.18, z: 1.35 }, {}, 0.13);
    adapter.poseBone('rightLowerArm', { y: 0.18, z: 1.12 }, {}, 0.13);
    adapter.poseBone('rightHand', { x: Math.sin(time * 5.2) * 0.18, z: 0.16 }, {}, 0.16);
  } else if (runtime.restPose.enabled && runtime.restPose.ikTargets.right) {
    updateRestArm('right');
  } else {
    const armSway = talking ? talkMotion * 0.025 : sway * 0.008;
    adapter.poseBone('rightShoulder', {}, {}, 0.08);
    adapter.poseBone('rightUpperArm', { z: armSway }, {}, 0.08);
    adapter.poseBone('rightLowerArm', {}, {}, 0.08);
    adapter.poseBone('rightHand', {}, {}, 0.08);
  }
  if (runtime.restPose.enabled && runtime.restPose.ikTargets.left) {
    updateRestArm('left');
  } else {
    adapter.poseBone('leftShoulder', {}, {}, 0.08);
    adapter.poseBone('leftUpperArm', { z: -talkMotion * 0.012 }, {}, 0.08);
    adapter.poseBone('leftLowerArm', {}, {}, 0.08);
    adapter.poseBone('leftHand', {}, {}, 0.08);
  }

  if (runtime.model) {
    runtime.model.position.y = runtime.model.userData.anydesBaseY + Math.sin(time * 1.55) * 0.003;
  }
}

function updateRestArm(side) {
  const targets = runtime.restPose.ikTargets[side];
  const adapter = runtime.adapter;
  if (!adapter || !targets) return;
  const vector = (value) => new THREE.Vector3(
    Number(value?.[0]) || 0,
    Number(value?.[1]) || 0,
    Number(value?.[2]) || 0,
  );
  adapter.aimBoneAt(`${side}Shoulder`, `${side}UpperArm`, vector(targets.shoulder), 0.14);
  adapter.aimBoneAt(`${side}UpperArm`, `${side}LowerArm`, vector(targets.elbow), 0.18);
  adapter.aimBoneAt(`${side}LowerArm`, `${side}Hand`, vector(targets.wrist), 0.2);
  adapter.aimBoneAt(`${side}Hand`, `${side}MiddleProximal`, vector(targets.hand), 0.2);
}

function gazeTarget(raw, pointer) {
  const gaze = String(raw || 'user').toLowerCase();
  if (gaze === 'down') return { x: pointer.x * 0.25, y: 0.65 };
  if (gaze === 'left') return { x: -0.72, y: 0 };
  if (gaze === 'right') return { x: 0.72, y: 0 };
  if (gaze === 'away') return { x: -0.45, y: -0.08 };
  return { x: pointer.x * 0.52, y: pointer.y * 0.34 };
}

function speechSequence(text) {
  const sequence = [];
  const mapping = { a: 'aa', e: 'ee', i: 'ih', o: 'oh', u: 'ou' };
  for (const character of String(text || '').normalize('NFD').toLowerCase()) {
    if (mapping[character]) sequence.push(mapping[character]);
  }
  return sequence.length ? sequence : ['aa', 'ih', 'oh'];
}

function applyState(payload = {}) {
  runtime.state = {
    ...runtime.state,
    expression: payload.expression || 'neutral',
    expressionIntensity: Number(payload.expression_intensity ?? payload.expressionIntensity ?? 0.5),
    gesture: payload.gesture || 'idle',
    gaze: payload.gaze || 'user',
    posture: payload.posture || 'relaxed',
    attention: payload.attention || 'user',
    speech: payload.speech || {},
    lipSync: payload.lip_sync || payload.lipSync || {},
  };
  const speech = runtime.state.speech;
  const text = String(speech.text || payload.speech_text || '');
  const lipSyncEnabled = runtime.state.lipSync.enabled !== false;
  if (text && lipSyncEnabled) {
    const pause = clamp01(Number(speech.pause_before || 0)) * 1000;
    const rate = Math.max(0.65, Number(speech.speech_rate || 0.95));
    const durationHint = Number(runtime.state.lipSync.duration_hint_ms || 0);
    const duration = durationHint || Math.min(12000, Math.max(700, text.length * 48 / rate));
    runtime.speech = {
      active: true,
      mode: 'legacy',
      start: performance.now() + pause,
      end: performance.now() + pause + duration,
      bodyEnd: performance.now() + pause + duration,
      sequence: speechSequence(text),
      timeline: [],
    };
  } else {
    runtime.speech.active = false;
  }
}

function startLipSyncTimeline(payload = {}) {
  const allowed = new Set(['aa', 'ih', 'ou', 'ee', 'oh', 'neutral']);
  const timeline = Array.isArray(payload.timeline)
    ? payload.timeline
      .map((cue) => ({
        startMs: Math.max(0, Number(cue.start_ms ?? cue.startMs) || 0),
        endMs: Math.max(0, Number(cue.end_ms ?? cue.endMs) || 0),
        viseme: String(cue.viseme || 'neutral'),
        weight: clamp01(cue.weight),
      }))
      .filter((cue) => allowed.has(cue.viseme) && cue.endMs > cue.startMs)
    : [];
  const durationMs = Math.max(
    0,
    Number(payload.duration_ms ?? payload.durationMs) ||
      timeline.reduce((maximum, cue) => Math.max(maximum, cue.endMs), 0),
  );
  const offsetMs = Math.max(0, Number(payload.offset_ms ?? payload.offsetMs) || 0);
  const startDelayMs = Math.max(0, Number(payload.start_delay_ms ?? payload.startDelayMs) || 0);
  const bodyDurationRaw = payload.body_duration_ms ?? payload.bodyDurationMs;
  const bodyDurationNumeric = Number(bodyDurationRaw);
  const bodyDurationMs = bodyDurationRaw == null || !Number.isFinite(bodyDurationNumeric)
    ? durationMs
    : Math.max(0, bodyDurationNumeric);
  const start = performance.now() + startDelayMs - offsetMs;
  runtime.speech = {
    active: durationMs > offsetMs,
    mode: 'audio_timeline',
    start,
    end: start + durationMs,
    bodyEnd: start + bodyDurationMs,
    sequence: ['aa'],
    timeline,
  };
}

function stopSpeaking() {
  runtime.speech.active = false;
  runtime.speech.timeline = [];
  runtime.speech.mode = 'audio_timeline';
}

function applyEnvironment(payload = {}) {
  runtime.environment = {
    warmLight: Number(payload.warm_light ?? runtime.environment.warmLight),
    coolLight: Number(payload.cool_light ?? runtime.environment.coolLight),
    lampGlow: Number(payload.lamp_glow ?? runtime.environment.lampGlow),
  };
  keyLight.intensity = 0.72 + runtime.environment.warmLight * 0.82;
  rimLight.intensity = 0.58 + runtime.environment.coolLight * 0.88;
  warmFill.intensity = 1.15 + runtime.environment.lampGlow * 2.35;
  renderer.toneMappingExposure = 1.15 + runtime.environment.coolLight * 0.08 + runtime.environment.warmLight * 0.08;
}

function setPointer(x, y) {
  runtime.pointer.x = THREE.MathUtils.clamp(Number(x) || 0, -1, 1);
  runtime.pointer.y = THREE.MathUtils.clamp(Number(y) || 0, -1, 1);
}

function setCameraMode(mode) {
  const selected = String(mode || '').toLowerCase();
  if (!runtime.camera.modes[selected]) return false;
  runtime.camera.mode = selected;
  return true;
}

function setRestPoseEnabled(enabled) {
  runtime.restPose.enabled = Boolean(enabled);
}

function setMotionSample(payload = null) {
  if (!payload || payload.enabled === false) {
    runtime.motionSample = null;
    return;
  }
  runtime.motionSample = {
    breath: THREE.MathUtils.clamp(Number(payload.breath) || 0, -1, 1),
    sway: THREE.MathUtils.clamp(Number(payload.sway) || 0, -1, 1),
  };
}

function updateCamera(delta) {
  const mode = runtime.camera.modes[runtime.camera.mode] || runtime.camera.modes.environment;
  const blend = 1 - Math.exp(-Math.max(0.001, delta) * 4.8);
  camera.position.lerp(mode.position, blend);
  runtime.camera.target.lerp(mode.target, blend);
  camera.fov = THREE.MathUtils.lerp(camera.fov, mode.fov, blend);
  camera.lookAt(runtime.camera.target);
  camera.updateProjectionMatrix();
}

function resize() {
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const requiredWidth = Math.floor(width * pixelRatio);
  const requiredHeight = Math.floor(height * pixelRatio);
  if (canvas.width !== requiredWidth || canvas.height !== requiredHeight) {
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

function animate(timestampMs = 0) {
  requestAnimationFrame(animate);
  if (profiler.targetFps > 0) {
    const interval = 1000 / profiler.targetFps;
    if (timestampMs - profiler.lastFrameMs < interval) return;
    profiler.lastFrameMs = timestampMs;
  }
  const delta = Math.min(0.05, clock.getDelta());
  const time = clock.elapsedTime;
  resize();
  updateExpressions(time);
  updatePose(time, delta);
  updateCamera(delta);
  if (postPipeline) {
    postPipeline.render();
  } else {
    renderer.render(scene, camera);
  }
  if (profiler.lastRenderedAt > 0) {
    profiler.frameIntervals.push(timestampMs - profiler.lastRenderedAt);
    if (profiler.frameIntervals.length > 240) profiler.frameIntervals.shift();
  }
  profiler.lastRenderedAt = timestampMs;
  runtime.frames += 1;
}

async function loadRestPoseProfile() {
  try {
    const response = await fetch('/pose.json', { cache: 'no-store' });
    if (!response.ok) return;
    const profile = await response.json();
    if (!profile || typeof profile !== 'object' || typeof profile.bones !== 'object') return;
    runtime.restPose.name = String(profile.name || '');
    runtime.restPose.bones = profile.bones;
    runtime.restPose.ikTargets = profile.ik_targets_vrm_local || {};
    for (const [name, specification] of Object.entries(profile.camera_modes || {})) {
      const position = specification?.camera_position;
      const target = specification?.look_at;
      const fov = Number(specification?.fov);
      if (!Array.isArray(position) || position.length !== 3) continue;
      if (!Array.isArray(target) || target.length !== 3) continue;
      if (!Number.isFinite(fov)) continue;
      runtime.camera.modes[name] = {
        position: new THREE.Vector3(...position.map(Number)),
        target: new THREE.Vector3(...target.map(Number)),
        fov,
      };
    }
  } catch (_error) {
    // A VRM without a companion rest-pose profile remains valid and neutral.
  }
}

async function loadModel() {
  const modelUrl = new URLSearchParams(window.location.search).get('model');
  if (!modelUrl) throw new Error('model_url_missing');
  await loadRestPoseProfile();
  const loader = new GLTFLoader();
  loader.register((parser) => new MToonMaterialLoaderPlugin(parser));
  const gltf = await loader.loadAsync(modelUrl);
  const adapter = new VrmAdapter(gltf);
  await adapter.initialize();
  configureModel(gltf.scene);
  stage.add(gltf.scene);
  runtime.adapter = adapter;
  runtime.model = gltf.scene;
  runtime.ready = true;
  loading.classList.add('ready');
}

window.AnydesAvatar = {
  applyState,
  applyEnvironment,
  setPointer,
  setCameraMode,
  setRestPoseEnabled,
  setMotionSample,
  setProfilerFps(value) {
    const numeric = Number(value);
    profiler.targetFps = Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
    profiler.lastFrameMs = 0;
  },
  startLipSyncTimeline,
  stopSpeaking,
  status() {
    return {
      ready: runtime.ready,
      error: runtime.error,
      frames: runtime.frames,
      modelVisible: Boolean(runtime.model?.visible),
      expressionCount: runtime.adapter?.expressions?.size || 0,
      mouthWeights: Object.fromEntries(['aa', 'ih', 'ee', 'ou', 'oh'].map(name => [name, runtime.adapter?.expressionValues.get(name) || 0])),
      boneCount: runtime.adapter?.bones?.size || 0,
      restPose: runtime.restPose.name,
      restPoseEnabled: runtime.restPose.enabled,
      cameraMode: runtime.camera.mode,
      motionSample: runtime.motionSample,
      lastMotion: runtime.lastMotion,
      eyeWhiteMaterials: runtime.eyeWhiteMaterials,
      mtoonMaterials: runtime.mtoonMaterials,
      renderProfile: HQ_PROFILE ? 'hq_efficient' : 'official',
      hqStage: HQ_PROFILE ? HQ_STAGE : 'disabled',
      hqMaterials: { ...runtime.hqMaterials },
      profilerTargetFps: profiler.targetFps,
      frameTime: summarizeSamples(profiler.frameIntervals),
      renderer: {
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        pixelRatio: renderer.getPixelRatio(),
        geometries: renderer.info.memory.geometries,
        textures: renderer.info.memory.textures,
        calls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
        postprocess: Boolean(postPipeline),
        postprocessSamples: postPipeline?.target?.samples || 0,
      },
    };
  },
};

loadModel().catch((error) => {
  runtime.error = String(error?.message || error || 'avatar_load_failed');
  loading.classList.add('ready');
});
animate();

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function summarizeSamples(values) {
  if (!values.length) return { samples: 0, p50Ms: 0, p95Ms: 0, fps: 0 };
  const ordered = [...values].sort((left, right) => left - right);
  const percentile = (fraction) => ordered[Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(ordered.length * fraction) - 1),
  )];
  const p50Ms = percentile(0.5);
  const p95Ms = percentile(0.95);
  return {
    samples: values.length,
    p50Ms: Number(p50Ms.toFixed(3)),
    p95Ms: Number(p95Ms.toFixed(3)),
    fps: Number((1000 / Math.max(0.001, p50Ms)).toFixed(2)),
  };
}
