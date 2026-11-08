import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import './style.css';

const $ = (selector) => document.querySelector(selector);
const video = $('#webcam');
const viewport = $('#viewport');
const canvas = $('#ar-canvas');
const cameraButton = $('#camera-button');
const suitButton = $('#suit-button');
const systemStatus = $('#system-status');
const errorBanner = $('#error-banner');
const loader = new GLTFLoader();
const clock = new THREE.Clock();
const OPTIONAL_TRACKING_INTERVAL_MS = 120;

const visionConfig = {
  wasm: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
  pose: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  face: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  hands: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
};

const state = {
  pose: null, face: null, hands: null, stream: null, running: false, initializing: false, raf: 0,
  cameraSession: 0, faceLoadState: 'offline', handLoadState: 'offline',
  lastVideoTime: -1, lastFaceDetection: 0, lastHandDetection: 0, lastFpsTime: performance.now(), frameCount: 0,
  suitStartedAt: 0, suitProgress: 0, fistWasClosed: false, mixer: null,
  root: null, customRoot: null, parts: {},
  poseVisibleParts: new Set(), poseDebugGroup: null, poseDebugDots: [], poseDebugGeometry: null,
  posePoints: [], facePoint: null, faceRoll: 0, faceYaw: 0, facePitch: 0, faceTransform: null, handsDetected: false,
  smoothedPose: null, lastPoseTime: 0, poseConfidence: 0,
};

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
camera.position.set(0, 0, 20);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.16;
renderer.setClearColor(0x000000, 0);

scene.add(new THREE.HemisphereLight(0xbdeeff, 0x321413, 2.2));
const keyLight = new THREE.DirectionalLight(0xffe6ce, 3.2);
keyLight.position.set(-2, 4, 8);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x11dfff, 14, 8);
rimLight.position.set(0, 0.2, 4);
scene.add(rimLight);
const pmremGenerator = new THREE.PMREMGenerator(renderer);
const roomEnvironment = new RoomEnvironment();
scene.environment = pmremGenerator.fromScene(roomEnvironment, 0.04).texture;
scene.environmentIntensity = 1.15;
roomEnvironment.dispose();
pmremGenerator.dispose();

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.78, 0.42, 0.12);
composer.addPass(bloom);
const ARMOR = new THREE.MeshPhysicalMaterial({ color: 0x26343b, metalness: 0.88, roughness: 0.2, clearcoat: 0.9, clearcoatRoughness: 0.14 });
const TRIM = new THREE.MeshPhysicalMaterial({ color: 0x101a20, metalness: 0.9, roughness: 0.23, clearcoat: 0.65 });
const DARK = new THREE.MeshStandardMaterial({ color: 0x11191f, metalness: 0.82, roughness: 0.3 });
const GLOW = new THREE.MeshStandardMaterial({ color: 0x8bfaff, emissive: 0x00caff, emissiveIntensity: 3.5, metalness: 0.08, roughness: 0.15 });

function installDissolve(material) {
  const instance = material.clone();
  instance.userData.suitUniforms = { progress: { value: -0.12 } };
  instance.onBeforeCompile = (shader) => {
    shader.uniforms.uSuitProgress = instance.userData.suitUniforms.progress;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSuitPosition;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSuitPosition = position;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vSuitPosition;
        uniform float uSuitProgress;
        float suitNoise(vec3 p) {
          p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
          p *= 17.0;
          return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }`)
      .replace('#include <opaque_fragment>', `float suitField = vSuitPosition.y * 0.56 + suitNoise(vSuitPosition * 7.0) * 0.42;
        float suitEdge = 1.0 - smoothstep(0.0, 0.12, abs(suitField - uSuitProgress));
        if (suitField > uSuitProgress) discard;
        outgoingLight += vec3(0.03, 0.62, 0.88) * suitEdge * 2.6;
        #include <opaque_fragment>`);
  };
  instance.customProgramCacheKey = () => 'nanotech-dissolve-v1';
  return instance;
}

function createSegment(name, material, radius = 0.09, radiusTop = radius) {
  const geometry = new THREE.CylinderGeometry(radiusTop, radius, 1, 12, 1, false);
  geometry.translate(0, 0.5, 0);
  const mesh = new THREE.Mesh(geometry, installDissolve(material));
  mesh.name = name;
  mesh.visible = false;
  mesh.castShadow = false;
  mesh.frustumCulled = false;
  state.root.add(mesh);
  state.parts[name] = mesh;
  return mesh;
}

function createSphere(name, material, scale) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 16), installDissolve(material));
  mesh.name = name;
  mesh.scale.set(...scale);
  mesh.visible = false;
  state.root.add(mesh);
  state.parts[name] = mesh;
  return mesh;
}

function createProceduralSuit() {
  state.root = new THREE.Group();
  state.root.name = 'procedural-suit';
  scene.add(state.root);
  createSphere('chest', ARMOR, [0.24, 0.3, 0.095]);
  createSphere('abdomen', DARK, [0.18, 0.2, 0.085]);
  createSphere('shoulderL', TRIM, [0.13, 0.13, 0.1]);
  createSphere('shoulderR', TRIM, [0.13, 0.13, 0.1]);
  createSphere('hipL', TRIM, [0.115, 0.12, 0.09]);
  createSphere('hipR', TRIM, [0.115, 0.12, 0.09]);
  createSphere('helmet', ARMOR, [0.15, 0.19, 0.13]);
  createSphere('handL', TRIM, [0.075, 0.085, 0.075]);
  createSphere('handR', TRIM, [0.075, 0.085, 0.075]);
  createSphere('kneeL', TRIM, [0.095, 0.1, 0.085]);
  createSphere('kneeR', TRIM, [0.095, 0.1, 0.085]);
  createSphere('bootL', DARK, [0.15, 0.085, 0.1]);
  createSphere('bootR', DARK, [0.15, 0.085, 0.1]);
  createSegment('upperArmL', ARMOR, 0.085, 0.105);
  createSegment('upperArmR', ARMOR, 0.085, 0.105);
  createSegment('forearmL', ARMOR, 0.07, 0.09);
  createSegment('forearmR', ARMOR, 0.07, 0.09);
  createSegment('thighL', ARMOR, 0.105, 0.125);
  createSegment('thighR', ARMOR, 0.105, 0.125);
  createSegment('shinL', ARMOR, 0.075, 0.095);
  createSegment('shinR', ARMOR, 0.075, 0.095);
  createSphere('arcReactor', GLOW, [0.072, 0.072, 0.032]);
  createSphere('eyeL', GLOW, [0.044, 0.018, 0.027]);
  createSphere('eyeR', GLOW, [0.044, 0.018, 0.027]);
  state.root.visible = false;
}

createProceduralSuit();

const poseConnections = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];

function createPoseDebug() {
  const group = new THREE.Group();
  group.name = 'pose-detection-debug';
  const material = new THREE.MeshBasicMaterial({ color: 0x49efff, depthTest: false, depthWrite: false });
  const geometry = new THREE.SphereGeometry(0.022, 8, 6);
  state.poseDebugDots = Array.from({ length: 33 }, (_, index) => {
    const dot = new THREE.Mesh(geometry, material);
    dot.name = `pose-landmark-${index}`;
    dot.renderOrder = 20;
    group.add(dot);
    return dot;
  });
  state.poseDebugGeometry = new THREE.BufferGeometry();
  state.poseDebugGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(poseConnections.length * 6), 3));
  const lines = new THREE.LineSegments(
    state.poseDebugGeometry,
    new THREE.LineBasicMaterial({ color: 0x45e7f2, transparent: true, opacity: 0.8, depthTest: false, depthWrite: false }),
  );
  lines.renderOrder = 19;
  group.add(lines);
  group.visible = false;
  scene.add(group);
  state.poseDebugGroup = group;
}

createPoseDebug();

function resize() {
  const { width, height } = viewport.getBoundingClientRect();
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  composer.setSize(width, height);
  camera.left = -width / height;
  camera.right = width / height;
  camera.top = 1;
  camera.bottom = -1;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
resize();

function pointToWorld(landmark) {
  const { width, height } = viewport.getBoundingClientRect();
  const aspect = width / height;
  const sourceAspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : aspect;
  const visibleWidth = Math.min(1, aspect / sourceAspect);
  const visibleHeight = Math.min(1, sourceAspect / aspect);
  const visibleX = (landmark.x - (1 - visibleWidth) * 0.5) / visibleWidth;
  const visibleY = (landmark.y - (1 - visibleHeight) * 0.5) / visibleHeight;
  return new THREE.Vector3((0.5 - visibleX) * 2 * aspect, (0.5 - visibleY) * 2, 3.2 - (landmark.z || 0) * 0.7);
}

function midpoint(a, b) { return a.clone().add(b).multiplyScalar(0.5); }

function smoothLandmarks(landmarks, now) {
  const alpha = state.smoothedPose
    ? THREE.MathUtils.clamp((now - state.lastPoseTime) / 55, 0.16, 0.76)
    : 1;
  state.lastPoseTime = now;
  if (!state.smoothedPose || state.smoothedPose.length !== landmarks.length) {
    state.smoothedPose = landmarks.map((point) => ({ ...point }));
    return state.smoothedPose;
  }
  for (let i = 0; i < landmarks.length; i += 1) {
    const source = landmarks[i];
    const previous = state.smoothedPose[i];
    previous.x += (source.x - previous.x) * alpha;
    previous.y += (source.y - previous.y) * alpha;
    previous.z += (source.z - previous.z) * alpha;
    previous.visibility = source.visibility;
  }
  return state.smoothedPose;
}

function placeSphere(name, position, scale = null, rotationZ = 0) {
  const mesh = state.parts[name];
  if (!mesh || !position) return;
  mesh.position.copy(position);
  mesh.rotation.set(0, 0, rotationZ);
  if (scale) mesh.scale.set(...scale);
  state.poseVisibleParts.add(name);
}

function placeSegment(name, start, end, radius = 0.08) {
  const mesh = state.parts[name];
  if (!mesh || !start || !end) return;
  const direction = end.clone().sub(start);
  const length = Math.max(direction.length(), 0.035);
  const geometryRadius = Math.max(mesh.geometry.parameters.radiusTop, mesh.geometry.parameters.radiusBottom);
  const radialScale = radius / geometryRadius;
  mesh.position.copy(start);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  mesh.scale.set(radialScale, length, radialScale);
  state.poseVisibleParts.add(name);
}

function mapLandmarks(landmarks, names) {
  return Object.fromEntries(Object.entries(names).map(([name, index]) => [name, landmarks[index] ? pointToWorld(landmarks[index]) : null]));
}

function updateProceduralSuit() {
  if (!state.posePoints.length) {
    state.poseVisibleParts.clear();
    return;
  }
  const p = state.posePoints;
  const point = (index) => p[index] && landmarkVisibility(p[index]) >= 0.25 ? pointToWorld(p[index]) : null;
  const l = {
    nose: point(0), leftShoulder: point(11), rightShoulder: point(12),
    leftElbow: point(13), rightElbow: point(14), leftWrist: point(15), rightWrist: point(16),
    leftHip: point(23), rightHip: point(24), leftKnee: point(25), rightKnee: point(26),
    leftAnkle: point(27), rightAnkle: point(28), leftToe: point(31), rightToe: point(32),
  };
  state.poseVisibleParts.clear();
  const shouldersVisible = l.leftShoulder && l.rightShoulder;
  const hipsVisible = l.leftHip && l.rightHip;
  const shoulderSpan = shouldersVisible
    ? Math.max(l.leftShoulder.distanceTo(l.rightShoulder), 0.2)
    : hipsVisible ? Math.max(l.leftHip.distanceTo(l.rightHip) * 1.35, 0.2) : 0.3;
  let chest = null;
  let bodyHeight = shoulderSpan * 1.15;

  if (shouldersVisible && hipsVisible) {
    const shoulders = midpoint(l.leftShoulder, l.rightShoulder);
    const hips = midpoint(l.leftHip, l.rightHip);
    bodyHeight = Math.max(shoulders.distanceTo(hips), 0.3);
    const torsoAngle = Math.atan2(l.rightShoulder.y - l.leftShoulder.y, l.rightShoulder.x - l.leftShoulder.x);
    chest = midpoint(shoulders, hips);
    placeSphere('chest', chest, [shoulderSpan * 0.48, bodyHeight * 0.43, shoulderSpan * 0.2], torsoAngle);
    placeSphere('abdomen', midpoint(chest, hips), [shoulderSpan * 0.34, bodyHeight * 0.23, shoulderSpan * 0.16], torsoAngle);
    rimLight.position.copy(chest).add(new THREE.Vector3(0, 0, 1));
    placeSphere('arcReactor', chest.clone().add(new THREE.Vector3(0, -bodyHeight * 0.02, 0.2)), [shoulderSpan * 0.15, shoulderSpan * 0.15, shoulderSpan * 0.075]);
  }

  if (l.leftShoulder) placeSphere('shoulderL', l.leftShoulder, [shoulderSpan * 0.23, shoulderSpan * 0.23, shoulderSpan * 0.19]);
  if (l.rightShoulder) placeSphere('shoulderR', l.rightShoulder, [shoulderSpan * 0.23, shoulderSpan * 0.23, shoulderSpan * 0.19]);
  if (l.leftHip) placeSphere('hipL', l.leftHip, [shoulderSpan * 0.19, shoulderSpan * 0.19, shoulderSpan * 0.15]);
  if (l.rightHip) placeSphere('hipR', l.rightHip, [shoulderSpan * 0.19, shoulderSpan * 0.19, shoulderSpan * 0.15]);

  placeSegment('upperArmL', l.leftShoulder, l.leftElbow, shoulderSpan * 0.13);
  placeSegment('upperArmR', l.rightShoulder, l.rightElbow, shoulderSpan * 0.13);
  placeSegment('forearmL', l.leftElbow, l.leftWrist, shoulderSpan * 0.11);
  placeSegment('forearmR', l.rightElbow, l.rightWrist, shoulderSpan * 0.11);
  if (l.leftWrist) placeSphere('handL', l.leftWrist, [shoulderSpan * 0.16, shoulderSpan * 0.18, shoulderSpan * 0.15]);
  if (l.rightWrist) placeSphere('handR', l.rightWrist, [shoulderSpan * 0.16, shoulderSpan * 0.18, shoulderSpan * 0.15]);

  placeSegment('thighL', l.leftHip, l.leftKnee, shoulderSpan * 0.16);
  placeSegment('thighR', l.rightHip, l.rightKnee, shoulderSpan * 0.16);
  if (l.leftKnee) placeSphere('kneeL', l.leftKnee, [shoulderSpan * 0.18, shoulderSpan * 0.19, shoulderSpan * 0.15]);
  if (l.rightKnee) placeSphere('kneeR', l.rightKnee, [shoulderSpan * 0.18, shoulderSpan * 0.19, shoulderSpan * 0.15]);
  placeSegment('shinL', l.leftKnee, l.leftAnkle, shoulderSpan * 0.12);
  placeSegment('shinR', l.rightKnee, l.rightAnkle, shoulderSpan * 0.12);
  for (const side of ['left', 'right']) {
    const ankle = l[`${side}Ankle`];
    if (!ankle) continue;
    const toe = l[`${side}Toe`];
    const direction = toe ? toe.clone().sub(ankle) : new THREE.Vector3(0, -0.08, 0);
    const bootCenter = ankle.clone().add(direction.clone().multiplyScalar(0.5));
    const bootAngle = Math.atan2(direction.y, direction.x);
    placeSphere(`boot${side === 'left' ? 'L' : 'R'}`, bootCenter, [shoulderSpan * 0.3, shoulderSpan * 0.15, shoulderSpan * 0.18], bootAngle);
  }

  const head = state.facePoint ?? (l.nose && l.nose.clone().add(new THREE.Vector3(0, bodyHeight * 0.14, 0)));
  if (head) {
    placeSphere('helmet', head, [shoulderSpan * 0.31, shoulderSpan * 0.38, shoulderSpan * 0.25], state.faceRoll);
    state.parts.helmet.rotation.set(state.facePitch, state.faceYaw, state.faceRoll);
    const eyeOffset = shoulderSpan * 0.1;
    const eyeY = head.y + shoulderSpan * 0.025;
    const eyeZ = head.z + shoulderSpan * 0.22;
    placeSphere('eyeL', new THREE.Vector3(head.x - eyeOffset, eyeY, eyeZ), [shoulderSpan * 0.09, shoulderSpan * 0.035, shoulderSpan * 0.025], state.faceRoll);
    placeSphere('eyeR', new THREE.Vector3(head.x + eyeOffset, eyeY, eyeZ), [shoulderSpan * 0.09, shoulderSpan * 0.035, shoulderSpan * 0.025], state.faceRoll);
  }

  state.root.visible = true;
}

function updatePoseDebug() {
  if (!state.poseDebugGroup?.visible) return;
  const positions = state.poseDebugGeometry.attributes.position;
  let connectionCount = 0;
  for (const [startIndex, endIndex] of poseConnections) {
    const start = state.posePoints[startIndex];
    const end = state.posePoints[endIndex];
    if (!start || !end || landmarkVisibility(start) < 0.25 || landmarkVisibility(end) < 0.25) continue;
    const a = pointToWorld(start);
    const b = pointToWorld(end);
    positions.setXYZ(connectionCount * 2, a.x, a.y, 4.5);
    positions.setXYZ(connectionCount * 2 + 1, b.x, b.y, 4.5);
    connectionCount += 1;
  }
  positions.needsUpdate = true;
  state.poseDebugGeometry.setDrawRange(0, connectionCount * 2);
  state.poseDebugDots.forEach((dot, index) => {
    const landmark = state.posePoints[index];
    const visible = landmark && landmarkVisibility(landmark) >= 0.25;
    dot.visible = Boolean(visible);
    if (visible) {
      const position = pointToWorld(landmark);
      dot.position.set(position.x, position.y, 4.55);
      const size = THREE.MathUtils.clamp(0.018 + (landmarkVisibility(landmark) * 0.025), 0.018, 0.043);
      dot.scale.setScalar(size / 0.022);
    }
  });
}

function partProgress(name, progress) {
  if (name === 'arcReactor') return Math.min(1, progress * 5.3);
  if (name === 'chest' || name === 'abdomen' || name.startsWith('shoulder') || name.startsWith('hip')) return THREE.MathUtils.clamp((progress - 0.12) / 0.38, 0, 1);
  if (name.startsWith('upperArm') || name.startsWith('forearm') || name.startsWith('hand')) return THREE.MathUtils.clamp((progress - 0.38) / 0.3, 0, 1);
  if (name.startsWith('thigh') || name.startsWith('shin') || name.startsWith('knee') || name.startsWith('boot')) return THREE.MathUtils.clamp((progress - 0.58) / 0.3, 0, 1);
  return THREE.MathUtils.clamp((progress - 0.84) / 0.16, 0, 1);
}

function updateSuitProgress(now) {
  if (state.suitStartedAt) {
    state.suitProgress = Math.min(1, (now - state.suitStartedAt) / 4800);
    if (state.suitProgress >= 1) state.suitStartedAt = 0;
  }
  for (const [name, part] of Object.entries(state.parts)) {
    part.visible = state.suitProgress > 0 && state.poseVisibleParts.has(name);
    const material = part.material;
    material.userData.suitUniforms.progress.value = partProgress(name, state.suitProgress) * 1.58 - 0.08;
  }
  const poseTracked = hasReliablePose(state.posePoints);
  state.root.visible = state.suitProgress > 0 && poseTracked && $('#model-select').value === 'procedural';
  if (state.customRoot) {
    state.customRoot.visible = state.suitProgress > 0 && poseTracked && $('#model-select').value === 'custom';
    state.customRoot.traverse((node) => {
      if (node.isMesh && node.material?.userData.suitUniforms) {
        node.material.userData.suitUniforms.progress.value = state.suitProgress * 1.6 - 0.08;
      }
    });
  }
  const value = Math.round(state.suitProgress * 100);
  $('#suit-progress-bar').style.width = `${value}%`;
  $('#suit-progress-value').textContent = `${value}%`;
  $('#suit-progress-label').textContent = value === 0 ? 'PRÊT À L’ACTIVATION' : value < 100 ? 'MATÉRIALISATION EN COURS' : 'ARMURE DÉPLOYÉE';
}

function updateFace(faceLandmarks) {
  state.facePoint = null;
  if (!faceLandmarks?.length) return;
  const face = faceLandmarks[0];
  const leftEye = face[33];
  const rightEye = face[263];
  const nose = face[1];
  if (!leftEye || !rightEye || !nose) return;
  const left = pointToWorld(leftEye);
  const right = pointToWorld(rightEye);
  const nosePoint = pointToWorld(nose);
  state.faceRoll = Math.atan2(right.y - left.y, right.x - left.x);
  state.faceYaw = THREE.MathUtils.clamp((nosePoint.x - (left.x + right.x) * 0.5) * 5, -0.5, 0.5);
  const forehead = face[10];
  const chin = face[152];
  if (forehead && chin) {
    const faceHeight = Math.max(chin.y - forehead.y, 0.001);
    const eyeLine = (leftEye.y + rightEye.y) * 0.5;
    state.facePitch = THREE.MathUtils.clamp(((nose.y - eyeLine) / faceHeight - 0.2) * 3, -0.55, 0.55);
    state.facePoint = midpoint(pointToWorld(forehead), pointToWorld(chin));
  } else {
    state.facePoint = nosePoint;
  }
  if (state.faceTransform?.length === 16) {
    const rotation = new THREE.Quaternion();
    new THREE.Matrix4().fromArray(state.faceTransform).decompose(new THREE.Vector3(), rotation, new THREE.Vector3());
    const euler = new THREE.Euler().setFromQuaternion(rotation, 'YXZ');
    state.facePitch = euler.x;
    state.faceYaw = euler.y;
    state.faceRoll = euler.z;
  }
}

function setMetric(id, active, onText = 'EN LIGNE', offText = 'HORS LIGNE') {
  const element = $(id);
  element.textContent = active ? onText : offText;
  element.classList.toggle('active', active);
}

function updateStatus() {
  const poseVisible = hasReliablePose(state.posePoints);
  setMetric('#pose-status', poseVisible, 'CORPS DÉTECTÉ');
  setMetric(
    '#tracking-confidence',
    state.poseConfidence >= 0.45,
    `${Math.round(state.poseConfidence * 100)}%`,
    'FAIBLE',
  );
  setMetric(
    '#face-status',
    Boolean(state.facePoint),
    'VISAGE DÉTECTÉ',
    state.faceLoadState === 'loading' ? 'CHARGEMENT…' : state.faceLoadState === 'error' ? 'INDISPONIBLE' : 'NON DÉTECTÉ',
  );
  setMetric(
    '#hand-status',
    state.handsDetected,
    state.fistWasClosed ? 'POING FERMÉ' : 'MAIN DÉTECTÉE',
    state.handLoadState === 'loading' ? 'CHARGEMENT…' : state.handLoadState === 'error' ? 'INDISPONIBLE' : 'NON DÉTECTÉE',
  );
  $('#tracking-label').textContent = poseVisible ? '/ CORPS VERROUILLÉ' : '/ RECHERCHE DU CORPS';
  const hint = $('#tracking-hint');
  hint.hidden = !state.running || poseVisible;
  hint.textContent = state.posePoints.length
    ? 'Corps partiellement détecté. Placez-vous face à la caméra et cadrez au minimum la tête, les épaules et le bassin.'
    : 'Aucun corps détecté. Placez-vous face à la webcam, bien éclairé, et reculez pour cadrer votre corps.';
}

function isFist(handLandmarks) {
  const tips = [8, 12, 16, 20];
  const joints = [6, 10, 14, 18];
  return tips.every((tip, index) => {
    const tipPoint = handLandmarks[tip];
    const jointPoint = handLandmarks[joints[index]];
    const wrist = handLandmarks[0];
    return Math.hypot(tipPoint.x - wrist.x, tipPoint.y - wrist.y) <
      Math.hypot(jointPoint.x - wrist.x, jointPoint.y - wrist.y) * 1.08;
  });
}

function landmarkVisibility(landmark) {
  return landmark?.visibility ?? landmark?.presence ?? 1;
}

function hasReliablePose(landmarks) {
  if (landmarks.length <= 28) return false;
  const torso = [11, 12, 23, 24];
  return torso.every((index) => landmarkVisibility(landmarks[index]) >= 0.35);
}

function getPoseConfidence(landmarks) {
  if (landmarks.length <= 28) return 0;
  const keyJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  return keyJoints.reduce((total, index) => total + landmarkVisibility(landmarks[index]), 0) / keyJoints.length;
}

function renderFrame(now) {
  if (!state.running) return;
  state.raf = requestAnimationFrame(renderFrame);

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const timestamp = performance.now();
    try {
      const poseResult = state.pose.detectForVideo(video, timestamp);
      const rawPose = poseResult.landmarks?.[0] ?? [];
      state.posePoints = rawPose.length ? smoothLandmarks(rawPose, timestamp) : [];
      state.poseConfidence = getPoseConfidence(state.posePoints);
      updatePoseDebug();
      if (state.face && timestamp - state.lastFaceDetection >= OPTIONAL_TRACKING_INTERVAL_MS) {
        state.lastFaceDetection = timestamp;
        try {
          const faceResult = state.face.detectForVideo(video, timestamp);
          state.faceTransform = faceResult.facialTransformationMatrixes?.[0]?.data ?? null;
          updateFace(faceResult.faceLandmarks?.[0] ? faceResult.faceLandmarks : null);
        } catch (error) {
          console.error('Erreur du suivi du visage; le suivi du corps continue.', error);
          state.face.close();
          state.face = null;
          state.faceLoadState = 'error';
          updateFace(null);
        }
      } else if (!state.face) {
        state.facePoint = null;
        state.faceTransform = null;
      }
      if (state.hands && timestamp - state.lastHandDetection >= OPTIONAL_TRACKING_INTERVAL_MS) {
        state.lastHandDetection = timestamp;
        try {
          const handResult = state.hands.detectForVideo(video, timestamp);
          const detectedHands = handResult.landmarks ?? [];
          state.handsDetected = detectedHands.length > 0;
          state.fistWasClosed = detectedHands.some(isFist);
        } catch (error) {
          console.error('Erreur du suivi des mains; le suivi du corps continue.', error);
          state.hands.close();
          state.hands = null;
          state.handLoadState = 'error';
          state.handsDetected = false;
          state.fistWasClosed = false;
        }
      } else if (!state.hands) {
        state.handsDetected = false;
        state.fistWasClosed = false;
      }
      if (hasReliablePose(state.posePoints)) updateProceduralSuit();
      else {
        state.poseVisibleParts.clear();
        state.root.visible = false;
        if (state.customRoot) state.customRoot.visible = false;
      }

      if (state.fistWasClosed && state.suitProgress >= 0.98 && !state.fistTriggered) {
        state.fistTriggered = true;
        startSuitUp();
      } else if (!state.fistWasClosed) {
        state.fistTriggered = false;
      }
      if (state.customRoot && hasReliablePose(state.posePoints)) positionCustomModel();
      updateStatus();
    } catch (error) {
      console.error('Échec du suivi de pose MediaPipe.', error);
      const detail = error instanceof Error ? error.message : 'erreur de suivi non détaillée';
      showError(`Échec du suivi du corps : ${detail}`);
      stopCamera();
      return;
    }
  }

  if (state.mixer) state.mixer.update(clock.getDelta());
  if (state.customRoot) updateCustomSkeleton();
  updateSuitProgress(now);
  composer.render();
  state.frameCount += 1;
  if (now - state.lastFpsTime > 1000) {
    $('#fps-status').textContent = `${Math.round(state.frameCount * 1000 / (now - state.lastFpsTime))} FPS`;
    state.frameCount = 0;
    state.lastFpsTime = now;
  }
}

function startSuitUp() {
  state.suitProgress = 0;
  state.suitStartedAt = performance.now();
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
  systemStatus.textContent = 'ERREUR SYSTÈME';
  $('.state-dot').classList.remove('active');
}

function taskOptions(modelAssetPath, delegate, countKey, count) {
  return {
    baseOptions: { modelAssetPath, delegate },
    runningMode: 'VIDEO',
    [countKey]: count,
  };
}

async function createTaskWithFallback(fileset, Task, modelAssetPath, countKey, count, label, configure = () => {}) {
  const create = (delegate) => {
    const options = taskOptions(modelAssetPath, delegate, countKey, count);
    configure(options);
    return Task.createFromOptions(fileset, options);
  };
  try {
    return await create('GPU');
  } catch (gpuError) {
    console.warn(`MediaPipe ${label} : délégué GPU indisponible, essai CPU.`, gpuError);
    return create('CPU');
  }
}

async function loadOptionalLandmarkers(fileset, FaceLandmarker, HandLandmarker, session) {
  state.faceLoadState = 'loading';
  state.handLoadState = 'loading';
  updateStatus();
  const results = await Promise.allSettled([
    createTaskWithFallback(fileset, FaceLandmarker, visionConfig.face, 'numFaces', 1, 'visage', (options) => {
      options.outputFacialTransformationMatrixes = true;
    }),
    createTaskWithFallback(fileset, HandLandmarker, visionConfig.hands, 'numHands', 2, 'mains'),
  ]);
  if (session !== state.cameraSession) {
    results.forEach((result) => {
      if (result.status === 'fulfilled') result.value.close();
    });
    return;
  }
  [state.face, state.hands] = results.map((result) => result.status === 'fulfilled' ? result.value : null);
  state.faceLoadState = results[0].status === 'fulfilled' ? 'ready' : 'error';
  state.handLoadState = results[1].status === 'fulfilled' ? 'ready' : 'error';
  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      console.error(`Chargement MediaPipe ${index === 0 ? 'visage' : 'mains'} impossible.`, result.reason);
    }
  });
  updateStatus();
}

async function createLandmarkers(session) {
  const { FilesetResolver, PoseLandmarker, FaceLandmarker, HandLandmarker } = await import('@mediapipe/tasks-vision');
  const fileset = await FilesetResolver.forVisionTasks(visionConfig.wasm);
  if (session !== state.cameraSession) return false;
  const pose = await createTaskWithFallback(fileset, PoseLandmarker, visionConfig.pose, 'numPoses', 1, 'pose');
  if (session !== state.cameraSession) {
    pose.close();
    return false;
  }
  state.pose = pose;
  void loadOptionalLandmarkers(fileset, FaceLandmarker, HandLandmarker, session);
  return true;
}

async function startCamera() {
  errorBanner.hidden = true;
  cameraButton.textContent = 'CHARGEMENT DES MODÈLES…';
  systemStatus.textContent = 'INITIALISATION DU SYSTÈME';
  let initializationStage = 'webcam';
  const session = ++state.cameraSession;
  state.initializing = true;
  cameraButton.disabled = false;
  cameraButton.textContent = 'ANNULER LE CHARGEMENT';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Cette page doit être ouverte sur localhost ou HTTPS pour accéder à la webcam.');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } });
    if (session !== state.cameraSession) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    video.srcObject = state.stream;
    await video.play();
    if (session !== state.cameraSession) return;
    initializationStage = 'vision';
    if (!await createLandmarkers(session)) return;
    state.initializing = false;
    state.running = true;
    $('#loading-overlay').hidden = true;
    systemStatus.textContent = 'CAMÉRA ACTIVE';
    $('.state-dot').classList.add('active');
    cameraButton.textContent = 'ARRÊTER LA CAMÉRA';
    cameraButton.disabled = false;
    suitButton.disabled = false;
    startSuitUp();
    clock.start();
    state.raf = requestAnimationFrame(renderFrame);
  } catch (error) {
    if (session !== state.cameraSession) return;
    console.error(`Échec de l’initialisation ${initializationStage === 'vision' ? 'MediaPipe' : 'webcam'}.`, error);
    state.initializing = false;
    stopCamera();
    const detail = error instanceof Error ? error.message : '';
    const message = error?.name === 'NotAllowedError'
      ? 'Accès caméra refusé. Autorisez la webcam dans les paramètres du navigateur, puis réessayez.'
      : initializationStage === 'vision'
        ? `Chargement MediaPipe impossible. Vérifiez votre connexion Internet, puis rechargez la page. Les fichiers WASM et les modèles sont téléchargés au premier démarrage.${detail ? ` Détail : ${detail}` : ''}`
        : `Impossible d’accéder à la webcam${detail ? ` : ${detail}` : '. Vérifiez les autorisations du navigateur et fermez les autres applications utilisant la caméra.'}`;
    showError(message);
    cameraButton.disabled = false;
    cameraButton.textContent = 'RÉESSAYER';
  }
}

function stopCamera() {
  state.cameraSession += 1;
  state.running = false;
  state.initializing = false;
  cancelAnimationFrame(state.raf);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.pose?.close();
  state.face?.close();
  state.hands?.close();
  state.pose = null;
  state.face = null;
  state.hands = null;
  state.faceLoadState = 'offline';
  state.handLoadState = 'offline';
  state.posePoints = [];
  state.poseConfidence = 0;
  state.smoothedPose = null;
  state.lastPoseTime = 0;
  state.facePoint = null;
  state.faceTransform = null;
  state.faceRoll = 0;
  state.faceYaw = 0;
  state.facePitch = 0;
  state.handsDetected = false;
  state.fistWasClosed = false;
  state.fistTriggered = false;
  state.lastVideoTime = -1;
  state.lastFaceDetection = 0;
  state.lastHandDetection = 0;
  state.poseVisibleParts.clear();
  state.root.visible = false;
  if (state.customRoot) state.customRoot.visible = false;
  state.poseDebugDots.forEach((dot) => { dot.visible = false; });
  state.poseDebugGeometry.setDrawRange(0, 0);
  video.srcObject = null;
  $('#loading-overlay').hidden = false;
  $('#tracking-hint').hidden = true;
  $('#loading-overlay p').textContent = 'CAMÉRA DÉSACTIVÉE';
  $('#loading-overlay > span').textContent = 'Initialisez la caméra pour commencer le suivi du corps.';
  systemStatus.textContent = 'SYSTÈME EN ATTENTE';
  $('.state-dot').classList.remove('active');
  cameraButton.textContent = 'INITIALISER LA CAMÉRA';
  cameraButton.disabled = false;
  suitButton.disabled = true;
  $('#fps-status').textContent = '— FPS';
  for (const id of ['#pose-status', '#tracking-confidence', '#face-status', '#hand-status']) setMetric(id, false);
}

function positionCustomModel() {
  const p = state.posePoints;
  if (!p[0] || !p[27] || !p[28]) return;
  const head = pointToWorld(p[0]);
  const feet = midpoint(pointToWorld(p[27]), pointToWorld(p[28]));
  const height = Math.max(head.distanceTo(feet), 0.5);
  const root = state.customRoot;
  root.position.set(head.x, feet.y, 3.05);
  // Fits a standing, origin-centered GLB to the detected head-to-ankle height.
  root.scale.setScalar(height / root.userData.modelHeight);
}

function classifyBone(name) {
  const boneName = name.replace(/^mixamorig/i, '').toLowerCase();
  const side = boneName.includes('left') ? 'left' : boneName.includes('right') ? 'right' : '';
  if (!side) {
    if (/head|neck/.test(boneName)) return { point: 'head', yaw: /head/.test(boneName) };
    if (/hips|pelvis|root/.test(boneName)) return { point: 'hips' };
    if (/chest|spine2|spine3/.test(boneName)) return { point: 'chest' };
    return null;
  }
  if (/shoulder|clavicle/.test(boneName)) return { point: `${side}Shoulder` };
  if (/forearm|lowerarm|elbow/.test(boneName)) return { start: `${side}Elbow`, end: `${side}Wrist` };
  if (/upperarm|arm/.test(boneName)) return { start: `${side}Shoulder`, end: `${side}Elbow` };
  if (/hand|wrist/.test(boneName)) return { point: `${side}Wrist` };
  if (/thigh|upleg|upperleg/.test(boneName)) return { start: `${side}Hip`, end: `${side}Knee` };
  if (/shin|calf|lowerleg|leg/.test(boneName)) return { start: `${side}Knee`, end: `${side}Ankle` };
  if (/foot|ankle/.test(boneName)) return { point: `${side}Ankle` };
  return null;
}

function updateCustomSkeleton() {
  if (!state.customRoot || !state.posePoints.length) return;
  const p = state.posePoints;
  const points = mapLandmarks(p, {
    head: 0, leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
    leftWrist: 15, rightWrist: 16, leftHip: 23, rightHip: 24,
    leftKnee: 25, rightKnee: 26, leftAnkle: 27, rightAnkle: 28,
  });
  points.head = state.facePoint ?? points.head;
  points.hips = midpoint(points.leftHip, points.rightHip);
  points.chest = midpoint(points.leftShoulder, points.rightShoulder);

  state.customRoot.updateMatrixWorld(true);
  state.customRoot.traverse((bone) => {
    const target = bone.userData.retargetTarget;
    if (!bone.isBone || !target || !bone.parent) return;
    const start = target.start ? points[target.start] : points[target.point];
    const end = target.end ? points[target.end] : null;
    if (!start) return;
    const localStart = bone.parent.worldToLocal(start.clone());
    bone.position.copy(localStart);
    if (end) {
      const localEnd = bone.parent.worldToLocal(end.clone());
      const direction = localEnd.sub(localStart);
      if (direction.lengthSq() > 0.00001) {
        bone.quaternion.setFromUnitVectors(bone.userData.retargetAxis, direction.normalize());
      }
    } else if (target.yaw) {
      bone.quaternion.copy(bone.userData.retargetRestRotation)
        .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), state.faceYaw));
    }
    bone.updateWorldMatrix(false, true);
  });
}

function loadCustomModel(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.glb')) {
    showError('Pour importer les textures et ressources en un seul fichier, utilisez un modèle .glb (et non .gltf).');
    return;
  }
  const objectUrl = URL.createObjectURL(file);
  loader.load(objectUrl, (gltf) => {
    URL.revokeObjectURL(objectUrl);
    if (state.customRoot) scene.remove(state.customRoot);
    const model = gltf.scene;
    const bounds = new THREE.Box3().setFromObject(model);
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const modelHeight = Math.max(size.y, 0.001);
    const root = new THREE.Group();
    root.name = 'user-glb-armor';
    model.position.set(-center.x, -bounds.min.y, -center.z);
    root.add(model);
    root.userData.modelHeight = modelHeight;
    model.traverse((node) => {
      if (node.isBone) {
        node.userData.retargetTarget = classifyBone(node.name);
        const childBone = node.children.find((child) => child.isBone);
        node.userData.retargetAxis = childBone?.position.clone().normalize() ?? new THREE.Vector3(0, 1, 0);
        node.userData.retargetRestRotation = node.quaternion.clone();
      }
      if (!node.isMesh) return;
      node.material = Array.isArray(node.material)
        ? node.material.map((material) => installDissolve(material))
        : installDissolve(node.material);
      node.frustumCulled = false;
    });
    scene.add(root);
    state.customRoot = root;
    state.mixer = gltf.animations.length ? new THREE.AnimationMixer(root) : null;
    gltf.animations.forEach((clip) => state.mixer.clipAction(clip).play());
    $('#model-select').value = 'custom';
    state.root.visible = false;
    errorBanner.hidden = true;
    if (state.running) systemStatus.textContent = 'CAMÉRA ACTIVE';
  }, undefined, (error) => {
    URL.revokeObjectURL(objectUrl);
    showError(`Chargement du GLB impossible : ${error.message}`);
  });
}

$('#model-select').addEventListener('change', (event) => {
  const useCustom = event.target.value === 'custom' && Boolean(state.customRoot);
  if (!useCustom) event.target.value = 'procedural';
  state.root.visible = !useCustom && state.suitProgress > 0;
  if (state.customRoot) state.customRoot.visible = Boolean(useCustom) && state.suitProgress > 0;
});
$('#model-file').addEventListener('change', (event) => loadCustomModel(event.target.files?.[0]));
$('#pose-debug-toggle').addEventListener('change', (event) => {
  state.poseDebugGroup.visible = event.target.checked;
  if (!event.target.checked) {
    state.poseDebugDots.forEach((dot) => { dot.visible = false; });
    state.poseDebugGeometry.setDrawRange(0, 0);
  } else {
    updatePoseDebug();
  }
});
cameraButton.addEventListener('click', () => state.running || state.initializing ? stopCamera() : startCamera());
suitButton.addEventListener('click', startSuitUp);
window.addEventListener('pagehide', stopCamera);
window.addEventListener('beforeunload', () => {
  state.pose?.close();
  state.face?.close();
  state.hands?.close();
});
