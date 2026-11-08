import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { PoseBridge } from "./tracker.js";

const $ = (selector) => document.querySelector(selector);
const video = $("#webcam");
const stage = $("#stage");
const canvas = $("#scene");
const cameraButton = $("#camera-button");
const suitButton = $("#suit-button");
const statusDot = $("#connection-dot");
const statusText = $("#connection-status");
const renderer = new THREE.WebGLRenderer({
  canvas,
  alpha: true,
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const aspect = Math.max(stage.clientWidth / Math.max(stage.clientHeight, 1), 0.1);
const camera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.1, 50);
camera.position.set(0, 0, 12);
camera.lookAt(0, 0, 0);
scene.add(new THREE.HemisphereLight(0xc8f7ff, 0x121b25, 2.25));
const keyLight = new THREE.DirectionalLight(0xffe6bf, 3.2);
keyLight.position.set(-2, 4, 8);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x00dfff, 18, 8);
rimLight.position.set(0, 0, 3);
scene.add(rimLight);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.85, 0.5, 0.14));

const state = {
  bridge: null,
  running: false,
  pose: null,
  transforms: new Map(),
  parts: new Map(),
  suitStartedAt: 0,
  suitProgress: 0,
  fistWasClosed: false,
  lastFrameAt: performance.now(),
  lastFpsUpdate: performance.now(),
  renderedFrames: 0,
  debugLines: null,
  debugDots: [],
  debugGroup: null,
  particles: null,
  particleStarts: null,
  particleTargets: null,
  particlePhases: null,
};

const ARMOR = new THREE.MeshPhysicalMaterial({
  color: 0x25363e,
  metalness: 0.88,
  roughness: 0.2,
  clearcoat: 0.95,
  clearcoatRoughness: 0.12,
});
const EDGE = new THREE.MeshPhysicalMaterial({
  color: 0x10191f,
  metalness: 0.9,
  roughness: 0.24,
  clearcoat: 0.8,
});
const GOLD = new THREE.MeshPhysicalMaterial({
  color: 0xb58337,
  metalness: 0.88,
  roughness: 0.23,
  clearcoat: 0.75,
});
const CYAN = new THREE.MeshStandardMaterial({
  color: 0x9bfaff,
  emissive: 0x00dff5,
  emissiveIntensity: 3.4,
  metalness: 0.1,
  roughness: 0.13,
});

function installMaterialization(material) {
  const instance = material.clone();
  instance.userData.suitUniforms = { progress: { value: 0 } };
  instance.onBeforeCompile = (shader) => {
    shader.uniforms.uSuitProgress = instance.userData.suitUniforms.progress;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vSuitLocal;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvSuitLocal = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
        varying vec3 vSuitLocal;
        uniform float uSuitProgress;
        float suitHash(vec3 p) {
          p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
          p *= 17.0;
          return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }`,
      )
      .replace(
        "#include <opaque_fragment>",
        `float suitField = vSuitLocal.y * 0.45 + suitHash(vSuitLocal * 13.0) * 0.12;
        float suitLimit = mix(-0.7, 0.7, uSuitProgress);
        float suitEdge = 1.0 - smoothstep(0.0, 0.08, abs(suitField - suitLimit));
        if (suitField > suitLimit) discard;
        outgoingLight += vec3(0.01, 0.55, 0.78) * suitEdge * 2.8;
        #include <opaque_fragment>`,
      );
  };
  instance.customProgramCacheKey = () => "stark-nanotech-morph-v1";
  return instance;
}

function addMesh(parent, geometry, material, position = [0, 0, 0], rotation = [0, 0, 0], scale = [1, 1, 1]) {
  const mesh = new THREE.Mesh(geometry, installMaterialization(material));
  mesh.position.set(...position);
  mesh.rotation.set(...rotation);
  mesh.scale.set(...scale);
  mesh.castShadow = false;
  mesh.frustumCulled = false;
  parent.add(mesh);
  return mesh;
}

function addArmorBox(parent, material, size, position, bevel = 0.035) {
  const geometry = new THREE.BoxGeometry(...size);
  const mesh = addMesh(parent, geometry, material, position);
  mesh.geometry.computeVertexNormals();
  if (bevel > 0) {
    const edges = new THREE.EdgesGeometry(geometry, 28);
    const outline = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x507987, transparent: true, opacity: 0.42 }),
    );
    outline.position.copy(mesh.position);
    outline.scale.copy(mesh.scale);
    parent.add(outline);
  }
  return mesh;
}

function newPart(name) {
  const group = new THREE.Group();
  group.name = `armor-${name}`;
  group.visible = false;
  scene.add(group);
  state.parts.set(name, { group, materials: [] });
  return group;
}

function rememberMaterials(name) {
  const entry = state.parts.get(name);
  entry.group.traverse((node) => {
    if (!node.isMesh) return;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (material.userData.suitUniforms) entry.materials.push(material.userData.suitUniforms.progress);
    }
  });
}

function createChest() {
  const group = newPart("torso");
  const shield = new THREE.Shape();
  shield.moveTo(-0.42, 0.43);
  shield.lineTo(0.42, 0.43);
  shield.lineTo(0.52, 0.18);
  shield.lineTo(0.36, -0.3);
  shield.lineTo(0, -0.49);
  shield.lineTo(-0.36, -0.3);
  shield.lineTo(-0.52, 0.18);
  shield.closePath();
  const shell = new THREE.ExtrudeGeometry(shield, {
    depth: 0.22,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.035,
    bevelThickness: 0.045,
  });
  shell.translate(0, 0, -0.11);
  addMesh(group, shell, ARMOR);
  addArmorBox(group, EDGE, [0.72, 0.08, 0.2], [0, -0.35, 0.03], 0);
  addArmorBox(group, GOLD, [0.07, 0.3, 0.11], [-0.44, 0.08, 0.02], 0);
  addArmorBox(group, GOLD, [0.07, 0.3, 0.11], [0.44, 0.08, 0.02], 0);
  addArmorBox(group, EDGE, [0.65, 0.09, 0.22], [0, 0.37, 0.02], 0);
  const inner = addMesh(group, new THREE.CylinderGeometry(0.18, 0.18, 0.065, 32), EDGE, [0, 0.08, 0.155], [Math.PI / 2, 0, 0]);
  inner.material.side = THREE.DoubleSide;
  addMesh(group, new THREE.CylinderGeometry(0.14, 0.14, 0.09, 32), CYAN, [0, 0.08, 0.205], [Math.PI / 2, 0, 0]);
  addMesh(group, new THREE.TorusGeometry(0.175, 0.018, 8, 32), GOLD, [0, 0.08, 0.25]);
  rememberMaterials("torso");
}

function createSegmentArmor(name, material, radius = 1) {
  const group = newPart(name);
  addMesh(group, new THREE.CylinderGeometry(radius * 0.92, radius, 1, 12, 1, false), material);
  addArmorBox(group, EDGE, [radius * 1.3, 0.12, radius * 0.5], [0, 0.18, radius * 0.72], 0);
  const plate = addArmorBox(group, GOLD, [radius * 0.18, 0.55, radius * 0.08], [0, -0.02, radius * 0.97], 0);
  plate.material.color.setHex(0x53636a);
  rememberMaterials(name);
}

function createShoulder(name) {
  const group = newPart(name);
  addMesh(group, new THREE.SphereGeometry(1, 20, 14), ARMOR, [0, 0, 0], [0, 0, 0], [1, 0.82, 0.95]);
  addArmorBox(group, EDGE, [1.1, 0.28, 0.28], [0, -0.22, 0.62], 0);
  addMesh(group, new THREE.TorusGeometry(0.47, 0.035, 8, 24), CYAN, [0, 0.12, 0.82], [0, 0, 0]);
  rememberMaterials(name);
}

function createHand(name) {
  const group = newPart(name);
  addArmorBox(group, EDGE, [0.78, 0.62, 0.35], [0, 0, 0], 0);
  addArmorBox(group, ARMOR, [0.68, 0.5, 0.39], [0, 0, 0.08], 0);
  addArmorBox(group, GOLD, [0.54, 0.09, 0.06], [0, -0.13, 0.3], 0);
  addMesh(group, new THREE.CylinderGeometry(0.18, 0.2, 0.045, 24), CYAN, [0, 0, 0.32], [Math.PI / 2, 0, 0]);
  rememberMaterials(name);
}

function createFoot(name) {
  const group = newPart(name);
  addArmorBox(group, EDGE, [0.7, 0.78, 0.5], [0, 0, 0], 0);
  addArmorBox(group, ARMOR, [0.62, 0.62, 0.55], [0, 0.04, 0.07], 0);
  addArmorBox(group, GOLD, [0.66, 0.15, 0.54], [0, -0.29, 0.08], 0);
  rememberMaterials(name);
}

function createHelmet() {
  const group = newPart("head");
  addMesh(group, new THREE.SphereGeometry(1, 24, 18), EDGE, [0, 0, 0], [0, 0, 0], [0.94, 1.08, 0.78]);
  addArmorBox(group, ARMOR, [1.35, 0.54, 0.54], [0, 0.17, 0.37], 0);
  addArmorBox(group, GOLD, [1.05, 0.13, 0.1], [0, -0.05, 0.67], 0);
  const visor = addArmorBox(group, CYAN, [1.03, 0.12, 0.035], [0, 0.19, 0.79], 0);
  visor.material.transparent = true;
  visor.material.opacity = 0.9;
  addArmorBox(group, ARMOR, [0.24, 0.45, 0.18], [0, -0.3, 0.66], 0);
  rememberMaterials("head");
}

function createArmorRig() {
  createChest();
  createHelmet();
  createShoulder("shoulder_left");
  createShoulder("shoulder_right");
  createSegmentArmor("upper_arm_left", ARMOR, 0.78);
  createSegmentArmor("upper_arm_right", ARMOR, 0.78);
  createSegmentArmor("forearm_left", EDGE, 0.72);
  createSegmentArmor("forearm_right", EDGE, 0.72);
  createHand("hand_left");
  createHand("hand_right");
  createSegmentArmor("thigh_left", ARMOR, 0.86);
  createSegmentArmor("thigh_right", ARMOR, 0.86);
  createSegmentArmor("shin_left", EDGE, 0.75);
  createSegmentArmor("shin_right", EDGE, 0.75);
  createFoot("foot_left");
  createFoot("foot_right");
  const reactor = newPart("reactor");
  addMesh(reactor, new THREE.CylinderGeometry(1, 1, 0.24, 32), EDGE, [0, 0, 0], [Math.PI / 2, 0, 0]);
  addMesh(reactor, new THREE.CylinderGeometry(0.76, 0.76, 0.27, 32), CYAN, [0, 0, 0.035], [Math.PI / 2, 0, 0]);
  addMesh(reactor, new THREE.TorusGeometry(0.88, 0.08, 10, 32), GOLD, [0, 0, 0.12]);
  rememberMaterials("reactor");
}

createArmorRig();

function createDebugRig() {
  const group = new THREE.Group();
  group.visible = false;
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(24 * 6), 3));
  const lines = new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0x00f3ff, transparent: true, opacity: 0.92, depthTest: false }),
  );
  lines.renderOrder = 20;
  group.add(lines);
  const geometry = new THREE.SphereGeometry(0.018, 8, 6);
  const material = new THREE.MeshBasicMaterial({ color: 0xffb700, depthTest: false });
  state.debugDots = Array.from({ length: 33 }, () => {
    const dot = new THREE.Mesh(geometry, material);
    dot.renderOrder = 21;
    group.add(dot);
    return dot;
  });
  scene.add(group);
  state.debugGroup = group;
  state.debugLines = lineGeometry;
}

createDebugRig();

function resize() {
  const bounds = stage.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = bounds.width / bounds.height;
  renderer.setSize(bounds.width, bounds.height, false);
  composer.setSize(bounds.width, bounds.height);
  camera.left = -ratio;
  camera.right = ratio;
  camera.top = 1;
  camera.bottom = -1;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(stage);
window.addEventListener("resize", resize);
resize();

function toWorld(landmark) {
  const bounds = stage.getBoundingClientRect();
  const ratio = bounds.width / Math.max(bounds.height, 1);
  return new THREE.Vector3((landmark.x - 0.5) * 2 * ratio, (0.5 - landmark.y) * 2, -(landmark.z || 0) * 0.35);
}

const connections = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];

function updateDebugPose(pose) {
  if (!state.debugGroup.visible) return;
  const positions = state.debugLines.attributes.position;
  let count = 0;
  for (const [first, second] of connections) {
    const a = pose.landmarks[first];
    const b = pose.landmarks[second];
    if (!a || !b || a.visibility < 0.2 || b.visibility < 0.2) continue;
    const start = toWorld(a);
    const end = toWorld(b);
    positions.setXYZ(count * 2, start.x, start.y, 4.7);
    positions.setXYZ(count * 2 + 1, end.x, end.y, 4.7);
    count += 1;
  }
  positions.needsUpdate = true;
  state.debugLines.setDrawRange(0, count * 2);
  state.debugDots.forEach((dot, index) => {
    const landmark = pose.landmarks[index];
    dot.visible = Boolean(landmark && landmark.visibility >= 0.2);
    if (dot.visible) {
      const position = toWorld(landmark);
      dot.position.set(position.x, position.y, 4.75);
    }
  });
}

function applyTransforms(pose, deltaSeconds) {
  state.transforms.clear();
  const alpha = 1 - Math.exp(-deltaSeconds * 16);
  for (const [name, entry] of state.parts) {
    const transform = pose.transforms?.[name];
    if (!transform?.matrix || transform.matrix.length !== 16) {
      entry.group.visible = false;
      continue;
    }
    const matrix = new THREE.Matrix4().fromArray(transform.matrix);
    const targetPosition = new THREE.Vector3();
    const targetQuaternion = new THREE.Quaternion();
    const targetScale = new THREE.Vector3();
    matrix.decompose(targetPosition, targetQuaternion, targetScale);
    if (!entry.group.visible) {
      entry.group.position.copy(targetPosition);
      entry.group.quaternion.copy(targetQuaternion);
      entry.group.scale.copy(targetScale);
      entry.group.visible = true;
    } else {
      entry.group.position.lerp(targetPosition, alpha);
      entry.group.quaternion.slerp(targetQuaternion, alpha);
      entry.group.scale.lerp(targetScale, alpha);
    }
    state.transforms.set(name, transform);
  }
  updateDebugPose(pose);
}

function partProgress(name, progress) {
  if (name === "reactor") return THREE.MathUtils.clamp(progress * 2.5, 0, 1);
  if (name === "torso" || name.startsWith("shoulder")) return THREE.MathUtils.clamp((progress - 0.08) / 0.42, 0, 1);
  if (name.startsWith("upper_arm") || name.startsWith("forearm") || name.startsWith("hand")) {
    return THREE.MathUtils.clamp((progress - 0.28) / 0.36, 0, 1);
  }
  if (name.startsWith("thigh") || name.startsWith("shin") || name.startsWith("foot")) {
    return THREE.MathUtils.clamp((progress - 0.49) / 0.34, 0, 1);
  }
  return THREE.MathUtils.clamp((progress - 0.78) / 0.22, 0, 1);
}

function createNanobots() {
  const count = 1100;
  const positions = new Float32Array(count * 3);
  state.particleStarts = new Float32Array(count * 3);
  state.particleTargets = new Float32Array(count * 3);
  state.particlePhases = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const start = new THREE.Vector3(
      (Math.random() - 0.5) * 2.8,
      (Math.random() - 0.5) * 2.4,
      0.6 + Math.random() * 2,
    );
    state.particleStarts.set([start.x, start.y, start.z], offset);
    positions.set([start.x, start.y, start.z], offset);
    state.particlePhases[index] = Math.random() * Math.PI * 2;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  const material = new THREE.PointsMaterial({
    color: 0x42f4ff,
    size: 0.035,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });
  state.particles = new THREE.Points(geometry, material);
  state.particles.visible = false;
  state.particles.renderOrder = 18;
  scene.add(state.particles);
}

createNanobots();

function prepareParticleTargets() {
  const targets = [...state.parts.values()]
    .filter((entry) => entry.group.visible)
    .map((entry) => entry.group.position);
  if (!targets.length) return false;
  const count = state.particlePhases.length;
  for (let index = 0; index < count; index += 1) {
    const target = targets[Math.floor(Math.random() * targets.length)];
    const offset = index * 3;
    const spread = 0.06 + Math.random() * 0.3;
    state.particleTargets[offset] = target.x + (Math.random() - 0.5) * spread;
    state.particleTargets[offset + 1] = target.y + (Math.random() - 0.5) * spread;
    state.particleTargets[offset + 2] = target.z + 0.15 + Math.random() * 0.2;
  }
  state.particles.visible = true;
  return true;
}

function updateNanobots(now) {
  if (!state.particles.visible) return;
  const attribute = state.particles.geometry.attributes.position;
  const positions = attribute.array;
  const progress = THREE.MathUtils.clamp(state.suitProgress, 0, 1);
  const eased = progress * progress * (3 - 2 * progress);
  for (let index = 0; index < state.particlePhases.length; index += 1) {
    const offset = index * 3;
    const phase = state.particlePhases[index] + now * 0.0014;
    const transient = Math.sin(progress * Math.PI) * 0.22;
    positions[offset] = THREE.MathUtils.lerp(state.particleStarts[offset], state.particleTargets[offset], eased) + Math.cos(phase) * transient;
    positions[offset + 1] = THREE.MathUtils.lerp(state.particleStarts[offset + 1], state.particleTargets[offset + 1], eased) + Math.sin(phase * 1.13) * transient;
    positions[offset + 2] = THREE.MathUtils.lerp(state.particleStarts[offset + 2], state.particleTargets[offset + 2], eased);
  }
  attribute.needsUpdate = true;
  state.particles.material.opacity = Math.max(0, 0.9 * (1 - progress * progress));
  if (progress >= 1) state.particles.visible = false;
}

function startSuitUp() {
  if (!state.running || !state.pose?.detected) return;
  state.suitProgress = 0;
  state.suitStartedAt = performance.now();
  prepareParticleTargets();
  $("#suit-label").textContent = "ASSEMBLAGE NANOTECH";
  $("#suit-button").disabled = true;
}

function updateHud(pose, fps, latency) {
  const detected = Boolean(pose?.detected);
  $("#pose-status").textContent = detected ? "CORPS VERROUILLÉ" : "RECHERCHE DU CORPS";
  $("#pose-status").classList.toggle("active", detected);
  $("#tracking-status").textContent = detected ? "/ CORPS VERROUILLÉ" : "/ CADRAGE REQUIS";
  $("#confidence").textContent = `${Math.round((pose?.confidence || 0) * 100)} %`;
  $("#confidence").classList.toggle("active", detected);
  $("#latency").textContent = Number.isFinite(latency) ? `${Math.round(latency)} ms` : "— ms";
  if (Number.isFinite(fps)) $("#fps").textContent = `${fps} FPS`;
  const parts = pose?.transforms || {};
  const chestReady = Boolean(parts.torso);
  const leftReady = Boolean(parts.forearm_left && parts.hand_left);
  const rightReady = Boolean(parts.forearm_right && parts.hand_right);
  $("#chest-status").textContent = chestReady ? "VERROUILLÉ" : "EN ATTENTE";
  $("#left-status").textContent = leftReady ? "VERROUILLÉ" : "EN ATTENTE";
  $("#right-status").textContent = rightReady ? "VERROUILLÉ" : "EN ATTENTE";
  $("#reactor-status").textContent = parts.reactor ? "NOMINAL" : "STANDBY";
  $("#chest-health").style.width = chestReady ? `${Math.round(pose.confidence * 100)}%` : "0%";
  $("#left-health").style.width = leftReady ? "100%" : "0%";
  $("#right-health").style.width = rightReady ? "100%" : "0%";
  $("#reactor-health").style.width = parts.reactor ? "100%" : "0%";
  const hint = $("#tracking-hint");
  hint.hidden = !state.running || detected;
  hint.textContent = pose?.landmarks?.length
    ? "Pose partielle : gardez les épaules et le bassin visibles. Les pièces suivent chaque articulation détectée."
    : "Aucun corps détecté. Placez-vous face à la caméra, avec une bonne lumière; gardez les épaules et le bassin dans le cadre.";
  suitButton.disabled = !state.running || !detected || Boolean(state.suitStartedAt);
}

function setSystemStatus(stateName, message) {
  statusText.textContent = message;
  statusDot.classList.toggle("online", stateName === "online" || stateName === "camera");
  if (stateName === "error") statusDot.classList.remove("online");
  if (stateName === "offline") {
    cameraButton.textContent = "INITIALISER LA CAMÉRA";
    cameraButton.disabled = false;
    suitButton.disabled = true;
    $("#tracking-hint").hidden = false;
    $("#tracking-hint").textContent = "Initialisez la webcam pour démarrer la détection des articulations.";
    $("#socket-status").textContent = "HORS LIGNE";
    $("#socket-status").classList.remove("active");
  } else {
    cameraButton.textContent = stateName === "camera" || stateName === "connecting" ? "ANNULER LE CHARGEMENT" : "ARRÊTER LA CAMÉRA";
    cameraButton.disabled = false;
    if (stateName === "online") {
      $("#socket-status").textContent = "CONNECTÉ";
      $("#socket-status").classList.add("active");
    } else if (stateName === "reconnecting") {
      $("#socket-status").textContent = "RECONNEXION…";
      $("#socket-status").classList.remove("active");
    } else if (stateName === "error") {
      cameraButton.textContent = "RÉESSAYER";
      $("#socket-status").textContent = "ERREUR";
    }
  }
}

function animate(now) {
  requestAnimationFrame(animate);
  const delta = Math.min((now - state.lastFrameAt) / 1000, 0.05);
  state.lastFrameAt = now;
  state.renderedFrames += 1;
  if (now - state.lastFpsUpdate >= 1000) {
    const fps = Math.round(state.renderedFrames * 1000 / (now - state.lastFpsUpdate));
    $("#fps").textContent = `${fps} FPS`;
    state.renderedFrames = 0;
    state.lastFpsUpdate = now;
  }
  if (state.suitStartedAt) {
    state.suitProgress = Math.min((now - state.suitStartedAt) / 4200, 1);
    if (state.suitProgress >= 1) state.suitStartedAt = 0;
  }
  for (const [name, entry] of state.parts) {
    const progress = partProgress(name, state.suitProgress);
    for (const uniform of entry.materials) uniform.value = progress;
  }
  const arc = state.parts.get("reactor");
  if (arc?.group.visible) {
    const pulse = 1 + Math.sin(now * 0.004) * 0.08;
    rimLight.intensity = 18 * pulse;
    rimLight.position.copy(arc.group.position);
    rimLight.position.z += 0.5;
  }
  updateNanobots(now);
  const percent = Math.round(state.suitProgress * 100);
  $("#suit-progress").style.width = `${percent}%`;
  $("#suit-value").textContent = `${percent}%`;
  if (percent >= 100) {
    $("#suit-label").textContent = "ARMURE DÉPLOYÉE";
    if (state.running && state.pose?.detected && !state.suitStartedAt) suitButton.disabled = false;
  } else if (percent > 0) {
    $("#suit-label").textContent = "VERROUILLAGE DES PLAQUES";
  } else {
    $("#suit-label").textContent = "PRÊT À L’ACTIVATION";
  }
  composer.render();
}

state.bridge = new PoseBridge({ video, targetFps: 20, captureWidth: 512, quality: 0.58 });
state.bridge.addEventListener("status", (event) => {
  const next = event.detail;
  state.running = next.state !== "offline" && next.state !== "error";
  if (next.state === "online" || next.state === "camera") state.running = true;
  setSystemStatus(next.state, next.message);
  if (next.state === "offline" || next.state === "error") {
    state.pose = null;
    state.suitProgress = 0;
    state.suitStartedAt = 0;
    for (const entry of state.parts.values()) entry.group.visible = false;
    state.particles.visible = false;
    state.debugDots.forEach((dot) => { dot.visible = false; });
  }
});
state.bridge.addEventListener("pose", (event) => {
  const pose = event.detail;
  state.pose = pose;
  const frameDelta = Math.max((performance.now() - state.lastFrameAt) / 1000, 1 / 120);
  if (pose.detected) {
    applyTransforms(pose, frameDelta);
  } else {
    for (const entry of state.parts.values()) entry.group.visible = false;
  }
  updateHud(pose, pose.fps, pose.processingMs + (pose.roundTripMs || 0));
  const fistClosed = (pose.landmarks?.[19]?.visibility || 0) > 0.55 &&
    (pose.landmarks?.[20]?.visibility || 0) > 0.55 &&
    pose.landmarks[19].y > pose.landmarks[17].y &&
    pose.landmarks[20].y > pose.landmarks[18].y;
  if (fistClosed && !state.fistWasClosed && state.suitProgress >= 0.98) startSuitUp();
  state.fistWasClosed = fistClosed;
});
state.bridge.addEventListener("fps", (event) => {
  $("#fps").textContent = `${event.detail.value} FPS`;
});
state.bridge.addEventListener("error", (event) => {
  $("#tracking-hint").hidden = false;
  $("#tracking-hint").textContent = event.detail.message;
});
state.bridge.addEventListener("connection-error", () => {
  $("#tracking-hint").hidden = false;
  $("#tracking-hint").textContent = "Serveur de vision indisponible. Relancez le serveur Python et la reconnexion sera automatique.";
});

cameraButton.addEventListener("click", () => {
  if (state.running) state.bridge.stop();
  else void state.bridge.start();
});
suitButton.addEventListener("click", startSuitUp);
$("#debug-toggle").addEventListener("change", (event) => {
  state.debugGroup.visible = event.target.checked;
  if (!event.target.checked) {
    state.debugLines.setDrawRange(0, 0);
    state.debugDots.forEach((dot) => { dot.visible = false; });
  } else if (state.pose) {
    updateDebugPose(state.pose);
  }
});
window.addEventListener("pagehide", () => state.bridge.stop());
requestAnimationFrame(animate);
