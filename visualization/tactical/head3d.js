// P1 head model for three.js: a person's head wearing the headset, with neck and shoulders, from assets/head.js
// (window.K8_HEAD, made in Blender by tools/blender/build_head.py). Axes: the face looks along −z, up is +y, origin in the middle of the head.
//
// makeHead() returns { rig, skin, pose(q) }:
//   rig   a Group with the headset (rigid).
//   skin  one continuous Mesh of head, neck and shoulders.
//   Put BOTH in one group that does NOT turn. pose(q) does the turning:
//   pose(q)  q = the head's rotation (a THREE.Quaternion). The neck rig in headpose.js shares it between two neck bones and the head,
//            bends the skin accordingly, and places the headset on the head (it turns by exactly q and shifts as the neck bends).
// It also sets window.K8HeadGL, which the Data side (a classic script, drawn on 2D canvases) uses to get the same lit 3D head as a picture.
// Story copy (ticket 06c): index.html does not load headpose.js, so it is imported here for its side effect (it sets window.K8HeadPose).
import './headpose.js';
import * as THREE from 'three';
import { loadHeadsetModel } from './hero_model.js';   // tactical (16 Nov): the real headset, Elin's CC BY "Meta Quest 3" (logo removed)

// tactical (16 Nov, Nic: "switch the models in those screens to the one we use in TAKE CONTROL"). The model viewer (?explore=1) and the
// Data side build their heads here, from the same head mesh as TAKE CONTROL's hero (assets/head.js and hero/head.js hold identical parts),
// so the Quest model, exported in that head's rest frame, fits these heads as it is. It is loaded once (the same file the hero uses, so
// it usually comes from the browser's cache); each head gets a copy that shares the geometry. The box headset's meshes are hidden, not
// removed: the pointer's start (front) and the framing are still measured from them, as in the hero. If the file fails, the box stays.
let HEADSET = null;
const headsetModel = () => (HEADSET ||= loadHeadsetModel(THREE).catch(e => { console.warn('headset model did not load; keeping the box headset', e); return null; }));
function useHeadsetModel(rig, onReady) {
  headsetModel().then(model => {
    if (!model) return;
    const copy = model.clone(true);
    for (const m of rig.children) if (m.isMesh) m.visible = false;
    rig.add(copy);
    if (onReady) onReady();
  });
}

// T2 (10 Nov): the clay look. A "matcap" (material capture) is a small picture of a lit ball: for each point of the surface the GPU looks up
// the pixel whose direction on the ball matches the surface's direction as seen from the camera, so the lighting comes with the material,
// costs one texture lookup, and looks the same in the Model scene and on the Data side whatever lights those scenes have. This one is
// painted in a canvas at load (no image file): a warm key light from the upper left front, a weak cool fill from the right, a soft
// highlight, and a rim light round the edge that outlines the head against the dark page. Each part's own colour multiplies it.
const POINTER_HEADS = 1.5;   // F3: the pointer's length in head-lengths (the story's hero.js)
// T2's clay matcap below is now the cheaper look, used on the "light" and "min" graphics tiers (makeHead().setCheap); the full tier uses LOOK.
let MATCAP = null;
function matcap() {
  if (MATCAP) return MATCAP;
  const N = 256, cv = document.createElement('canvas'); cv.width = cv.height = N;
  const g = cv.getContext('2d'), img = g.createImageData(N, N), d = img.data;
  const nrm = v => { const l = Math.hypot(...v); return v.map(c => c / l); }, dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const KEY = nrm([-0.5, 0.6, 0.65]), FILL = nrm([0.85, -0.1, 0.5]), HALF = nrm([KEY[0], KEY[1], KEY[2] + 1]), RIM = nrm([0.35, 0.5, -0.8]);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const x = (i + 0.5) / N * 2 - 1, y = 1 - (j + 0.5) / N * 2, r2 = x * x + y * y;
    const n = r2 >= 1 ? nrm([x, y, 0.001]) : [x, y, Math.sqrt(1 - r2)];   // past the edge: the ball's own outline (texture filtering reads it)
    const key = Math.max(0, (dot(n, KEY) + 0.2) / 1.2), fill = Math.max(0, dot(n, FILL)), spec = Math.pow(Math.max(0, dot(n, HALF)), 28);
    const occ = 0.82 + 0.18 * (0.5 + 0.5 * n[1]);                         // a little darker underneath, as under a soft sky
    const edge = Math.pow(1 - n[2], 2.2), rim = edge * (0.45 + 0.55 * Math.max(0, 1 + dot(n, RIM)) / 2) * 1.15;   // strongest on the edge facing the back light
    const c = [[1.0, 0.96, 0.9], [0.62, 0.7, 0.86]];                       // key: warm white; fill and rim: cool
    for (let k = 0; k < 3; k++) {
      const v = occ * (0.16 + 0.78 * key * c[0][k] + 0.2 * fill * c[1][k]) + 0.1 * spec + rim * (0.55 + 0.45 * c[1][k]);
      d[4 * (j * N + i) + k] = Math.round(255 * Math.min(1, v));
    }
    d[4 * (j * N + i) + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  MATCAP = new THREE.CanvasTexture(cv); MATCAP.colorSpace = THREE.SRGBColorSpace;
  return MATCAP;
}
// The matcap is multiplied by the part's colour, so on the dark shoulders and visor its rim would vanish. The rim is therefore also added
// on top, the same for every colour: brightest where the surface turns away from the camera (normal.z near 0 in camera space).
const RIM_GLSL = 'outgoingLight = diffuseColor.rgb * matcapColor.rgb + vec3( 0.55, 0.6, 0.7 ) * 0.42 * pow( 1.0 - clamp( normal.z, 0.0, 1.0 ), 3.0 );';
function clay(color) {
  const m = new THREE.MeshMatcapMaterial({ color, matcap: matcap() });
  m.onBeforeCompile = s => { s.fragmentShader = s.fragmentShader.replace('outgoingLight = diffuseColor.rgb * matcapColor.rgb;', RIM_GLSL); };
  m.customProgramCacheKey = () => 'k8clay';
  return m;
}

// F3 (Nic, 10 Nov: "port the story lights and materials"): the story site's look (../story/hero.js LOOK and stage_task.js light()),
// copied here so the Data side and the Model side match it. Physically based materials: `roughness` is how matte a surface is,
// `clearcoat` a thin glossy layer on top (the visor, the shell), `sheen` a soft glow at grazing angles that keeps skin and cloth from
// looking like plastic. Keys are the part names in assets/head.js (the skin part's two lists are "skin" and "cloth"). Same values as the story.
export const LOOK = {
  skin:    { color: '#c8a590', roughness: 0.58, sheen: 0.3, sheenColor: '#ff9d80', sheenRoughness: 0.6 },
  cloth:   { color: '#474c57', roughness: 0.92, sheen: 0.5, sheenColor: '#8f97a6', sheenRoughness: 0.8 },
  body:    { color: '#ecebe7', roughness: 0.34, clearcoat: 0.5, clearcoatRoughness: 0.25 },          // the headset's shell
  plate:   { color: '#08090b', roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.04 },             // the visor: reflects the lights
  pad:     { color: '#2b2c2f', roughness: 0.85 },
  strap:   { color: '#33363b', roughness: 0.8 }, top: { color: '#33363b', roughness: 0.8 }, backpad: { color: '#33363b', roughness: 0.8 },
};
const physical = (look, color) => new THREE.MeshPhysicalMaterial({ color, roughness: 0.72, metalness: 0, ...(LOOK[look] || {}) });
// The story's lights, fixed to the head's frame (`parent`: mesh axes, the face looking along −z), not to the camera: a warm KEY from the
// front, above and the far side from the camera (it models the face); a cool FILL from the camera's side at about a quarter of the key;
// a RIM from behind and above (a bright edge that separates the dark clothing and strap from a black page); a weak sky/ground light.
// The camera stands on the player's left (dataview.js CAM; the story's CAM_YAW +30° is also the player's left), so the story's
// "side" is −1 here. ref = the heading the body faces (radians); call again to move the lights when it changes.
// The reflection map (scene.environment): two soft white panels and a dim floor turned into an environment by three.js's
// PMREMGenerator, made once per renderer, so the glossy visor and shell have something to reflect.
export function lookLights(parent, renderer, scene, ref = 0) {
  let L = parent.userData.k8look;
  if (!L) {
    const aim = new THREE.Object3D(); parent.add(aim);
    const add = (color, power) => { const l = new THREE.DirectionalLight(color, power); l.target = aim; parent.add(l); return l; };
    L = parent.userData.k8look = { key: add(0xfff1e0, 2.6), fill: add(0xcfdcff, 0.65), rim: add(0xffffff, 2.2), ref: null };
    parent.add(new THREE.HemisphereLight(0xffffff, 0x15171b, 0.35));
    if (renderer && scene && !scene.environment) {
      const room = new THREE.Scene(), glow = (w, h, x, y, z, v) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(v, v, v), side: THREE.DoubleSide })); m.position.set(x, y, z); m.lookAt(0, 0, 0); room.add(m); };
      glow(6, 4, -5, 5, 4, 6); glow(5, 5, 6, 2, 3, 2.5); glow(20, 20, 0, -6, 0, 0.12);
      const pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(room, 0.04).texture; scene.environmentIntensity = 0.55; pm.dispose();
      room.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    }
  }
  if (L.ref !== ref) {   // story: at(f, u, r) = F·f + up·u + R·r·side, in head-heights (only the directions matter for these lights)
    const F = [-Math.sin(ref), 0, -Math.cos(ref)], R = [Math.cos(ref), 0, -Math.sin(ref)], side = -1;
    const at = (l, f, u, r) => l.position.set(F[0] * f + R[0] * r * side, u, F[2] * f + R[2] * r * side);
    at(L.key, 3, 3, -2); at(L.fill, 2, 0.5, 3); at(L.rim, -3, 2.6, 1); L.ref = ref;
  }
}

export function makeHead() {
  const H = window.K8_HEAD, rigPose = window.K8HeadPose && window.K8HeadPose.make(); if (!H || !rigPose) return null;
  const rig = new THREE.Group(); let skin = null, restN = null;
  const looks = [];   // F3: [mesh, the story's physical material(s), the cheaper clay material(s)], for setCheap()
  for (const p of H.parts) {
    const g = new THREE.BufferGeometry(), a = new Float32Array(p.p.length); for (let i = 0; i < a.length; i++) a[i] = p.p[i] * H.unit;
    g.setAttribute('position', new THREE.BufferAttribute(a, 3));
    if (p.group === 'skin') {
      g.setIndex(p.i.concat(p.i2)); g.addGroup(0, p.i.length, 0); g.addGroup(p.i.length, p.i2.length, 1); g.computeVertexNormals();
      restN = g.attributes.normal.array.slice();
      const ph = [physical('skin', LOOK.skin.color), physical('cloth', LOOK.cloth.color)];
      skin = new THREE.Mesh(g, ph); looks.push([skin, ph, [clay(p.color), clay(p.color2)]]);
      skin.frustumCulled = false;   // its vertices move, so its stored bounds would go stale
    } else {
      g.setIndex(p.i); g.computeVertexNormals();
      if (p.name === 'led') { rig.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: p.color }))); continue; }
      const ph = physical(p.name, (LOOK[p.name] || {}).color || p.color), m = new THREE.Mesh(g, ph);   // smooth normals: the bevels read as rounded edges
      rig.add(m); looks.push([m, ph, clay(p.color)]);
    }
  }
  // F3: where the pointer starts and how long it is (the story's rule, hero.js): from the headset's foremost point on the centre line,
  // at the height of the shell's middle, 1.5 head-lengths forward; head length = the front-to-back extent of the skin the head bone carries
  let zLo = Infinity, zHi = -Infinity;
  { const sk = H.parts.find(p => p.group === 'skin'); if (sk && sk.w) for (let k = 0; k < sk.w.length / 3; k++) if (sk.w[3 * k + 2] > 200) { const z = sk.p[3 * k + 2] * H.unit; if (z < zLo) zLo = z; if (z > zHi) zHi = z; } }
  const hsBox = new THREE.Box3(); rig.children.forEach(m => { m.geometry.computeBoundingBox(); hsBox.union(m.geometry.boundingBox); });
  const shell = rig.children.find((m, k) => H.parts.filter(p => p.group !== 'skin')[k].name === 'body') || rig.children[0];
  const front = new THREE.Vector3(0, (shell.geometry.boundingBox.min.y + shell.geometry.boundingBox.max.y) / 2, hsBox.min.z);
  const headLength = isFinite(zHi - zLo) ? zHi - zLo : 0.6, pointerLength = POINTER_HEADS * headLength;
  let cheapNow = false;
  const setCheap = on => { if (on === cheapNow) return; cheapNow = on; for (const [m, ph, cl] of looks) m.material = on ? cl : ph; };
  const last = new THREE.Quaternion(9, 9, 9, 9);
  // q = the head's rotation. headpose.js shares it between the neck bones and the head, moves every skin vertex, and says where the
  // head ended up; the headset (rig) is fixed to the head, so it gets that same turn and shift.
  let lastRef = null;
  function pose(q, ref = 0) {   // ref: the heading the body faces for this recording (radians), see headpose.js
    if (!skin || (last.equals(q) && lastRef === ref)) return; last.copy(q); lastRef = ref;
    const pos = skin.geometry.attributes.position, nor = skin.geometry.attributes.normal;
    const head = rigPose.apply([q.x, q.y, q.z, q.w], pos.array, restN, nor.array, ref);
    pos.needsUpdate = true; nor.needsUpdate = true;
    rig.quaternion.set(head.q[0], head.q[1], head.q[2], head.q[3]); rig.position.set(head.t[0], head.t[1], head.t[2]);
  }
  pose(new THREE.Quaternion());
  useHeadsetModel(rig);
  // front, headLength, pointerLength: in mesh units, in the headset's own (rest) axes. setCheap(true): the clay matcap instead of the
  // physical materials (for the lighter graphics tiers: no lights, one texture lookup)
  return { rig, skin, pose, front, headLength, pointerLength, setCheap };
}

// ---- the same head as a picture, for the Data side ----
// K8HeadGL.draw(ctx, { w, h, dpr, cx, cy, s, q, toView, scale }): renders the head for rotation q ([w, x, y, z] in the Data side's own axes:
// forward +z, up +y; null = no head, only the shoulders) and paints it into the 2D canvas context. cx, cy = where the head's centre goes,
// s = pixels per Data-side unit, toView = the Data side's own camera function, scale = Data-side units per mesh unit.
// One hidden WebGL canvas serves every head on the page. Returns false if WebGL could not start (the caller then draws its own head).
let GL = null;
function glInit() {
  if (GL !== null) return GL;
  try {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: true });
    renderer.setClearColor(0x000000, 0);
    // F3: the story's renderer settings: sRGB output, and NeutralToneMapping, which keeps bright highlights from clipping to flat white
    renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NeutralToneMapping; renderer.toneMappingExposure = 1;
    const scene = new THREE.Scene(), hd = makeHead(); if (!hd) return (GL = false);
    const root = new THREE.Group(), turn = hd.rig; root.matrixAutoUpdate = false; root.add(hd.rig, hd.skin); scene.add(root);
    lookLights(root, renderer, scene, 0);   // F3: in root, i.e. in the head's own (mesh) axes, so they stay fixed to the head when the camera orbits
    // (F3: T2's contact shadow under the shoulders is gone, by Nic's decision: no floor of any kind; the rim light grounds the head)
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200); cam.position.set(0, 0, 100);
    const pcam = new THREE.PerspectiveCamera();   // T3: used when the caller asks for perspective (o.persp); its projection is set by hand in draw()
    // the pointer (F3, the story's rule): one thin straight rod from the visor's front, 1.5 head-lengths long, fixed to the head
    // (mesh units; the face looks along −z). No ball at its end.
    const beamMat = new THREE.MeshBasicMaterial({ color: 0xffd27a, toneMapped: false }), beam = new THREE.Group();
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.0075, 0.0075, hd.pointerLength, 10), beamMat); rod.rotation.x = Math.PI / 2;
    rod.position.set(hd.front.x, hd.front.y, hd.front.z - hd.pointerLength / 2); beam.add(rod); hd.rig.add(beam);
    GL = { renderer, scene, hd, root, turn, cam, pcam, beam, beamMat, rod, q: new THREE.Quaternion(), key: '' };
  } catch (e) { GL = false; }
  return GL;
}
window.K8HeadGL = {
  draw(ctx, o) {
    const G = glInit(); if (!G) return false;
    const W = Math.round(o.w * o.dpr), Hh = Math.round(o.h * o.dpr);
    if (G.key !== W + 'x' + Hh) { G.renderer.setPixelRatio(1); G.renderer.setSize(W, Hh, false); G.key = W + 'x' + Hh; }
    // mesh axes -> Data-side axes (half a turn about "up", scaled) -> the Data side's camera -> three's view axes (z toward the viewer)
    const k = o.scale, c = [[-k, 0, 0], [0, k, 0], [0, 0, -k]].map(e => { const t = o.toView(e); return [t[0], t[1], -t[2]]; });
    G.root.matrix.set(c[0][0], c[1][0], c[2][0], 0, c[0][1], c[1][1], c[2][1], 0, c[0][2], c[1][2], c[2][2], 0, 0, 0, 0, 1); G.root.matrixWorldNeedsUpdate = true;
    // The visible window at the head's centre: cx, cy is where the centre lands, s pixels per Data-side unit.
    const L = -o.cx / o.s, R = (o.w - o.cx) / o.s, T = o.cy / o.s, B = -(o.h - o.cy) / o.s;
    let cam = G.cam;
    if (o.persp) {   // T3: gentle perspective, the eye o.persp units in front of the head's centre; the same window at the centre's depth,
      // so a point at view depth z lands at cx + s·x·D/(D + z): exactly the caller's own 2D projection (dataview.js PT)
      const D = o.persp, n = 0.5, f = D + 40, k = n / D; cam = G.pcam; cam.position.set(0, 0, D); cam.updateMatrixWorld();
      cam.projectionMatrix.makePerspective(L * k, R * k, T * k, B * k, n, f); cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    } else { cam.left = L; cam.right = R; cam.top = T; cam.bottom = B; cam.updateProjectionMatrix(); }
    G.hd.setCheap(!!o.cheap);   // F3: the clay matcap on the lighter tiers, the story's materials on "full"
    lookLights(G.root, null, null, o.ref || 0);
    const has = !!o.q; G.turn.visible = has; G.hd.skin.material[0].visible = has;   // [0]: the face and neck (the clothing stays)
    G.beam.visible = !!o.beam; if (o.beam) G.beamMat.color.set(o.beam);
    // F3 addendum 4: o.ptrLen (head-lengths, optional; default 1.5) lets the caller shorten the pointer so its tip stays in the canvas
    const len = (o.ptrLen || POINTER_HEADS) * G.hd.headLength;
    G.rod.scale.y = len / G.hd.pointerLength; G.rod.position.z = G.hd.front.z - len / 2;
    if (has) G.q.set(-o.q[1], o.q[2], -o.q[3], o.q[0]).normalize(); else G.q.identity();   // the same rotation, written in the mesh's axes
    G.hd.pose(G.q, o.ref || 0);
    G.renderer.render(G.scene, cam);
    ctx.drawImage(G.renderer.domElement, 0, 0, o.w, o.h);
    return true;
  },
  // T3: can the 3D head be drawn here (WebGL starts and the head model loaded)? Tried once, then remembered.
  ok() { return !!glInit(); },
  // F3: the pointer, for the caller's framing: its start (the visor's front) and its tip, in mesh units in the headset's rest axes
  // (face along −z); null if the 3D head cannot be drawn
  // (headLength: mesh units; the tip is at 1.5 head-lengths, the longest the pointer gets)
  pointer() { const G = glInit(); return G ? { front: G.hd.front.toArray(), tip: [G.hd.front.x, G.hd.front.y, G.hd.front.z - G.hd.pointerLength], headLength: G.hd.headLength } : null; },
};
