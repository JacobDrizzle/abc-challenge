// stage_task.js: deck 1's live stage, minimal (SPEC G1, G2, G4 "hook"; CONTRACTS section 3; ticket 04a).
// Loaded by the deck script as <script type="module" src="stage_task.js">; `three` comes through the page's import map.
//
// It owns the renderer, the camera and default lights on the deck's stage element (#taskStage), loads the head snapshot and the
// clip (hero/head.js, hero/headpose.js, hero/clip.js) as classic scripts, adds the hero part (hero.js) to its scene and runs
// the replay on a real-time clock while the deck is current. The scene around the hero (the dataset and window shots, the camera
// path along the deck) is scene_task.js's.
//
// The live-stage contract (CONTRACTS 3), as built:
//   mount(el)            canvas filling el; draws; sets data-live="on" on el after the first frame. Returns a promise.
//   setState(s)          'current': clock runs and frames are drawn; 'near': the last frame stays (one frame is drawn when the
//                        scroll position changes); 'far': nothing drawn.
//   setPos(s, key, extra) moves the camera along the scene's path and sets what is shown (scene_task.js); redraws once if the
//                        replay is not running (paused), so scrolling always shows the right picture.
//   release()            disposes everything and loses the WebGL context; mount may be called again afterwards. Also done by itself
//                        when the context is lost (webglcontextlost: the browser dropped it, e.g. after a GPU reset) and when mount
//                        fails part-way (e.g. head data the hero cannot be built from): data-live goes, so the deck's cards come back.
// Registered with window.K8Story.addLive('task', stage) when that exists; always also exposed as window.K8StageTask.
// Test hooks (CONTRACTS 7): window.__hero = { frames(), clock(), setClock(seconds), sample(), paused(), tips() }, and for ticket 04b's
// checks readoutSample(), probe(), probeAt(seconds).
// Sizes (review Q6): the stage's width and height are read only when it is resized (fit(), from the ResizeObserver, which runs after
// layout), never while drawing a frame, so a frame never makes the browser lay the page out again after the deck has written its styles.
// The readout (ticket 04b) is filled here, in draw(): see readout().
import * as THREE from 'three';
import { createHero } from './hero.js';
import { loadHeadsetModel } from './hero_model.js';   // tactical/: the real headset (Elin's CC BY Quest 3, logo removed)
import { createTaskScene } from './scene_task.js?r=f2';   // ?r=f2: version tag, see MODULES in story.js (the split labels changed on 13 Nov)

const BUST_FRACTION = 0.68;               // the bust's height as a fraction of the stage's height at s = 0 (ticket: 60 to 75%)
// Where the bust's silhouette centre sits across the stage (0 = left edge, 1 = right edge). The head faces left on screen, and
// its pointers and arc (ticket 04b) reach about half a stage width out in front of the face, so the bust stands right of
// centre: SPEC B6 "room on the side the head turns to most". Done with a view offset (a sideways shift of the picture), so the
// camera's direction, the perspective and the bust's height are unchanged.
const BUST_X = 0.72;
const FOV = 24;                           // degrees, vertical (SPEC B6 Camera)
const CAM_YAW = 30, CAM_PITCH = 8;        // degrees off the clip's mean facing direction, and above it (SPEC B6 Camera)
const PLAY_RATE = 0.25;                   // the replay runs at this fraction of real speed (Nic, 10 Nov: half, then quarter); the page says so beside the readout
const MAX_STEP = 0.25;                    // seconds: the most the clock may advance in one frame (SPEC B3)
const DIP_HALF = 0.125, DIP_DEPTH = 0.85; // the seam dip: about 250 ms in all, down to 15% opacity at the jump itself
const READOUT_MS = 250;                   // the readout changes at most 4 times a second (SPEC B4: "a few times a second")

const here = p => new URL(p, import.meta.url).href;
const DATA = [['K8_HERO_HEAD', 'hero/head.js'], ['K8HeroPose', 'hero/headpose.js'], ['K8_HERO_CLIP', 'hero/clip.js?r=q1']];   // clip.js was regenerated for the final model on 13 Nov, hence its version tag

// Insert a classic <script> and wait for it; skipped when its global is already there.
function loadScript(global, src) {
  if (window[global]) return Promise.resolve();
  return new Promise((ok, fail) => {
    const s = document.createElement('script'); s.src = here(src); s.async = false;
    s.onload = () => window[global] ? ok() : fail(new Error(src + ' did not define ' + global));
    s.onerror = () => fail(new Error('could not load ' + src));
    document.head.appendChild(s);
  });
}

// ---- state
let el = null, canvas = null, renderer = null, scene = null, camera = null, hero = null, ro = null, mounting = null;
let state = 'far', paused = false, ready = false, raf = 0, last = null;
let clock = 0, wrapped = false, frames = 0, posS = 0;
let part = null, labelsEl = null, offX = 0, once = 0;   // the scene around the head; its label layer; the hook's sideways shift in px; a pending single redraw
const cam = { position: new THREE.Vector3(), target: new THREE.Vector3(), fov: FOV, offset: 1 };
let size = [1, 1];                                   // the stage's [width, height] in CSS px, as last measured by fit()
let roAt = -Infinity, roK = -1, roEls = null;   // the readout: when it last changed, which clip sample it shows, its two elements
const drawn = { t: 0, frame: 0, posS: 0, clock: 0 };   // what the last drawn frame showed (test hook only)

function shouldRun() { return ready && state === 'current' && !paused && !document.hidden; }
function schedule() {
  if (shouldRun()) { if (!raf) raf = requestAnimationFrame(tick); }
  else { if (raf) cancelAnimationFrame(raf); raf = 0; last = null; }   // a stopped clock restarts without a jump
}

// The real-time clock. dt is real seconds since the last drawn frame, clamped to MAX_STEP so a stalled tab (or a long task)
// moves the replay on by a quarter of a second at most instead of skipping ahead. The loop seam: the clock stops exactly at the
// last recorded sample for one frame, and the next frame starts again exactly at sample 0, so the only jump is the recording's
// own 1.8° between its last and first sample (no blending, no pose that was not recorded). It costs at most one frame per loop.
function advance(dt) {
  const P = hero.period;
  if (clock >= P) { clock = 0; wrapped = true; return; }
  clock = Math.min(P, clock + dt * PLAY_RATE);
}
// Opacity of the live content near the seam: 1 far from it, down to 1 − DIP_DEPTH at the jump, over DIP_HALF either side.
function dip() {
  const P = hero.period, d = Math.min(P - clock, wrapped ? clock : Infinity);
  if (!(d < DIP_HALF)) return 1;
  const k = 0.5 + 0.5 * Math.cos(Math.PI * d / DIP_HALF);
  return 1 - DIP_DEPTH * k;
}
// The dip hides the head's jump at the loop seam, so it is applied only while the head is what the reader is looking at (the
// first three steps); from the dataset steps on the scene does not blink (SPEC G2: only data in time runs on the clock).
function setOpacity() {
  if (!canvas) return;
  const near = posS <= 2.2 ? 1 : posS >= 3 ? 0 : 1 - (posS - 2.2) / 0.8;
  canvas.style.opacity = (hero ? 1 - (1 - dip()) * near : 1).toFixed(3);
}

// The readout (#heroReadout in the first step's text column). At playback sample i the ghost shows the prediction made NOW for
// i + 20; the prediction that has just come due is the one made 20 samples (200 ms) ago, at i − 20, whose target is the pose on
// screen now. So the readout shows err[i − 20] (that prediction's error) and hold[i − 20] (the error of "nothing changes", i.e.
// keeping the pose of i − 20), straight from the clip file, rounded to 0.1°. No averages. It holds its last values while
// i − 20 would fall before the clip's start (the first 20 samples after the seam) and during the seam dip; it is not a live
// region (a number changing 4 times a second would flood a screen reader; #heroDesc describes the panel instead).
function readout(now, force) {
  if (!force && now - roAt < READOUT_MS) return;
  const C = window.K8_HERO_CLIP, k = hero.sampleIndex() - hero.horizon;
  if (k < 0 || dip() < 1) return;                              // hold
  roAt = now;
  if (k === roK) return;
  if (!roEls) { const e = document.querySelector('#heroReadout [data-r="err"]'), h = document.querySelector('#heroReadout [data-r="hold"]'); if (!e || !h) return; roEls = [e, h]; }
  roK = k;
  roEls[0].textContent = C.err[k].toFixed(1) + '°';
  roEls[1].textContent = C.hold[k].toFixed(1) + '°';
}

// The camera for the current scroll position: the scene gives a place, a point to look at and a field of view for every s.
function aimCamera() {
  const [w, h] = size;
  part.camera(posS, cam);
  camera.position.copy(cam.position); camera.lookAt(cam.target);
  camera.fov = cam.fov; camera.setViewOffset(w, h, offX * cam.offset, 0, w, h);   // also updates the projection
  camera.updateMatrixWorld(true);
  return [w, h];
}
function draw() {
  hero.update(clock);
  part.update(hero.fraction());
  readout(performance.now(), false);
  setOpacity();
  const [w, h] = aimCamera();
  renderer.render(scene, camera);
  part.placeLabels(camera, w, h);
  frames++;
  drawn.t = performance.now(); drawn.frame = frames; drawn.posS = posS; drawn.clock = clock;   // for the read-only test hook __hero.view()
  if (frames === 1 && el) el.setAttribute('data-live', 'on');
}
function tick(now) {
  raf = 0;
  if (!shouldRun()) { last = null; return; }
  const dt = last === null ? 0 : Math.min(MAX_STEP, Math.max(0, (now - last) / 1000)); last = now;
  advance(dt);
  draw();
  raf = requestAnimationFrame(tick);
}

// ---- camera: fixed, looking at the bust from 30° off the clip's mean facing direction and 8° above it, at the distance that
// makes the bust's silhouette (every vertex projected) BUST_FRACTION of the stage's height.
const _v = new THREE.Vector3();
function silhouette() {
  let lo = Infinity, hi = -Infinity, left = Infinity, right = -Infinity;
  scene.updateMatrixWorld(true); camera.updateMatrixWorld(true);
  for (const m of [hero.skin, ...hero.headset.children.filter(o => o.isMesh)]) {   // the bust itself: not its pointer (ticket 04b)
    const a = m.geometry.attributes.position.array;
    for (let i = 0; i < a.length; i += 3) { _v.set(a[i], a[i + 1], a[i + 2]).applyMatrix4(m.matrixWorld).project(camera); if (_v.y < lo) lo = _v.y; if (_v.y > hi) hi = _v.y; if (_v.x < left) left = _v.x; if (_v.x > right) right = _v.x; }
  }
  return { lo, hi, left, right };     // in normalised device coordinates (−1 bottom … 1 top, −1 left … 1 right)
}
function fit() {
  const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
  size = [w, h];
  renderer.setSize(w, h, false);
  camera.fov = FOV; camera.aspect = w / h; camera.clearViewOffset(); camera.updateProjectionMatrix();
  // the bust's own box: skin and headset meshes only (the ghost, pointers and arc of ticket 04b reach out in front of the face)
  const box = new THREE.Box3(); for (const m of [hero.skin, ...hero.headset.children.filter(o => o.isMesh)]) box.expandByObject(m, true);
  const c = box.getCenter(new THREE.Vector3());
  const yaw = hero.bodyRef + CAM_YAW * Math.PI / 180, pitch = CAM_PITCH * Math.PI / 180;
  // the face looks along −z, turned by the body's heading about +y: direction (−sin a, 0, −cos a)
  const dir = new THREE.Vector3(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
  let d = 6;
  for (let k = 0; k < 6; k++) {         // projected height is close to inversely proportional to distance: a few rounds settle it
    camera.position.copy(c).addScaledVector(dir, d); camera.lookAt(c); camera.updateMatrixWorld(true);
    const s = silhouette(), frac = (s.hi - s.lo) / 2;
    d *= frac / BUST_FRACTION;
    // keep the silhouette centred vertically: aim at the point halfway between its projected top and bottom
    if (k === 2) { const mid = (s.hi + s.lo) / 2; c.y += mid * Math.tan(FOV * Math.PI / 360) * d; }
  }
  camera.position.copy(c).addScaledVector(dir, d); camera.lookAt(c); camera.updateMatrixWorld(true);
  // shift the picture sideways so the silhouette's centre lands at BUST_X of the width (a negative x offset moves content right)
  const s = silhouette(), cx = ((s.left + s.right) / 2 + 1) / 2 * w;
  offX = cx - BUST_X * w;
  camera.setViewOffset(w, h, offX, 0, w, h);
  if (part) part.setKey0(camera.position, c, FOV);                 // the hook is the first place on the scene's camera path
}
// The frame the scene is laid out in: the bust's centre, its height (the scene's unit), the direction the player faces on
// average, and the side the hook's camera stands on (so later shots stay on that side and the camera never crosses the head).
function frame() {
  const box = new THREE.Box3(); for (const m of [hero.skin, ...hero.headset.children.filter(o => o.isMesh)]) box.expandByObject(m, true);
  const origin = box.getCenter(new THREE.Vector3()), U = box.max.y - box.min.y, a = hero.bodyRef;
  const F = new THREE.Vector3(-Math.sin(a), 0, -Math.cos(a)), R = new THREE.Vector3(Math.cos(a), 0, -Math.sin(a));
  const yaw = a + CAM_YAW * Math.PI / 180, dir = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
  return { origin, U, F, R, side: dir.dot(R) >= 0 ? 1 : -1 };
}

// ---- light (ticket 05, SPEC B6 Light). The lights stand in the scene, not on the camera, so the head is lit the same way from
// wherever the camera looks at it: a warm KEY from the front, above and to the far side of the camera (it models the face); a cool
// FILL from the camera's side at about a quarter of the key (it lifts the shadows); a RIM from behind and above (it draws a bright
// edge that separates the dark clothing and the strap from the black page). An ENVIRONMENT gives the glossy visor and shell
// something to reflect: it is made in code from a tiny scene of two soft white panels and a dim floor, turned into a reflection
// map by three.js's PMREMGenerator (no image file, no extra library).
function light({ origin, U, F, R, side }) {
  const at = (f, u, r) => origin.clone().addScaledVector(F, f * U).addScaledVector(new THREE.Vector3(0, 1, 0), u * U).addScaledVector(R, r * U * side);
  const aim = new THREE.Object3D(); aim.position.copy(origin); scene.add(aim);
  const add = (color, power, pos) => { const l = new THREE.DirectionalLight(color, power); l.position.copy(pos); l.target = aim; scene.add(l); return l; };
  add(0xfff1e0, 2.6, at(3, 3, -2));
  add(0xcfdcff, 0.65, at(2, 0.5, 3));
  add(0xffffff, 2.2, at(-3, 2.6, 1));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x15171b, 0.35));
  const room = new THREE.Scene(), glow = (w, h, x, y, z, v) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(v, v, v), side: THREE.DoubleSide })); m.position.set(x, y, z); m.lookAt(0, 0, 0); room.add(m); };
  glow(6, 4, -5, 5, 4, 6); glow(5, 5, 6, 2, 3, 2.5); glow(20, 20, 0, -6, 0, 0.12);
  const pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(room, 0.04).texture; scene.environmentIntensity = 0.55; pm.dispose();
  room.traverse(o => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
}

// ---- the contract
async function mount(target) {
  if (mounting) return mounting;
  if (window.K8_BOOT && window.K8_BOOT.live3d === false) return;
  el = target;
  mounting = (async () => {
    // Test WebGL2 on a bare canvas first: three.js logs an error when it cannot get a context, and a failed start must stay
    // silent (the deck keeps showing its pictures).
    const cv = document.createElement('canvas'), gl = cv.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true });
    if (!gl) return;
    // Everything after the test can fail part-way (a data file missing, head data the hero cannot be built from, three.js unable to
    // start). Whatever was made by then is freed by release() in the catch, the WebGL context included, and the mount ends quietly:
    // no error in the console, no rejected promise left unhandled, and the deck keeps its cards. mounting is cleared too, so a
    // later mount may try again.
    try {
      for (const [g, s] of DATA) await loadScript(g, s);
      renderer = new THREE.WebGLRenderer({ canvas: cv, context: gl, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.NeutralToneMapping; renderer.toneMappingExposure = 1;   // keeps bright highlights from clipping to flat white, without shifting colours
      renderer.setClearColor(0x000000, 0);                     // transparent: the page shows through
      canvas = renderer.domElement;
      canvas.setAttribute('tabindex', '-1'); canvas.setAttribute('aria-hidden', 'true');
      Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', display: 'block', pointerEvents: 'none' });
      // A lost context (the browser can drop one at any time, e.g. after a GPU reset or when too many are open) leaves nothing to draw
      // with: free everything (the context is already gone, so it is not lost again) and let the deck show its cards.
      canvas.addEventListener('webglcontextlost', onLost);
      if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 100);
      scene.add(camera);
      hero = createHero({ THREE, clip: window.K8_HERO_CLIP, head: window.K8_HERO_HEAD, pose: window.K8HeroPose });
      hero.setPaused(paused);
      scene.add(hero.group);
      // the real headset replaces the box one as soon as it has loaded (the box shows meanwhile; a failed load keeps the box)
      loadHeadsetModel(THREE).then(m => { if (hero) { hero.useModel(m); if (ready) draw(); } }).catch(e => console.warn('headset model:', e));
      el.appendChild(canvas);
      labelsEl = document.createElement('div'); labelsEl.className = 'sc-labels'; el.appendChild(labelsEl);
      labelsEl.setAttribute('aria-hidden', 'true');   // ticket 09: the whole label layer stays out of the accessibility tree (its figures are in the text)
      const fr = frame();
      camera.far = 80 * fr.U; camera.near = 0.02 * fr.U;
      light(fr);
      part = createTaskScene({ THREE, clip: window.K8_HERO_CLIP, ...fr, accent: hero.accent, labelsEl });
      scene.add(part.group); part.setPos(posS);
      fit();
      ro = new ResizeObserver(() => { if (!renderer) return; fit(); if (state !== 'far' && ready) draw(); });
      ro.observe(el);
      ready = true;
      draw();                                                  // the first frame, whatever the state: the stage must show the bust
      schedule();
    } catch (e) {
      if (renderer) release();
      else { const x = gl.getExtension('WEBGL_lose_context'); if (x) x.loseContext(); mounting = null; }   // only the test context exists
    }
  })();
  return mounting;
}
function onLost() { release(true); }
function setState(s) { state = s === 'current' || s === 'near' ? s : 'far'; schedule(); }
function setPos(s, key, extra) {
  posS = +s || 0;
  if (!ready) return;
  part.setPos(posS);
  // While the replay runs, the next frame shows the new position anyway. When it does not (paused, or the deck is only near),
  // draw one frame so the picture always matches the scroll position.
  if (!raf && !once && state !== 'far') once = requestAnimationFrame(() => { once = 0; if (ready && state !== 'far') draw(); });
}
// lost = true: the context is already lost (webglcontextlost), so it is not lost a second time (that would be a WebGL error).
function release(lost) {
  if (canvas) canvas.removeEventListener('webglcontextlost', onLost);   // forceContextLoss below fires the event too
  if (raf) cancelAnimationFrame(raf); raf = 0; last = null; ready = false;
  if (ro) ro.disconnect(); ro = null;
  if (once) cancelAnimationFrame(once); once = 0;
  if (part) part.dispose(); part = null; if (labelsEl) labelsEl.remove(); labelsEl = null;
  if (hero) hero.dispose(); hero = null;
  if (renderer) { renderer.dispose(); if (lost !== true) renderer.forceContextLoss(); }
  if (canvas) canvas.remove();
  if (el) el.removeAttribute('data-live');
  renderer = canvas = scene = camera = null; mounting = null; frames = 0; clock = 0; wrapped = false; roAt = -Infinity; roK = -1; roEls = null;
}
function setPaused(p) {
  paused = !!p; if (hero) hero.setPaused(paused);
  const b = document.getElementById('heroPause'); if (b) b.setAttribute('aria-pressed', paused ? 'true' : 'false');
  schedule();
}

// #heroPause (a real button in the first step's text column, CONTRACTS 4): toggles the replay. Delegated, so it works whether
// the button is in the markup before or after this module runs.
document.addEventListener('click', e => { const b = e.target && e.target.closest && e.target.closest('#heroPause'); if (b) setPaused(!paused); });
{ const b = document.getElementById('heroPause'); if (b) b.setAttribute('aria-pressed', 'false'); }
document.addEventListener('visibilitychange', schedule);       // a hidden tab draws nothing and its clock stands still

export const stage = { mount, setState, setPos, release };
window.K8StageTask = stage;
window.__hero = {
  frames: () => frames,
  clock: () => clock,
  // (ticket 04b: also lets the readout change on the next drawn frame, so a check can set a time and read the result)
  setClock(sec) { const P = hero ? hero.period : 11.99; let t = +sec || 0; t %= P; if (t < 0) t += P; clock = t; wrapped = false; roAt = -Infinity; },
  sample: () => (hero ? Math.floor(Math.min(hero.samples - 1, clock * hero.rate)) : 0),
  paused: () => paused,
  rate: PLAY_RATE,
  // the two pointer tips (true, ghost) on screen, in CSS pixels of the window (as drawn last)
  tips: () => (hero && camera ? hero.measure().tips.map(toScreen) : null),
  // ticket 04b, for the checks: the clip sample the readout shows (−1 = none yet), and the hero part's measure() as plain numbers
  // (headset and ghost world quaternions [x,y,z,w], arc angle, tips in CSS px). probeAt(sec) poses the part at that clock time
  // first, also while paused, and draws nothing; the clock itself is not changed. While the replay runs, the next drawn frame puts
  // the replay's pose back; while it is paused, hero.update() does nothing, so the probed pose stays until the replay runs again.
  readoutSample: () => roK,
  probe: () => probe(),
  probeAt(sec) { if (!hero) return null; hero.poseAt(+sec || 0); return probe(); },
  // read only: the last drawn frame's time, scroll position and clock, and the camera it was drawn with (world position, target,
  // field of view, the share of the hook's sideways shift); nothing is computed or changed
  view: () => (camera ? { ...drawn, pos: cam.position.toArray(), target: cam.target.toArray(), fov: cam.fov, offset: cam.offset, offX } : null),
};
function probe() {
  if (!hero) return null;
  const m = hero.measure();
  return { f: m.f, headset: m.headset, ghost: m.ghost, arcDeg: m.arcDeg, arcDegPosed: m.arcDegPosed, gap: m.gap, tips: camera ? m.tips.map(toScreen) : null,
    radius: m.radius, headLength: m.headLength, pointerLength: m.pointerLength };
}
// world position -> [x, y] in CSS pixels of the window, through the stage's camera and the canvas's place on the page
function toScreen(v) {
  camera.updateMatrixWorld(true);
  const p = v.clone().project(camera), r = canvas.getBoundingClientRect();
  return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height];
}
if (window.K8Story && typeof window.K8Story.addLive === 'function') window.K8Story.addLive('task', stage);
